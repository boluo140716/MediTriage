"""回归：badcase 沉淀 / 工具失败重规划 / 上下文溢出加固（纯单元）。

守护六件事：
① log_rag_miss 结构化落盘且永不抛错；search_knowledge 的低相关
   兜底分支真的会留痕（线上 miss 不再只有兜底话术、无处可查）；
② 工具执行失败（success=False）后，循环会回灌一条重规划引导，
   而不是只把裸 JSON 丢给模型；
③ 上下文超限报错的两种 vLLM 措辞（input tokens / characters）都能识别；
④ 溢出自救裁剪只截中间观察结果，不动 system 与最后一条 user 消息；
⑤ 同一 loop 被并发复用（同 worker 领多个子任务）时，工具预算与
   去重守卫按 run 隔离，不跨子任务串扰；
⑥ finalize 阶段抛出上下文超限时仍走强制总结，不静默返回空结果。
"""
import asyncio
import json
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

from meditriage.core.agent_loop import AgentLoop
from meditriage.core.llm_client import LLMClient, LLMResponse, ToolCall
from meditriage.knowledge import badcase


# ---------- ①：badcase 被动沉淀 ----------

def test_log_rag_miss_writes_structured_record(tmp_path):
    badcase.log_rag_miss(
        "罕见病基因治疗", [{"metadata": {"doc_id": "d1"}, "score": 0.12}],
        route="search", reason="low_relevance", base_dir=tmp_path,
    )
    lines = (tmp_path / "rag_misses.jsonl").read_text(
        encoding="utf-8").strip().splitlines()
    rec = json.loads(lines[0])
    assert rec["reason"] == "low_relevance"
    assert rec["route"] == "search"
    assert rec["top"][0] == {"doc_id": "d1", "score": 0.12}


def test_log_rag_miss_never_raises(tmp_path):
    # results 形态异常也不能反噬主流程
    badcase.log_rag_miss("q", [{"score": "garbage"}], base_dir=tmp_path)
    badcase.log_rag_miss("q", None, base_dir=Path("/proc/nonwritable"))


def _load_search_skill():
    skill_dir = (Path(__file__).resolve().parents[1] / "meditriage" / "skills"
                 / "search-knowledge" / "script")
    sys.path.insert(0, str(skill_dir))
    try:
        return import_module("search")
    finally:
        sys.path.remove(str(skill_dir))


def test_search_low_relevance_leaves_badcase_trace(tmp_path, monkeypatch):
    search = _load_search_skill()

    class _WeakKB:
        def search(self, query, top_k, filter_type=None):
            return [{"id": "x", "content": "弱相关", "metadata": {},
                     "score": 0.05}]

    monkeypatch.setattr(search, "get_knowledge_base", lambda: _WeakKB())
    monkeypatch.setattr(badcase, "LOG_DIR", tmp_path)
    out = asyncio.run(search.search_knowledge("覆盖外主题"))
    assert out.get("not_found") is True
    trace = tmp_path / "badcase" / "rag_misses.jsonl"
    assert trace.is_file()
    rec = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
    assert rec["reason"] == "low_relevance" and rec["n_hits"] == 1


# ---------- ②：工具失败 → 重规划引导 ----------

class _FakeLLMClient:
    def create_tool_message(self, tool_call_id, tool_name, result):
        return {"role": "tool", "tool_call_id": tool_call_id,
                "content": str(result)}


class _FakeAgent:
    agent_id = "test_agent"

    def __init__(self, tool_result):
        self._tool_result = tool_result
        self.llm_client = _FakeLLMClient()

    async def execute_tool(self, tool_name, arguments):
        return self._tool_result


def _run_tool_calls(tool_result):
    loop = AgentLoop(max_iterations=5)
    loop.validator = None
    loop._called_signatures = set()
    messages = []
    llm_response = LLMResponse(
        content=None,
        tool_calls=[ToolCall(
            id="t1", name="search_knowledge", arguments={"query": "x"})],
        finish_reason="tool_calls",
    )
    asyncio.run(loop._handle_tool_calls(
        _FakeAgent(tool_result), llm_response, messages, None,
        SimpleNamespace(iteration=1), lambda t, d: None,
    ))
    return messages


def test_failed_tool_triggers_replan_guidance():
    messages = _run_tool_calls(
        {"success": False, "error": "Skill execution failed"})
    assert messages[-1]["role"] == "user"
    assert "请勿以相同参数重试" in messages[-1]["content"]


def test_successful_tool_no_replan_guidance():
    messages = _run_tool_calls({"answer": "正常结果"})
    assert messages[-1]["role"] == "tool"


# ---------- ⑤/⑥：并发 run 隔离 + finalize 溢出仍强制总结 ----------

class _ScriptedLLMClient:
    """先返回一次 tool_calls，再返回文本答复的脚本化客户端。"""

    def __init__(self):
        self.calls = 0

    async def chat_with_tools(self, messages, tools=None, **kwargs):
        self.calls += 1
        # 无 tools 的调用是 _force_summary 的总结请求
        if tools is None or self.calls > 1:
            return LLMResponse(
                content="最终答复", tool_calls=[], finish_reason="stop")
        return LLMResponse(
            content=None,
            tool_calls=[ToolCall(
                id=f"t{self.calls}", name="search_knowledge",
                arguments={"query": "同一查询"})],
            finish_reason="tool_calls",
        )

    def create_tool_message(self, tool_call_id, tool_name, result):
        return {"role": "tool", "tool_call_id": tool_call_id,
                "content": str(result)}

    def is_ctx_overflow(self, e):
        return "maximum context length" in str(e)


class _LoopAgent:
    """走完整 loop.run 的最小 Agent 桩。"""
    agent_id = "test_agent"
    config = {"temperature": 0.0}

    def __init__(self, executed):
        self.llm_client = _ScriptedLLMClient()
        self._executed = executed

    def get_system_prompt(self):
        return "你是测试 Agent"

    def get_tools_for_llm(self):
        return [{"type": "function",
                 "function": {"name": "search_knowledge"}}]

    def format_user_input(self, input_data):
        return input_data.get("question", "q")

    async def execute_tool(self, tool_name, arguments):
        await asyncio.sleep(0.05)   # 制造并发时间窗重叠
        self._executed.append(tool_name)
        return {"answer": "检索结果"}


def test_concurrent_runs_isolated_budget_and_dedup():
    """同一 loop 并发跑两个含相同 (工具,参数) 调用的 run：
    去重与预算按 run 隔离，两次都真实执行（修复前仅执行 1 次）。"""
    executed = []
    loop = AgentLoop(max_iterations=3)
    loop.validator = None

    async def both():
        a1 = _LoopAgent(executed)
        a2 = _LoopAgent(executed)
        return await asyncio.gather(
            loop.run(a1, {"question": "子任务A"}),
            loop.run(a2, {"question": "子任务B"}),
        )

    r1, r2 = asyncio.run(both())
    assert executed.count("search_knowledge") == 2
    assert r1.get("answer") and r2.get("answer")


class _OverflowOnFinalizeAgent(_LoopAgent):
    """post_process_result 抛出形如上下文超限的异常，复现 finalize 溢出。"""

    async def post_process_result(self, result, final_response):
        raise RuntimeError(
            "This model's maximum context length is 8192 tokens ...")


def test_finalize_overflow_still_force_summarizes():
    """finalize 阶段溢出：必须走强制总结产出非空答案，而非静默空 {}。"""
    loop = AgentLoop(max_iterations=3)
    loop.validator = None
    agent = _OverflowOnFinalizeAgent([])
    result = asyncio.run(loop.run(agent, {"question": "q"}))
    assert result.get("answer")
    assert result.get("warning") == "max_iterations_reached"


# ---------- ③：溢出报错两种措辞都识别 ----------

_TOKEN_VARIANT = (
    "This model's maximum context length is 8192 tokens. However, you "
    "requested 2048 output tokens and your prompt contains at least 6145 "
    "input tokens, for a total of at least 8193 tokens."
)
_CHAR_VARIANT = (
    "This model's maximum context length is 8192 tokens. However, you "
    "requested 8192 output tokens and your prompt contains 2736 characters "
    "(more than 0 characters, which is the upper bound for 0 input tokens)."
)


def test_ctx_overflow_detects_both_variants():
    client = object.__new__(LLMClient)
    assert client.is_ctx_overflow(Exception(_TOKEN_VARIANT)) is True
    assert client.is_ctx_overflow(Exception(_CHAR_VARIANT)) is True
    assert client.is_ctx_overflow(Exception("connection refused")) is False


# ---------- ④：溢出自救裁剪 ----------

def test_shrink_preserves_system_and_last_user():
    long = "证据" * 1000
    messages = [
        {"role": "system", "content": "系统提示" * 200},
        {"role": "user", "content": "第一轮问题"},
        {"role": "assistant", "content": long},
        {"role": "tool", "content": long},
        {"role": "user", "content": "最后的问题" * 200},
    ]
    sys_len = len(messages[0]["content"])
    last_len = len(messages[-1]["content"])
    AgentLoop._shrink_messages_for_overflow(messages, keep_chars=400)
    assert len(messages[0]["content"]) == sys_len          # system 不动
    assert len(messages[-1]["content"]) == last_len        # 最后 user 不动
    assert len(messages[2]["content"]) <= 400 + 10         # 中间被截
    assert messages[3]["content"].endswith("…（超长截断）")
