"""回归：证据面板与上下文用量上报（纯单元，不依赖 vLLM/Milvus）。

守护三件事：
① 工具未命中（not_found）/重复跳过（duplicate_skipped）时，
   tool_call_completed 事件携带 not_found 标志——前端据此不把
   写给模型的内部兜底指令（"【知识库未收录】…"）渲染成"证据来源"；
② search_knowledge 检索结果为空的分支同样标记 not_found
   （此前只有低相关弃权分支带标志，空结果文案会漏进证据面板）；
③ 视觉链路不走 AgentLoop，须自行补发带 prompt_tokens 的
   llm_response 事件，否则前端 ctx 徽标在影像轮永不更新。
"""
import asyncio
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

from meditriage.core.agent_loop import AgentLoop
from meditriage.core.llm_client import LLMResponse, ToolCall


# ---------- ①②：tool_call_completed 的 not_found 标志 ----------

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


def _run_tool_calls(tool_result, calls=None):
    loop = AgentLoop(max_iterations=5)
    loop.validator = None
    loop._called_signatures = set()
    events = []

    def emit(etype, data):
        events.append((etype, data))

    llm_response = LLMResponse(
        content=None,
        tool_calls=calls or [ToolCall(
            id="t1", name="search_knowledge", arguments={"query": "x"})],
        finish_reason="tool_calls",
    )
    asyncio.run(loop._handle_tool_calls(
        _FakeAgent(tool_result), llm_response, [], None,
        SimpleNamespace(iteration=1), emit,
    ))
    return [d for t, d in events if t == "tool_call_completed"]


def test_not_found_result_flagged_in_event():
    done = _run_tool_calls({
        "answer": "【知识库未收录】检索到的内容相关性都很低…",
        "not_found": True,
    })
    assert done and done[0].get("not_found") is True
    assert "citations" not in done[0]


def test_normal_result_not_flagged():
    done = _run_tool_calls({
        "answer": "高血压患者的降压目标为140/90mmHg以下。",
        "citations": [{"label": "ESC 2024", "excerpt": "…"}],
    })
    assert done and "not_found" not in done[0]
    assert done[0]["citations"]


def test_duplicate_skipped_flagged_in_event():
    same = [
        ToolCall(id="t1", name="search_knowledge", arguments={"query": "x"}),
        ToolCall(id="t2", name="search_knowledge", arguments={"query": "x"}),
    ]
    done = _run_tool_calls(
        {"answer": "正常结果", "citations": [{"label": "L"}]}, calls=same)
    assert len(done) == 2
    assert "not_found" not in done[0]          # 首次真实执行
    assert done[1].get("not_found") is True    # 重复跳过的回灌提示非证据


def _load_search_skill():
    skill_dir = (Path(__file__).resolve().parents[1] / "meditriage" / "skills"
                 / "search-knowledge" / "script")
    sys.path.insert(0, str(skill_dir))
    try:
        return import_module("search")
    finally:
        sys.path.remove(str(skill_dir))


def test_search_knowledge_empty_results_flagged_not_found(monkeypatch, tmp_path):
    from meditriage.knowledge import badcase
    monkeypatch.setattr(badcase, "LOG_DIR", tmp_path)  # 不污染真实 badcase 日志
    search = _load_search_skill()

    class _EmptyKB:
        def search(self, query, top_k, filter_type=None):
            return []

    monkeypatch.setattr(search, "get_knowledge_base", lambda: _EmptyKB())
    out = asyncio.run(search.search_knowledge("不存在的主题"))
    assert out.get("not_found") is True
    assert out["total_found"] == 0


# ---------- ③：视觉链路补发 llm_response 用量事件 ----------

class _FakeVisionClient:
    def __init__(self):
        self.last_usage = None

    async def chat(self, messages, max_tokens=None):
        self.last_usage = SimpleNamespace(
            prompt_tokens=1234, total_tokens=1300)
        return "影像未见明显异常。"


def test_vision_turn_reports_prompt_tokens(monkeypatch):
    from meditriage.core import vision_handler
    monkeypatch.setattr(vision_handler, "LLMClient", _FakeVisionClient)
    events = []
    out = asyncio.run(vision_handler.process_image_query(
        "data:image/png;base64,xxx", "这张影像有什么异常？",
        event_emitter=lambda t, d: events.append((t, d)),
    ))
    assert out["answer"]
    usage = [d for t, d in events if t == "llm_response"]
    assert usage and usage[0]["prompt_tokens"] == 1234
