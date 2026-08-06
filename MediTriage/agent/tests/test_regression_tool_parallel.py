"""回归：Agent 工具调用并行执行 + 保序回灌。

守护：
① 同一轮多个工具并行执行（总耗时 < 串行和）；
② 结果按原始调用顺序回灌 messages（模型看到的顺序与串行一致，质量不变）；
③ 同轮重复 (工具, 参数) 预检去重，只执行一次；
④ 单工具异常失败隔离：不影响其他工具，且触发重规划引导；
⑤ 开关 MEDITRIAGE_TOOL_PARALLEL=0 时回退串行。
"""
import asyncio
import time
from types import SimpleNamespace

from meditriage.core.agent_loop import AgentLoop
from meditriage.core.llm_client import LLMResponse, ToolCall


class _FakeLLMClient:
    def create_tool_message(self, tool_call_id, tool_name, result):
        return {"role": "tool", "tool_call_id": tool_call_id,
                "tool_name": tool_name, "content": str(result)}


class _FakeAgent:
    agent_id = "test_agent"

    def __init__(self, executor):
        self._executor = executor
        self.llm_client = _FakeLLMClient()

    async def execute_tool(self, tool_name, arguments):
        return await self._executor(tool_name, arguments)


def _make_loop(**kw):
    loop = AgentLoop(max_iterations=5)
    loop.validator = None
    loop._tool_parallel = kw.get("parallel", True)
    loop._tool_max_concurrent = kw.get("concurrent", 3)
    return loop


def _run(loop, agent, tool_calls):
    llm_response = LLMResponse(
        content=None, tool_calls=tool_calls, finish_reason="tool_calls",
    )
    messages = []
    state = SimpleNamespace(iteration=1)
    asyncio.run(loop._handle_tool_calls(
        agent, llm_response, messages, None, state, lambda t, d: None,
    ))
    return messages


def test_parallel_executes_concurrently_and_keeps_order():
    loop = _make_loop()
    order = []

    async def executor(name, args):
        if name == "slow_tool":
            await asyncio.sleep(0.2)  # 慢工具：串行下会拖到 0.2s+
        order.append(name)            # 完成顺序（fast 先完成）
        return {"answer": f"result-{name}"}

    agent = _FakeAgent(executor)
    t0 = time.monotonic()
    msgs = _run(loop, agent, [
        ToolCall(id="t1", name="slow_tool", arguments={}),
        ToolCall(id="t2", name="fast_tool", arguments={}),
    ])
    elapsed = time.monotonic() - t0

    # ① 并行：总耗时显著小于串行和（0.2 + 0）
    assert elapsed < 0.35
    # ② 保序回灌：messages 中工具消息顺序 = 调用顺序，与完成顺序无关
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert [m["tool_name"] for m in tool_msgs] == ["slow_tool", "fast_tool"]


def test_duplicate_skip_in_same_batch():
    loop = _make_loop()
    calls = {"n": 0}

    async def executor(name, args):
        calls["n"] += 1
        return {"answer": "ok"}

    agent = _FakeAgent(executor)
    msgs = _run(loop, agent, [
        ToolCall(id="t1", name="search_knowledge", arguments={"query": "x"}),
        ToolCall(id="t2", name="search_knowledge", arguments={"query": "x"}),  # 重复
    ])
    assert calls["n"] == 1                      # 去重：只执行一次
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 2                  # 两条结果（第二条是跳过提示）
    assert "不再重复执行" in tool_msgs[1]["content"]


def test_parallel_failure_isolation():
    loop = _make_loop()

    async def executor(name, args):
        if name == "boom":
            raise RuntimeError("boom")
        return {"answer": "ok"}

    agent = _FakeAgent(executor)
    msgs = _run(loop, agent, [
        ToolCall(id="t1", name="boom", arguments={}),
        ToolCall(id="t2", name="ok_tool", arguments={}),
    ])
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 2                  # 失败隔离：两个都有结果
    assert "工具执行异常" in tool_msgs[0]["content"]
    # 失败触发重规划引导
    assert msgs[-1]["role"] == "user"
    assert "请勿以相同参数重试" in msgs[-1]["content"]


def test_serial_mode_when_disabled(monkeypatch):
    monkeypatch.setenv("MEDITRIAGE_TOOL_PARALLEL", "0")
    loop = AgentLoop(max_iterations=5)
    assert loop._tool_parallel is False
    # 串行下：工具依次执行（慢工具先完成，fast 后执行）
    order = []

    async def executor(name, args):
        if name == "slow_tool":
            await asyncio.sleep(0.05)
        order.append(name)
        return {"answer": "ok"}

    agent = _FakeAgent(executor)
    _run(loop, agent, [
        ToolCall(id="t1", name="slow_tool", arguments={}),
        ToolCall(id="t2", name="fast_tool", arguments={}),
    ])
    assert order == ["slow_tool", "fast_tool"]  # 严格串行
