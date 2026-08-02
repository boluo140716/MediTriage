"""回归：LLM 客户端参数与解析健壮性（纯单元，不调 vLLM）。

守护三件事：
① 显式 temperature=0（LeadAgent 确定性路由）不被 `or` 吞成默认 0.7；
② 单个 tool call 的参数 JSON 畸形只跳过该 call，不作废整轮响应；
③ content=None 不崩溃（某些 finish_reason 下合法出现）。
"""
import asyncio

from meditriage.core.llm_client import LLMClient


class _Func:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _TC:
    def __init__(self, id_, name, arguments):
        self.id = id_
        self.function = _Func(name, arguments)


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, msg, finish_reason="stop"):
        self.message = msg
        self.finish_reason = finish_reason


class _Resp:
    def __init__(self, msg, finish_reason="stop"):
        self.choices = [_Choice(msg, finish_reason)]
        self.usage = None


def _client_with_fake(monkeypatch, resp, captured):
    c = LLMClient()

    async def fake(params):
        captured.update(params)
        return resp

    monkeypatch.setattr(c, "_create_with_overflow_guard", fake)
    return c


def test_temperature_zero_honored(monkeypatch):
    captured = {}
    c = _client_with_fake(monkeypatch, _Resp(_Msg(content="ok")), captured)
    asyncio.run(c.chat([{"role": "user", "content": "hi"}], temperature=0))
    assert captured["temperature"] == 0


def test_temperature_zero_honored_with_tools(monkeypatch):
    captured = {}
    c = _client_with_fake(monkeypatch, _Resp(_Msg(content="ok")), captured)
    asyncio.run(c.chat_with_tools(
        [{"role": "user", "content": "hi"}], temperature=0,
    ))
    assert captured["temperature"] == 0


def test_content_none_returns_empty(monkeypatch):
    c = _client_with_fake(monkeypatch, _Resp(_Msg(content=None)), {})
    out = asyncio.run(c.chat([{"role": "user", "content": "hi"}]))
    assert out == ""


def test_malformed_toolcall_skipped_not_fatal(monkeypatch):
    msg = _Msg(content=None, tool_calls=[
        _TC("a", "search_knowledge", '{"query": "高血压"}'),
        _TC("b", "assess_risk", '{"symptoms": "胸痛'),  # 截断的畸形 JSON
    ])
    c = _client_with_fake(
        monkeypatch, _Resp(msg, finish_reason="tool_calls"), {})
    resp = asyncio.run(c.chat_with_tools([{"role": "user", "content": "hi"}]))
    # 合法 call 保留，畸形 call 被跳过，整轮不抛
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "search_knowledge"
    assert resp.tool_calls[0].arguments == {"query": "高血压"}
