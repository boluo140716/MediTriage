"""回归：最终回答流式输出（answer_chunk SSE 事件，思考完成后打字机效果）。

守护：
① 分片按序拼接 == 完整回答，结尾有 done 标记；
② 空/过短回答不推流（直接 result，无感知差异）；
③ 端到端（TestClient + 假 swarm）：answer_chunk 事件先于 result，且顺序正确。
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def _import_server():
    if str(WEB_DIR) not in sys.path:
        sys.path.insert(0, str(WEB_DIR))
    import server
    return server


def test_stream_answer_emits_chunks_in_order():
    server = _import_server()
    events = []
    text = "您好" * 60  # 120 字符
    asyncio.run(server._stream_answer(
        lambda t, d: events.append((t, d)), text,
        chunk_size=20, interval=0.001,
    ))
    chunks = [d["chunk"] for t, d in events if t == "answer_chunk" and "chunk" in d]
    assert "".join(chunks) == text
    assert events[-1] == ("answer_chunk", {"done": True})


def test_stream_skips_empty_and_short():
    server = _import_server()
    events = []
    asyncio.run(server._stream_answer(
        lambda t, d: events.append((t, d)), "", chunk_size=20
    ))
    assert events == []
    asyncio.run(server._stream_answer(
        lambda t, d: events.append((t, d)), "短回答", chunk_size=20
    ))
    assert events == []


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # 纯规则 + 临时库：避免转诊评估依赖 LLM/真实 DB
    monkeypatch.setenv("MEDITRIAGE_ESCALATION_DB", str(tmp_path / "esc.db"))
    monkeypatch.setenv("MEDITRIAGE_ESCALATION_LLM", "0")
    server = _import_server()

    async def fake_swarm(question, session_id, event_emitter, user_id=None):
        event_emitter("session_started", {"session_id": session_id})
        return {
            "answer": "高血压患者日常饮食建议" * 30,
            "suggestions": [], "disclaimer": "",
            "swarm_enabled": True, "agents_involved": ["consultation_agent"],
            "session_id": session_id,
        }

    monkeypatch.setattr(server, "process_with_swarm", fake_swarm)
    from fastapi.testclient import TestClient
    with TestClient(server.app) as c:
        yield c


def test_ask_streams_answer_before_result(client):
    r = client.post("/api/ask", json={
        "question": "高血压饮食怎么安排？",
        "session_id": "stream-e2e",
    })
    assert r.status_code == 200
    order = []
    chunks = []
    for frame in r.text.split("\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        et, dl = "message", ""
        for line in frame.split("\n"):
            if line.startswith("event: "):
                et = line[7:]
            elif line.startswith("data: "):
                dl += line[6:]
        try:
            p = json.loads(dl)
        except Exception:
            continue
        order.append(p.get("type", et))
        if p.get("type") == "answer_chunk" and "chunk" in p.get("data", {}):
            chunks.append(p["data"]["chunk"])
    assert order.count("answer_chunk") >= 1
    assert order.index("answer_chunk") < order.index("result")
    assert "".join(chunks) == "高血压患者日常饮食建议" * 30
