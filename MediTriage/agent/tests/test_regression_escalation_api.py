"""回归：转诊 Web API（FastAPI TestClient，tmp SQLite，不起真实服务）。

覆盖：
① GET /doctor 返回医生端页面；
② 危机问诊建单后：列表可见、详情可取、医生回复后状态流转；
③ 重复回复 409、不存在 404、空回复 422；
④ status 过滤参数。
"""
import asyncio
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # 先注入环境再首次调用单例，保证服务用 tmp DB + 纯规则模式
    monkeypatch.setenv("MEDITRIAGE_ESCALATION_DB", str(tmp_path / "esc.db"))
    monkeypatch.setenv("MEDITRIAGE_ESCALATION_LLM", "0")
    # 重置转诊服务进程单例：避免前面测试污染（API 端点用单例，需与本 fixture 同 DB）
    import meditriage.workflows.escalation as _esc_mod
    monkeypatch.setattr(_esc_mod, "_service", None)
    web_dir = Path(__file__).resolve().parents[1] / "web"
    if str(web_dir) not in sys.path:
        sys.path.insert(0, str(web_dir))
    from meditriage.workflows import get_escalation_service
    from fastapi.testclient import TestClient
    from server import app
    with TestClient(app) as c:
        yield c, get_escalation_service()


def _seed_crisis(svc, session_id="api-flow"):
    """用危机案例建一张转诊单（纯规则模式，危机关键词强制转诊）。"""
    return asyncio.run(svc.evaluate(
        question="我胸口疼得厉害，还冒冷汗，怎么办？",
        answer="建议尽快就医，避免剧烈活动。",
        session_id=session_id,
    ))


def test_doctor_page(client):
    c, _ = client
    r = c.get("/doctor")
    assert r.status_code == 200
    assert "escalation" in r.text.lower() or "转诊" in r.text


def test_full_escalation_flow(client):
    c, svc = client
    esc = _seed_crisis(svc)
    assert esc is not None

    # 列表
    r = c.get("/api/escalations")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] >= 1
    item = next(x for x in data["items"] if x["escalation_id"] == esc["escalation_id"])
    assert item["status"] == "escalated"
    assert item["risk_level"] == "high"

    # 详情（含结构化交接摘要）
    d = c.get(f"/api/escalations/{esc['escalation_id']}")
    assert d.status_code == 200
    detail = d.json()
    assert detail["summary"]["症状"]
    assert detail["question"] == "我胸口疼得厉害，还冒冷汗，怎么办？"

    # 医生回复 -> 状态流转
    rp = c.post(
        f"/api/escalations/{esc['escalation_id']}/reply",
        json={"reply": "建议尽快到心内科就诊，完善心电图检查。"},
    )
    assert rp.status_code == 200
    assert rp.json()["status"] == "doctor_replied"
    assert rp.json()["doctor_reply"].startswith("建议尽快到心内科")

    # 重复回复 -> 409
    rp2 = c.post(
        f"/api/escalations/{esc['escalation_id']}/reply",
        json={"reply": "再次回复"},
    )
    assert rp2.status_code == 409

    # 不存在 -> 404
    assert c.get("/api/escalations/ESC-NOTEXIST").status_code == 404
    assert c.post(
        "/api/escalations/ESC-NOTEXIST/reply", json={"reply": "x"}
    ).status_code == 404

    # 空回复 -> 422（pydantic min_length）
    assert c.post(
        f"/api/escalations/{esc['escalation_id']}/reply", json={"reply": ""}
    ).status_code == 422


def test_escalations_status_filter(client):
    c, svc = client
    esc = _seed_crisis(svc, session_id="api-filter")
    # 默认只返回待处理（escalated）
    r = c.get("/api/escalations")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "escalated"
    assert any(x["escalation_id"] == esc["escalation_id"] for x in data["items"])
    # 显式过滤
    r1 = c.get("/api/escalations", params={"status": "escalated"})
    assert r1.status_code == 200 and r1.json()["count"] >= 1
    r2 = c.get("/api/escalations", params={"status": "doctor_replied"})
    assert r2.status_code == 200
    # 回复后：默认队列不再含该单，status=all 仍可见
    c.post(
        f"/api/escalations/{esc['escalation_id']}/reply",
        json={"reply": "建议心内科门诊就诊"},
    )
    r3 = c.get("/api/escalations")
    assert not any(
        x["escalation_id"] == esc["escalation_id"] for x in r3.json()["items"]
    )
    r4 = c.get("/api/escalations", params={"status": "all"})
    assert any(
        x["escalation_id"] == esc["escalation_id"] for x in r4.json()["items"]
    )
    # 非法 status -> 422
    assert c.get("/api/escalations", params={"status": "bogus"}).status_code == 422


def test_ask_result_carries_escalation(client):
    """/api/ask 的 result 事件在危机命中时带 escalation 字段（黑盒验证）。

    不真跑 swarm：直接调用 server 的 _attach_escalation helper，
    确认 payload 会被附上转诊单信息。
    """
    c, svc = client
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web"))
    from server import _attach_escalation

    async def run():
        created = []
        payload = await _attach_escalation(
            payload={"answer": "建议尽快就医", "mode": "swarm"},
            question="家人突然说话不清楚、一侧手脚没力气，怎么办？",
            session_id="api-ask",
            user_id=None,
            on_created=created.append,
        )
        return payload, created

    payload, created = asyncio.run(run())
    assert payload["escalation"]["escalation_id"].startswith("ESC-")
    assert payload["escalation"]["status"] == "escalated"
    # on_created 回调先于 result 触发（患者端 SSE 通知）
    assert len(created) == 1
    assert created[0]["escalation_id"] == payload["escalation"]["escalation_id"]


def test_delete_single_and_batch(client):
    """医生端删除转诊单：单删 + 批量删除 + 幂等。"""
    c, svc = client
    # 建两张单
    e1 = svc.store.create(
        session_id="del-1", question="测试删除1",
        summary={"风险等级": "low"}, risk_level="low",
        confidence=1.0, reasons=[],
    )
    e2 = svc.store.create(
        session_id="del-2", question="测试删除2",
        summary={"风险等级": "high"}, risk_level="high",
        confidence=0.3, reasons=["危机"],
    )
    # 单删
    r1 = c.delete("/api/escalations/" + e1["escalation_id"])
    assert r1.status_code == 200 and r1.json()["ok"] is True
    # 幂等：再删不存在 -> 404
    assert c.delete("/api/escalations/" + e1["escalation_id"]).status_code == 404
    # 批量删（含一个不存在 id）
    r2 = c.post("/api/escalations/batch-delete", json={
        "ids": [e2["escalation_id"], "ESC-NOTEXIST"]})
    assert r2.status_code == 200
    assert r2.json()["deleted"] == 1
    # 列表已清空
    lst = svc.store.list(status=None)
    assert len(lst) == 0
