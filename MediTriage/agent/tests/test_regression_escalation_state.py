"""回归：转诊状态机与幂等建单（纯单元，tmp SQLite，不起服务）。

守护：
① 建单即流转 ai_processing -> escalated；
② 同会话未回复不重复建单（幂等），已回复后可再建；
③ 非法流转（ai_processing -> doctor_replied）拒绝；
④ 医生回复：escalated -> doctor_replied，重复回复/空回复拒绝。
"""
import pytest

from meditriage.workflows.escalation_store import (
    EscalationStore,
    STATUS_AI_PROCESSING,
    STATUS_ESCALATED,
    STATUS_DOCTOR_REPLIED,
)


@pytest.fixture()
def store(tmp_path):
    s = EscalationStore(db_path=str(tmp_path / "esc.db"))
    yield s
    s.close()


def _summary(**kw):
    d = {"症状": "胸口疼", "已给建议": "建议尽快就医", "风险等级": "high",
         "待确认点": "需医生复核"}
    d.update(kw)
    return d


def _create(store, session_id="s1", **kw):
    return store.create(
        session_id=session_id,
        question=kw.get("question", "胸口疼得厉害，怎么办？"),
        summary=_summary(),
        risk_level=kw.get("risk_level", "high"),
        confidence=kw.get("confidence", 0.0),
        reasons=kw.get("reasons", ["危机关键词命中：胸口疼"]),
        user_id="u1",
    )


def test_create_transitions_to_escalated(store):
    esc = _create(store)
    assert esc["status"] == STATUS_ESCALATED
    assert esc["escalation_id"].startswith("ESC-")
    assert esc["summary"]["症状"] == "胸口疼"
    assert esc["reasons"] == ["危机关键词命中：胸口疼"]
    assert esc["confidence"] == 0.0
    assert esc["risk_level"] == "high"


def test_duplicate_create_is_idempotent(store):
    """同一提问（忽略空白差异）重复触发 -> 复用已有单。"""
    a = _create(store)
    b = _create(store)  # 完全相同的提问（_create 默认问句）
    assert a["id"] == b["id"]
    assert a["escalation_id"] == b["escalation_id"]
    assert len(store.list()) == 1


def test_same_session_new_crisis_creates_new_ticket(store):
    """同会话内换一个危机（不同提问）-> 开新单，避免返回过期信息。"""
    a = _create(store)  # 胸口疼
    b = _create(store, question="家人突然说话不清楚、一侧手脚没力气，怎么办？")
    assert b["id"] != a["id"]
    assert b["question"] != a["question"]
    assert len(store.list()) == 2


def test_new_ticket_after_replied(store):
    a = _create(store)
    store.reply(a["id"], "已联系患者，建议门诊就诊")
    b = _create(store)
    assert b["id"] != a["id"]
    assert len(store.list()) == 2


def test_reply_transition(store):
    esc = _create(store)
    row = store.reply(esc["escalation_id"], "建议门诊随访，避免剧烈运动")
    assert row["status"] == STATUS_DOCTOR_REPLIED
    assert row["doctor_reply"] == "建议门诊随访，避免剧烈运动"
    assert row["replied_at"]
    # 再次回复：状态已终态，拒绝
    with pytest.raises(ValueError):
        store.reply(esc["escalation_id"], "再回一次")


def test_reply_empty_rejected(store):
    esc = _create(store)
    with pytest.raises(ValueError):
        store.reply(esc["id"], "   ")


def test_illegal_transition_rejected(store):
    """ai_processing -> doctor_replied 跨级流转必须拒绝。"""
    # 直接插入一条 ai_processing 状态的行（create 会自动流转到 escalated）
    store._execute(
        "INSERT INTO escalations (escalation_id, session_id, user_id, question, "
        "answer_preview, summary_json, risk_level, confidence, reasons_json, "
        "status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("ESC-TEST1", "s-illegal", "u1", "q", "", "{}", "medium", 0.5, "[]",
         STATUS_AI_PROCESSING, "2026-08-03T00:00:00", "2026-08-03T00:00:00"),
    )
    with pytest.raises(ValueError):
        store._transition("ESC-TEST1", STATUS_DOCTOR_REPLIED)
    # 合法路径：ai_processing -> escalated 可流转
    row = store._transition("ESC-TEST1", STATUS_ESCALATED)
    assert row["status"] == STATUS_ESCALATED


def test_get_by_id_and_esc_id(store):
    esc = _create(store)
    assert store.get(esc["id"])["escalation_id"] == esc["escalation_id"]
    assert store.get(esc["escalation_id"])["id"] == esc["id"]
    assert store.get("ESC-NOTEXIST") is None
    assert store.get(99999) is None


def test_reply_nonexistent_raises(store):
    with pytest.raises(KeyError):
        store.reply("ESC-NOTEXIST", "回复")
