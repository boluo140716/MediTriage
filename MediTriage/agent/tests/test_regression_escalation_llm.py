"""回归：LLM 置信度打分分支（纯单元，注入 fake LLM，不起服务）。

守护：
① 模糊场景 LLM 低置信 -> 转诊；
② LLM 高置信 -> 不转诊；
③ LLM 抛异常 -> 安全兜底转人工（宁过不欠）；
④ LLM 输出非法 JSON -> 安全兜底转人工。
"""
import asyncio

import pytest

from meditriage.workflows.escalation import EscalationService
from meditriage.workflows.escalation_store import EscalationStore


class FakeLLM:
    def __init__(self, text=None, exc=None):
        self.text = text
        self.exc = exc
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        if self.exc:
            raise self.exc
        return self.text


@pytest.fixture()
def make_service(tmp_path):
    def _make(llm_text=None, llm_exc=None):
        store = EscalationStore(db_path=str(tmp_path / "esc.db"))
        svc = EscalationService(
            store=store, llm_client=FakeLLM(text=llm_text, exc=llm_exc),
            use_llm=True, threshold=0.4,
        )
        return svc, store
    return _make


def _run(svc, **kw):
    return asyncio.run(svc.evaluate(**kw))


def test_low_confidence_escalates(make_service):
    svc, store = make_service(llm_text='{"confidence": 0.2, "risk_level": "high", '
                                       '"summary": {"症状": "头晕", "已给建议": "休息", '
                                       '"风险等级": "high", "待确认点": "是否需检查"}}')
    esc = _run(svc, question="我最近经常头晕，休息也不好", answer="建议规律作息，观察症状",
               session_id="llm-low")
    assert esc is not None
    assert esc["confidence"] < 0.4
    assert any("置信度" in r for r in esc["reasons"])
    assert esc["summary"]["症状"] == "头晕"
    # 交接摘要固定四键（LLM 缺键时规则兜底补齐）
    for k in ("症状", "已给建议", "风险等级", "待确认点"):
        assert esc["summary"].get(k), f"summary 缺键: {k}"


def test_high_confidence_not_escalate(make_service):
    svc, store = make_service(llm_text='{"confidence": 0.85, "risk_level": "low"}')
    esc = _run(svc, question="我最近经常头晕", answer="建议规律作息，多喝水",
               session_id="llm-high")
    assert esc is None


def test_llm_exception_fallback_escalate(make_service):
    svc, store = make_service(llm_exc=RuntimeError("api down"))
    esc = _run(svc, question="我最近经常头晕", answer="建议规律作息",
               session_id="llm-exc")
    assert esc is not None
    assert any("兜底转人工" in r for r in esc["reasons"])


def test_llm_bad_json_fallback_escalate(make_service):
    svc, store = make_service(llm_text="抱歉，我无法评估。")
    esc = _run(svc, question="我最近经常头晕", answer="建议规律作息",
               session_id="llm-bad")
    assert esc is not None
    assert any("兜底转人工" in r for r in esc["reasons"])


def test_llm_confidence_out_of_range_treated_bad(make_service):
    svc, store = make_service(llm_text='{"confidence": 3.7, "risk_level": "low"}')
    esc = _run(svc, question="我最近经常头晕", answer="建议规律作息",
               session_id="llm-range")
    assert esc is not None  # 非法分 -> 兜底转人工


def test_fallback_rate_limit_per_session(make_service):
    """LLM 故障兜底限频：同一会话最多建 1 张兜底单。"""
    svc, store = make_service(llm_exc=RuntimeError("api down"))
    # 同一会话两条不同提问：第一条兜底建单，第二条被限频跳过
    esc1 = _run(svc, question="我最近经常头晕", answer="建议规律作息",
                session_id="llm-rate")
    esc2 = _run(svc, question="我最近经常背痛", answer="建议热敷观察",
                session_id="llm-rate")
    assert esc1 is not None
    assert esc2 is None
    assert len(store.list(status=None)) == 1


def test_fallback_limit_is_per_session(make_service):
    """不同会话互不影响限频计数。"""
    svc, store = make_service(llm_exc=RuntimeError("api down"))
    e1 = _run(svc, question="我最近经常头晕", answer="建议规律作息", session_id="sA")
    e2 = _run(svc, question="我最近经常背痛", answer="建议热敷观察", session_id="sB")
    assert e1 is not None and e2 is not None
