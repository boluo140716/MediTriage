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


def test_llm_exception_no_rule_signal_not_escalate(make_service):
    """评分失败 + 无规则高危信号 -> 不转（不再'宁过不欠'全转人工）。"""
    svc, store = make_service(llm_exc=RuntimeError("api down"))
    esc = _run(svc, question="我最近经常头晕", answer="建议规律作息",
               session_id="llm-exc")
    assert esc is None


def test_llm_bad_json_no_rule_signal_not_escalate(make_service):
    svc, store = make_service(llm_text="抱歉，我无法评估。")
    esc = _run(svc, question="我最近经常头晕", answer="建议规律作息",
               session_id="llm-bad")
    assert esc is None


def test_llm_confidence_out_of_range_not_escalate(make_service):
    """非法分（越界）解析失败 -> 无规则高危信号不转。"""
    svc, store = make_service(llm_text='{"confidence": 3.7, "risk_level": "low"}')
    esc = _run(svc, question="我最近经常头晕", answer="建议规律作息",
               session_id="llm-range")
    assert esc is None


def test_llm_failure_with_seek_care_not_escalate(make_service):
    """评分失败 + 回答含"立即就医"话术 -> 不转（回答侧就医提示不再触发转诊）。"""
    svc, store = make_service(llm_exc=RuntimeError("api down"))
    esc = _run(svc, question="我最近经常头晕",
               answer="情况比较危险，建议立即就医。",
               session_id="llm-seekcare")
    assert esc is None


def test_llm_medium_not_escalate(make_service):
    """评分 medium -> 不转（中危由 AI 答复+就医建议，不强制转人工）。"""
    svc, store = make_service(llm_text='{"confidence": 0.3, "risk_level": "medium"}')
    esc = _run(svc, question="我最近经常头晕", answer="建议规律作息",
               session_id="llm-med")
    assert esc is None


def test_agent_high_escalates(make_service):
    """Agent 结构化评估 high -> 直接转（不等 LLM 评分）。"""
    svc, store = make_service(llm_text='{"confidence": 0.9, "risk_level": "low"}')
    esc = _run(svc, question="我最近经常头晕", answer="风险等级：高危，建议尽快就医",
               session_id="llm-agh", result={"risk_level": "high"})
    assert esc is not None
    assert esc["risk_level"] == "high"
