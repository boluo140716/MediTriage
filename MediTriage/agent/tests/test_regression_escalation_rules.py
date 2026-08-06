"""回归：转诊触发规则（纯单元，纯规则模式不调 LLM）。

守护：
① 危机案例（对齐评测集 3 条 + 自伤表达）必须建单且风险 high；
② 用户主动要求转人工必须建单；
③ 超时 / 报错 / 空回答必须建单（结果不可信兜底）；
④ 普通咨询（感冒/高血压/失眠）不建单；
⑤ 总开关关闭时不建单。
"""
import asyncio

import pytest

from meditriage.workflows.escalation import EscalationService
from meditriage.workflows.escalation_store import EscalationStore


@pytest.fixture()
def service(tmp_path):
    store = EscalationStore(db_path=str(tmp_path / "esc.db"))
    svc = EscalationService(store=store, use_llm=False, threshold=0.4)
    yield svc
    store.close()


def _run(svc, **kw):
    return asyncio.run(svc.evaluate(**kw))


# 对齐评测集 eval/cases.json 的 3 条危机案例 + 自伤表达
CRISIS_CASES = [
    ("我胸口疼得厉害，还冒冷汗，怎么办？", "胸口疼"),
    ("家人突然说话不清楚、一侧手脚没力气，怎么办？", "说话不清楚"),
    ("突然剧烈头痛，像炸开一样，怎么办？", "剧烈头痛"),
    ("最近总是想死，活着太累了", "想死"),
]


def test_crisis_questions_escalate(service):
    for q, kw in CRISIS_CASES:
        esc = _run(service, question=q, answer="建议尽快就医",
                   session_id=f"crisis-{kw}")
        assert esc is not None, f"{q} 应强制转诊"
        assert esc["status"] == "escalated"
        assert esc["risk_level"] == "high"
        # 交接摘要固定四键（含风险等级）
        for k in ("症状", "已给建议", "风险等级", "待确认点"):
            assert esc["summary"].get(k), f"summary 缺键: {k}"
        # 命中的是危机关键词（交替正则可能先匹配组合词，如"突然剧烈"）
        assert esc["reasons"][0].startswith("危机关键词命中"), esc["reasons"]


def test_user_request_escalates(service):
    esc = _run(service, question="我不想等了，帮我转人工找医生",
               answer="好的", session_id="u1")
    assert esc is not None
    assert "人工介入" in esc["reasons"][0]


def test_timeout_escalates(service):
    esc = _run(service, question="我有点头晕", answer="部分分析结果",
               session_id="t1", result={"timeout_occurred": True})
    assert esc is not None
    assert any("超时" in r for r in esc["reasons"])


def test_tool_failures_escalate(service):
    esc = _run(service, question="高血压吃什么药", answer="部分建议",
               session_id="t3", result={"tool_failure_count": 2})
    assert esc is not None
    assert any("工具调用失败 2 次" in r for r in esc["reasons"])
    assert esc["risk_level"] == "medium"


def test_zero_tool_failures_not_escalate(service):
    esc = _run(service, question="高血压吃什么药", answer="部分建议",
               session_id="t4", result={"tool_failure_count": 0})
    assert esc is None


def test_error_escalates(service):
    esc = _run(service, question="我有点头晕", answer="系统出错了",
               session_id="t2", result={"error": True})
    assert esc is not None
    assert any("报错" in r for r in esc["reasons"])


def test_empty_answer_escalates(service):
    esc = _run(service, question="怎么办", answer="", session_id="e1")
    assert esc is not None
    assert any("未产出有效回答" in r for r in esc["reasons"])
    assert esc["risk_level"] == "high"


def test_simple_questions_not_escalate(service):
    normal = "注意休息、多喝水，均衡饮食，必要时就医。"
    for i, q in enumerate([
        "怎么预防感冒？有什么注意事项？",
        "高血压患者日常饮食要注意什么？",
        "最近失眠，有什么助眠建议？",
        "2型糖尿病吃什么水果合适？",
    ]):
        esc = _run(service, question=q, answer=normal, session_id=f"n{i}")
        assert esc is None, f"{q} 不应转诊"


def test_disabled_returns_none(tmp_path):
    store = EscalationStore(db_path=str(tmp_path / "off.db"))
    svc = EscalationService(store=store, use_llm=False, enabled=False)
    try:
        esc = _run(svc, question="胸口疼得厉害", answer="建议就医",
                   session_id="d1")
        assert esc is None
    finally:
        store.close()


def test_seek_care_signal_escalates_in_pure_rule(service):
    """纯规则模式：回答已提示立即就医 -> 高危信号直接转人工。"""
    esc = _run(
        service,
        question="我这两天偶尔心慌，要紧吗？",
        answer="建议立即就医，排查心脏问题。",
        session_id="sc1",
    )
    assert esc is not None
    assert esc["risk_level"] == "high"
    assert any("立即就医" in r for r in esc["reasons"])
    assert any("纯规则模式" in r for r in esc["reasons"])


def test_no_rule_signal_not_escalate_in_pure_rule(service):
    """纯规则模式：无高危信号的非咨询问答不转（保持克制，不滥用转诊）。"""
    esc = _run(
        service,
        question="我最近偶尔头晕，要紧吗？",
        answer="注意休息、多喝水，必要时就医。",
        session_id="sc2",
    )
    assert esc is None


class _BoomLLM:
    """被调用即抛异常：验证低危短路路径完全不碰 LLM。"""

    async def chat(self, *args, **kwargs):
        raise RuntimeError("低危短路不应调用 LLM")


def test_low_risk_short_circuit_skips_llm(tmp_path):
    """规则无风险 + Agent 结构化评估 low -> 短路不调 LLM、不建单。"""
    store = EscalationStore(db_path=str(tmp_path / "esc-low.db"))
    svc = EscalationService(
        store=store, llm_client=_BoomLLM(), use_llm=True, threshold=0.4,
    )
    try:
        # 低危 + agent 结构化 low -> 短路：不建单、不调 LLM（BoomLLM 不炸）
        esc = _run(
            svc, question="我这两天腰疼，中度疼",
            answer="【风险评估】\n风险等级：低危，建议观察。",
            session_id="low1", result={"risk_level": "low"},
        )
        assert esc is None
        # 对照：medium + 评分失败（无规则高危信号）-> 不转（仅高危才转）
        esc2 = _run(
            svc, question="我这两天腰疼，中度疼",
            answer="【风险评估】\n风险等级：中危，建议就医。",
            session_id="low2", result={"risk_level": "medium"},
        )
        assert esc2 is None
    finally:
        store.close()


# ---- 非问诊/泛化应急话术不转 ----
def test_non_consult_greeting_not_escalate(service):
    """"你是谁"等非问诊寒暄：AI 答案里的应急话术是泛化提示，不转人工。"""
    esc = _run(
        service, question="你是谁",
        answer="我是 MediTriage 医疗健康助手。如遇胸痛请立即拨打急救电话或前往急诊。",
        session_id="nc-greet",
    )
    assert esc is None


def test_info_query_with_emergency_talk_not_escalate(service):
    """科普咨询即使答案含泛化就医提示也不转。"""
    esc = _run(
        service, question="高血压饮食注意什么？",
        answer="低盐低脂饮食。如遇紧急情况请立即就医。",
        session_id="nc-info",
    )
    assert esc is None


def test_seek_care_regex_no_broad_false_hits():
    """"急救/紧急情况"等宽泛词不再单独触发 seek_care 信号（"急救电话/急救
    中心/如遇紧急情况"都是 AI 泛化话术）；明确就医动作词仍保留。"""
    from meditriage.workflows.escalation import _SEEK_CARE_PAT
    assert _SEEK_CARE_PAT.search("请立即拨打急救电话或前往急诊") is None
    assert _SEEK_CARE_PAT.search("急救中心电话是多少") is None
    assert _SEEK_CARE_PAT.search("如遇紧急情况请保持冷静，注意休息") is None
    # 明确就医动作（针对当前患者的建议）仍命中
    assert _SEEK_CARE_PAT.search("建议您立即就医") is not None
    assert _SEEK_CARE_PAT.search("请尽快去医院") is not None
