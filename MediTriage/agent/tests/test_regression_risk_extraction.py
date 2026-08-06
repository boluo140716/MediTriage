"""回归：风险等级标签提取（供转诊层结构化判定）。

守护：
① "风险等级：X" 与 "被判定为/评估为/风险等级为 X" 均能提取；
② 升级警示语（"可能升级为高危"）不误判；
③ emergency > high > medium > low 判定正确。
"""
from meditriage.agents.diagnostic_agent import extract_risk_level


def test_extract_risk_level_various_formats():
    assert extract_risk_level("风险等级：低危") == "low"
    assert extract_risk_level("风险等级为中危") == "medium"
    assert extract_risk_level("被判定为**低危**") == "low"
    assert extract_risk_level("评估为**高危**") == "high"
    assert extract_risk_level("风险等级：紧急") == "emergency"


def test_extract_risk_level_ignores_escalation_wording():
    # 升级警示语不是风险等级标签，不误判
    assert extract_risk_level("若不处理，病情可能升级为高危情况") == "unknown"
    assert extract_risk_level("没有风险等级标签") == "unknown"


def test_extract_risk_level_priority():
    assert extract_risk_level("风险等级：中高") == "high"  # 中高保守取高
