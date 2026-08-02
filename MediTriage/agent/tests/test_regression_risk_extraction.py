"""回归：诊断 Agent 风险等级提取（纯单元，不依赖服务）。

只解析"风险等级:"紧跟的标签词，不被升级警示语"高危"全文误判。
直接覆盖 agents.diagnostic_agent.extract_risk_level（模块级纯函数）。
"""
import pytest

from meditriage.agents.diagnostic_agent import extract_risk_level


@pytest.mark.parametrize("text,expected", [
    # 中危答案后文含"升级为高危"，全文子串逻辑会误判为 high
    ("风险等级：中危（需重视但非紧急）\n理由：…可能升级为高危情况，需立即处理。", "medium"),
    ("【风险评估】\n风险等级：中高（需警惕但无需过度恐慌）\n紧急程度：…", "high"),  # 中高保守取高
    ("风险等级：高危 🔴\n建议立即就医", "high"),
    ("风险等级：低危 🟢，非紧急但需观察", "low"),  # 后文含"紧急"不应误升级
    ("风险等级：紧急 🚨 立即拨打急救", "emergency"),
    ("没有风险等级字样的普通回答", "unknown"),
])
def test_extract_risk_level(text, expected):
    assert extract_risk_level(text) == expected, (
        f"{text[:20]!r}… → 期望 {expected}")
