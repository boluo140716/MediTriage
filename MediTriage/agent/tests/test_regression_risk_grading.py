"""回归：assess_risk 临床红旗分级（确定性纯单元，不依赖 vLLM/Milvus）。

覆盖 emergency 档可达、口语归一、症状组合，以及紧急建议不被误兜底为
'观察'。"""
import sys
from pathlib import Path

import pytest

# assess-risk skill 脚本目录（不在 swarm 根下，单独入 path）
_SKILL = (
    Path(__file__).resolve().parent.parent
    / "meditriage/skills/assess-risk/script"
)
sys.path.insert(0, str(_SKILL))

from risk import _grade_risk, _recommendation  # noqa: E402


@pytest.mark.parametrize("text,expected", [
    # 胸痛+呼吸困难组合（口语'胸口剧痛'/'喘不上气'归一）
    ("胸口剧痛还喘不上气", "emergency"),
    ("突然叫不醒了", "emergency"),  # 昏迷（口语'叫不醒'）
    ("一直抽搐", "emergency"),
    ("咯血不止", "emergency"),
    ("嘴歪了半边身子没力气", "emergency"),  # 卒中体征
    ("胸痛", "high"),  # 高危单症状
    ("持续呕吐好几天", "medium"),  # 收紧高危表：持续呕吐降为 medium（普通肠胃炎常见）
    ("蜱虫叮咬后发烧和皮疹", "medium"),  # 发烧→发热+皮疹中危组合
    ("拉肚子好几天", "medium"),  # 病程修饰词
    ("轻微咳嗽流鼻涕", "low"),
])
def test_grade_risk_tiers(text, expected):
    level, _reasons = _grade_risk(text)
    assert level == expected, f"{text!r} → {level}（期望 {expected}）"


def test_emergency_tier_reachable():
    """emergency 档曾是死代码（无任何路径可达），此处守护其可达性。"""
    assert _grade_risk("昏迷")[0] == "emergency"


def test_emergency_recommendation_not_downgraded():
    """emergency 建议绝不能被兜底成'密切观察'的低危建议。"""
    rec = _recommendation("emergency")
    assert ("120" in rec) or ("急诊" in rec)
    assert "密切观察" not in rec
