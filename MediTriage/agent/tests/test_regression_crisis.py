"""回归：自伤/轻生危机支持（确定性，离线，不依赖 vLLM/Milvus）。

守护：
1. 明确自伤词被 assess_risk 判为 emergency（确定性分诊）。
2. 自伤表达的最终答复最前置入具体危机热线 12356，且不重复、不误伤正常问题
   （旧版只说“拨打热线”却无具体号码，此处防其回退）。
"""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent  # agent/
_SKILL = _ROOT / "meditriage/skills/assess-risk/script"
sys.path.insert(0, str(_SKILL))
from risk import _grade_risk  # noqa: E402

from meditriage.swarm.swarm_coordinator import (  # noqa: E402
    _ensure_crisis_support,
    _SELF_HARM_PAT,
)


@pytest.mark.parametrize("text", [
    "我想自杀", "不想活了", "想轻生", "想自残",
    "活着没意义，想结束这一切", "生无可恋", "活不下去了",
])
def test_self_harm_detected(text):
    assert _SELF_HARM_PAT.search(text), f"{text!r} 应被识别为自伤信号"


@pytest.mark.parametrize("text", ["我想自杀", "不想活了", "想轻生", "想自残"])
def test_self_harm_graded_emergency(text):
    assert _grade_risk(text)[0] == "emergency"


def test_crisis_block_prepended():
    out = _ensure_crisis_support(
        "我觉得活着没意义，想结束这一切", "【回答】请多休息。"
    )
    assert "12356" in out
    assert out.startswith("【心理危机支持】")
    assert "【回答】请多休息。" in out  # 原答复保留在后


def test_crisis_no_dup_if_already_present():
    ans = "若有需要请拨打 12356 心理援助热线。"
    assert _ensure_crisis_support("我想自杀", ans) == ans  # 已含则不重复注入


def test_non_crisis_unchanged():
    ans = "高血压要低盐饮食。"
    assert _ensure_crisis_support("高血压怎么办", ans) == ans


# ---- 口语化/委婉表达（压测实测漏检：模型转而自己编过时热线号码）----

@pytest.mark.parametrize("text", [
    "最近真的撑不下去了", "感觉自己快要消失了", "活着好累",
    "想解脱了", "不想醒来", "生不如死",
])
def test_colloquial_self_harm_detected(text):
    assert _SELF_HARM_PAT.search(text), f"{text!r} 应被识别为自伤信号"


def test_answer_side_crisis_triggers_hotline():
    """模型答复已在谈自杀风险（识别到了危机）→ 同样必须带官方热线。"""
    out = _ensure_crisis_support(
        "最近压力好大",
        "您的描述提示可能存在自杀风险，建议拨打心理热线 400-161-9995。",
    )
    assert "12356" in out
    assert out.startswith("【心理危机支持】")


def test_benign_answer_not_injected():
    ans = "规律作息、适度运动有助于缓解压力。"
    assert _ensure_crisis_support("最近压力好大", ans) == ans
