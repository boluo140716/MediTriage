"""回归：guardrails 硬安全规则（纯单元，不依赖 vLLM/Milvus）。

守护三条医疗安全边界（保证治愈 / 自行用药 / 定夺侵入性治疗）确实被
validator 检出，且不误伤正常的低危观察建议。
"""
import pytest

from meditriage.guardrails import ConstraintValidator

V = ConstraintValidator()


def _violations(agent_id: str, text: str):
    return V.validate_output(agent_id, text)["violations"]


def test_guarantee_cure_detected():
    v = _violations("consultation_agent", "放心，这个疗法保证治愈你的病。")
    assert any("保证治愈" in x for x in v), v


def test_self_treatment_detected():
    v = _violations("diagnostic_agent", "你可以自行购买抗生素服用。")
    assert any("自行用药" in x for x in v), v


def test_recommend_treatment_detected():
    v = _violations("research_agent", "建议立即接受手术。")
    assert any("侵入性治疗" in x for x in v), v


def test_benign_low_risk_no_false_positive():
    """正常低危建议（注意休息、必要时就医）不应触发硬安全违规。"""
    v = _violations("diagnostic_agent", "症状较轻，注意休息、多喝水，必要时再就医。")
    assert not any(
        ("自行用药" in x) or ("保证治愈" in x) or ("侵入性治疗" in x)
        for x in v
    ), v


# ---- 越界违规必须可修复（检测到 → auto_fixable → AutoFixer 追加纠正提示）----

def test_overreach_marked_auto_fixable():
    for text in [
        "你可以自行购买抗生素服用。",        # 自行用药
        "放心，这个疗法保证治愈你的病。",      # 保证治愈
        "建议立即接受手术。",                # 侵入性治疗
    ]:
        r = V.validate_output("diagnostic_agent", text)
        assert "add_overreach_caveat" in r["auto_fixable"], (text, r)


def test_benign_not_marked_overreach():
    r = V.validate_output("diagnostic_agent", "症状较轻，注意休息、多喝水，必要时再就医。")
    assert "add_overreach_caveat" not in r["auto_fixable"], r


def test_autofixer_appends_caveat_idempotent():
    from meditriage.guardrails import AutoFixer
    f = AutoFixer()
    out = f.fix_output("你可以自行购买抗生素服用。", ["add_overreach_caveat"])
    assert "不构成诊断结论" in out
    assert out.startswith("你可以自行购买抗生素服用。")
    # 幂等：再修一次不重复追加
    assert f.fix_output(out, ["add_overreach_caveat"]) == out


# ---- 误报回归：解释性"就是"、否定句、安全劝阻不得标违规 ----

def test_no_fp_explanatory_jiushi():
    v = _violations("diagnostic_agent", "高血压就是指血压持续高于140/90mmHg的状态。")
    assert not any("明确诊断" in x for x in v), v


def test_no_fp_negated_diagnosis():
    v = _violations("diagnostic_agent", "目前不能确诊为肺炎，需要进一步拍胸片检查。")
    assert not any("明确诊断" in x for x in v), v


def test_no_fp_safety_advice_against_self_medication():
    for text in ["请勿自行服用抗生素。", "不要自己用药，应先就医评估。"]:
        v = _violations("diagnostic_agent", text)
        assert not any("自行用药" in x for x in v), (text, v)


def test_no_fp_negated_cure_and_surgery():
    v1 = _violations("diagnostic_agent", "医学上不能保证治愈，但规范治疗可以良好控制。")
    assert not any("保证治愈" in x for x in v1), v1
    v2 = _violations("diagnostic_agent", "目前不建议手术，建议先保守观察。")
    assert not any("侵入性治疗" in x for x in v2), v2


# ---- 漏报回归：真实越界表述必须检出 ----

def test_diagnosis_variants_detected():
    for text in ["您患有高血压。", "您得的是2型糖尿病。", "可以确定是心绞痛。"]:
        v = _violations("diagnostic_agent", text)
        assert any("明确诊断" in x for x in v), (text, v)


def test_cure_promise_with_hui_detected():
    v = _violations("diagnostic_agent", "放心，一定会痊愈的。")
    assert any("保证治愈" in x for x in v), v


def test_surgery_without_connector_detected():
    v = _violations("diagnostic_agent", "建议尽快手术治疗。")
    assert any("侵入性治疗" in x for x in v), v


def test_prescription_drugname_with_dose_detected():
    for text in ["可以吃对乙酰氨基酚500mg缓解。", "布洛芬 400mg，每6小时一次。"]:
        v = _violations("diagnostic_agent", text)
        assert any("药物处方" in x for x in v), (text, v)


def test_dietary_doses_not_prescription():
    v = _violations(
        "diagnostic_agent",
        "高血压患者每日钾摄入3500-5000mg，蔬菜500克，饮水2000毫升。",
    )
    assert not any("药物处方" in x for x in v), v


# ---- swarm 汇总出口护栏：越界汇总文本被补救 ----

def test_swarm_exit_guard_fixes_overreach():
    from meditriage.swarm.swarm_coordinator import _guard_final_answer
    out = _guard_final_answer("放心，这个疗法保证治愈你的病。")
    assert "不构成诊断结论" in out, out


def test_swarm_exit_guard_passes_clean_text():
    from meditriage.swarm.swarm_coordinator import _guard_final_answer
    clean = "症状较轻，注意休息，必要时就医。"
    assert _guard_final_answer(clean) == clean
