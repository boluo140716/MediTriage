"""回归：用药剂量安全兜底（确定性，离线）。

守护：答复含具体药物剂量时注入用药安全提示（婴幼儿/孕妇更强提示），
且不误伤膳食用量（钾/钠 mg、蔬菜克、饮水毫升）。
"""
from meditriage.swarm.swarm_coordinator import _ensure_medication_safety


def test_infant_med_strong_warning():
    out = _ensure_medication_safety(
        "我家3个月宝宝发烧喂多少布洛芬",
        "布洛芬常规剂量为 5-10 mg/kg/次，每6小时一次",
    )
    assert out.startswith("【用药安全】")
    assert "婴幼儿" in out and ("就医" in out or "儿科" in out)


def test_adult_drug_dose_footer():
    out = _ensure_medication_safety(
        "二甲双胍吃多少",
        "二甲双胍起始剂量每天500毫克，可增至2000毫克",
    )
    assert "【用药安全】" in out
    assert "医生或药师" in out
    assert out.startswith("二甲双胍")  # 原文在前，提示在后


def test_dietary_quantity_not_flagged():
    ans = "高血压患者每日钾摄入3500-5000mg，蔬菜500克，饮水2000毫升"
    # 无药名、非按公斤剂量 → 不应被误判为处方
    assert _ensure_medication_safety("高血压饮食", ans) == ans


def test_no_dose_unchanged():
    ans = "多休息，多喝水，注意保暖。"
    assert _ensure_medication_safety("感冒怎么办", ans) == ans


def test_pediatric_suspension_ml_detected():
    """儿科混悬液按 ml 给药——此前单位表漏 ml 整类漏检。"""
    out = _ensure_medication_safety(
        "我家宝宝发烧美林怎么喂",
        "美林混悬液每次5ml，每日最多4次",
    )
    assert out.startswith("【用药安全】")


def test_gram_dose_with_drugname_detected():
    out = _ensure_medication_safety(
        "阿莫西林怎么吃",
        "阿莫西林 0.5g 每日三次口服",
    )
    assert "【用药安全】" in out


def test_tablet_count_with_drugname_detected():
    out = _ensure_medication_safety(
        "布洛芬吃几片",
        "布洛芬每次1-2片，每日不超过3次",
    )
    assert "【用药安全】" in out
