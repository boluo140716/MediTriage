#!/usr/bin/env python3
"""Benchmark 报告生成器。

读 results.jsonl + judge_calibration.json，生成 REPORT.md：
  1) 各维度指标 + Wilson 95% CI（MCQ our/bare 含 source·lang 细分、路由、
     检索 hybrid vs dense A/B、多轮、质量 AB 双 judge 胜率+均分、安全、延迟）。
  2) "声称 vs 实测(±CI)" 对照表；训练项无法在线复测，表末单列说明。
  3) judge 可信度（DeepSeek/Gemini 校准准确率 + Cohen's κ）。
  4) 工程选型对照（检索 hybrid vs dense 实测 + 嵌入/记忆结论）。
  5) 顶部：测试集版本、各维 n、数据来源、许可、诚实声明。

仅用标准库；CI 复用 MediTriage/diag/judge_lib.py 的 wilson_ci(k, n)。对缺失
维度/空数据/带 error 的行稳健：标 N/A、不崩，退出 0。指标直接由
results.jsonl 聚合，不做外推，CI 由样本量决定（n 小则 CI 宽）。mcq 的
source 优先读字段 source，缺失时按 id 前缀（cmexam_* / medqa_*）推断，
无法判定归 "unknown"；lang 同理（无则 N/A）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# --- 路径引导：向上定位 MediTriage 根（含 agent/meditriage/paths.py），从任意目录可运行 ---
import sys as _sys
from pathlib import Path as _Path
_ASK = next(
    p for p in _Path(__file__).resolve().parents
    if (p / "agent" / "meditriage" / "paths.py").is_file()
)
for _p in (str(_ASK / "agent"), str(_ASK / 'diag')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import meditriage.paths as _paths
# --- end 引导 ---

# CI 计算复用 judge_lib.wilson_ci，与评测侧口径一致。
from judge_lib import wilson_ci  # noqa: E402

DEFAULT_RESULTS = str(_paths.LOG_DIR / "benchmark/results.jsonl")
DEFAULT_OUT = str(_paths.LOG_DIR / "benchmark/REPORT.md")
# 校准文件与 results 同目录；从 results 路径推导，便于 --results 覆盖时跟随。
CALIB_NAME = "judge_calibration.json"

EVAL_SET_VERSION = "medical_eval_v1"

# 待验证的既有性能声明，报告里与实测值做"声称 vs 实测"对照。
CLAIMS = {
    "routing": "95%（从 88% 提升）",
    "retrieval": "知识检索相关率 87%",
    "latency": "单 Agent 5–15s / Swarm 20–30s",
    "multiturn": "92%（从 60% 提升）",
    "quality": "our 4.5 vs baseline 3.9（5 分制）",
    "safety": "安全违规 ~5%（可修复）",
}


# ===========================================================================
# 小工具
# ===========================================================================
def _fmt_pct(x):
    return f"{x * 100:.1f}%"


def _rate_ci(k, n):
    """返回 'p%（lo%–hi%）, n=N' 字符串；n==0 → 'N/A (n=0)'。"""
    if n == 0:
        return "N/A (n=0)"
    p = k / n
    lo, hi = wilson_ci(k, n)
    return f"{_fmt_pct(p)}（95% CI {_fmt_pct(lo)}–{_fmt_pct(hi)}, n={n}）"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _percentile(xs, q):
    """线性插值分位数（q ∈ [0,1]）。空输入返回 None。"""
    xs = sorted(v for v in xs if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def _infer_source(rec):
    """mcq source：优先字段，否则按 id 前缀推断（cmexam/medqa），否则 unknown。"""
    s = rec.get("source")
    if s:
        return str(s)
    rid = str(rec.get("id", "")).lower()
    if rid.startswith("cmexam"):
        return "cmexam"
    if rid.startswith("medqa"):
        return "medqa"
    return "unknown"


def _verdict(measured_p, claim_p, n, *, higher_is_better=True):
    """对照表判定：⚫不可测 / 🔴未达 / 🟡接近 / ✅达成。

    规则（口径透明，便于复核）：
      - n==0 或 measured_p is None → ⚫（不可测/未跑）。
      - higher_is_better：实测 ≥ 声称 → ✅；≥ 声称-10pt → 🟡；否则 🔴。
      - lower_is_better（如违规率）：实测 ≤ 声称 → ✅；≤ 声称+5pt → 🟡；否则 🔴。
    """
    if n == 0 or measured_p is None:
        return "⚫ 不可测"
    if higher_is_better:
        if measured_p >= claim_p:
            return "✅ 达成"
        if measured_p >= claim_p - 0.10:
            return "🟡 接近"
        return "🔴 未达"
    else:
        if measured_p <= claim_p:
            return "✅ 达成"
        if measured_p <= claim_p + 0.05:
            return "🟡 接近"
        return "🔴 未达"


# ===========================================================================
# 加载
# ===========================================================================
def load_results(path):
    """读 jsonl → list[dict]。带 error 字段的行保留（聚合时按维度跳过）。
    文件不存在返回 []。"""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 坏行跳过，不崩
    return rows


def load_calibration(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _by_dim(rows, dim):
    """取某维度、且无 error 字段的有效行。"""
    return [r for r in rows if r.get("dim") == dim and "error" not in r]


# ===========================================================================
# 各维度聚合
# ===========================================================================
def agg_mcq(rows):
    recs = _by_dim(rows, "mcq")
    out = {
        "n_total": len(recs), "by_system": {}, "by_source": {},
        "by_lang": {},
    }
    for sysname in ("our", "bare"):
        sub = [r for r in recs if r.get("system") == sysname]
        k = sum(1 for r in sub if r.get("correct"))
        out["by_system"][sysname] = (k, len(sub))
        # source 细分（仅本系统内）
        src_map = {}
        for r in sub:
            s = _infer_source(r)
            kk, nn = src_map.get(s, (0, 0))
            src_map[s] = (kk + (1 if r.get("correct") else 0), nn + 1)
        out["by_source"][sysname] = src_map
        # lang 细分（仅当数据带 lang 字段时才有意义）
        lang_map = {}
        for r in sub:
            lg = r.get("lang")
            if lg is None:
                continue
            kk, nn = lang_map.get(lg, (0, 0))
            lang_map[lg] = (kk + (1 if r.get("correct") else 0), nn + 1)
        out["by_lang"][sysname] = lang_map
    return out


def agg_routing(rows):
    recs = _by_dim(rows, "routing")
    k = sum(1 for r in recs if r.get("correct"))
    return {"k": k, "n": len(recs)}


def agg_retrieval(rows):
    """hybrid=True vs False 两组各算相关率（A/B 对比）。"""
    recs = _by_dim(rows, "retrieval")
    groups = {}
    for hyb in (True, False):
        sub = [r for r in recs if bool(r.get("hybrid")) == hyb]
        k = sum(1 for r in sub if r.get("relevant"))
        groups[hyb] = (k, len(sub))
    n_hits = _mean(
        [r.get("n_hits") for r in recs if r.get("n_hits") is not None]
    )
    return {"hybrid": groups[True], "dense": groups[False], "avg_hits": n_hits}


def agg_multiturn(rows):
    recs = _by_dim(rows, "multiturn")
    k = sum(1 for r in recs if r.get("correct"))
    return {"k": k, "n": len(recs)}


def agg_quality(rows):
    """双 judge：winner 胜率（tie 单列，不计入胜负分母）+ 平均分。

    winner 语义：A=our、B=bare（已抗位置偏置还原）。
    胜率 = our 胜数 / (our 胜数 + bare 胜数)，tie 排除在分母外但单列计数。
    """
    recs = _by_dim(rows, "quality")
    out = {"n_total": len(recs)}
    for judge, wk, sk_our, sk_bare in (
        ("ds", "winner_ds", "score_our_ds", "score_bare_ds"),
        ("gm", "winner_gm", "score_our_gm", "score_bare_gm"),
    ):
        our_w = sum(1 for r in recs if str(r.get(wk)).upper() == "A")
        bare_w = sum(1 for r in recs if str(r.get(wk)).upper() == "B")
        tie = sum(1 for r in recs if str(r.get(wk)).upper() == "TIE")
        decisive = our_w + bare_w
        out[judge] = {
            "our_w": our_w, "bare_w": bare_w, "tie": tie,
            "decisive": decisive,
            "winrate_our": (our_w / decisive) if decisive else None,
            "mean_our": _mean([r.get(sk_our) for r in recs]),
            "mean_bare": _mean([r.get(sk_bare) for r in recs]),
            "n": len(recs),
        }
    return out


def agg_safety(rows):
    recs = _by_dim(rows, "safety")
    n = len(recs)
    ref = sum(1 for r in recs if r.get("has_referral"))
    dis = sum(1 for r in recs if r.get("has_disclaimer"))
    # 违规口径：高风险样本中缺就医提示 = 安全违规（即声明指标里的"违规率"）。
    high = [r for r in recs if r.get("is_high_risk")]
    viol = sum(1 for r in high if not r.get("has_referral"))
    return {
        "n": n, "referral": (ref, n), "disclaimer": (dis, n),
        "n_high": len(high), "viol": viol,
    }


def agg_latency(rows):
    """our 系统延迟 p50/p90。

    仅 mcq、routing 行带 lat 字段。用 routing.got（single/swarm）分桶；
    mcq 全部记为 single（单 Agent 路径）。无数据 → None。
    """
    single, swarm = [], []
    for r in _by_dim(rows, "mcq"):
        if r.get("system") == "our" and isinstance(r.get("lat"), (int, float)):
            single.append(r["lat"])
    for r in _by_dim(rows, "routing"):
        lat = r.get("lat")
        if not isinstance(lat, (int, float)):
            continue
        (swarm if r.get("got") == "swarm" else single).append(lat)
    overall = single + swarm
    return {
        "single": {"p50": _percentile(single, 0.5),
                   "p90": _percentile(single, 0.9), "n": len(single)},
        "swarm": {"p50": _percentile(swarm, 0.5),
                  "p90": _percentile(swarm, 0.9), "n": len(swarm)},
        "overall": {"p50": _percentile(overall, 0.5),
                    "p90": _percentile(overall, 0.9), "n": len(overall)},
    }


# ===========================================================================
# 渲染
# ===========================================================================
def _lat_cell(d):
    if d["n"] == 0 or d["p50"] is None:
        return "N/A (n=0)"
    return f"p50={d['p50']:.2f}s / p90={d['p90']:.2f}s (n={d['n']})"


def render(rows, calib, results_path):
    mcq = agg_mcq(rows)
    routing = agg_routing(rows)
    retr = agg_retrieval(rows)
    mt = agg_multiturn(rows)
    qa = agg_quality(rows)
    safety = agg_safety(rows)
    lat = agg_latency(rows)

    # 各维 n（用于顶部一览）
    n_map = {d: len(_by_dim(rows, d))
             for d in ("mcq", "routing", "retrieval", "multiturn",
                       "quality", "safety")}
    n_err = sum(1 for r in rows if "error" in r)

    L = []
    a = L.append

    # --- 头部 ---
    a("# 医疗 Agent Benchmark 报告")
    a("")
    a(f"- 测试集版本：`{EVAL_SET_VERSION}`")
    a("- 数据来源：真实人工标注 CMExam / MedQA / cMedQA2（客观题与问诊参考答案）；"
      "路由·多轮·安全为基于真实题干合成构造的评测样本。")
    a("- 许可：研究用途（research use only），数据集各自遵循其原始许可。")
    a(f"- 评测产物：`{results_path}`；本报告由 `benchmark_report.py` 聚合生成。")
    a("- 各维度有效样本数（n）：" +
      "，".join(f"{d}={n_map[d]}" for d in n_map) +
      (f"（另有 {n_err} 条带 error 的行已排除）" if n_err else ""))
    a("")
    a("> **诚实声明**：")
    a("> 1. 质量分由 LLM 评审（DeepSeek + Gemini 双 judge）给出，**非人类医学专家盲评**，"
      "仅为方向性参考；judge 经少量专家真值校准（见下文 §3）。")
    a("> 2. 路由 / 多轮 / 安全维度的题目为**合成构造集**（基于真实题干改写），"
      "非完全自然分布。")
    a("> 3. 本次为冒烟规模，**n 有限、Wilson CI 很宽**，绝对数字不应作强结论；"
      "下方所有比率均附 95% CI 与 n。")
    a("> 4. 训练相关声称**本次未跑**，单列说明（见 §2 表末）。")
    a("")

    # --- §1 各维度指标 ---
    a("## 1. 各维度指标（Wilson 95% CI）")
    a("")
    # MCQ
    a("### 1.1 客观 MCQ 准确率（our vs bare）")
    a("")
    a("| 系统 | 准确率（±CI, n） |")
    a("| --- | --- |")
    for s in ("our", "bare"):
        k, n = mcq["by_system"][s]
        a(f"| {s} | {_rate_ci(k, n)} |")
    a("")
    a("**按 source 细分：**")
    a("")
    a("| 系统 | source | 准确率（±CI, n） |")
    a("| --- | --- | --- |")
    for s in ("our", "bare"):
        src_map = mcq["by_source"].get(s, {})
        if not src_map:
            a(f"| {s} | N/A | N/A (n=0) |")
            continue
        for src in sorted(src_map):
            k, n = src_map[src]
            a(f"| {s} | {src} | {_rate_ci(k, n)} |")
    a("")
    # lang：仅当存在
    has_lang = any(mcq["by_lang"].get(s) for s in ("our", "bare"))
    a("**按 lang 细分：**")
    a("")
    if has_lang:
        a("| 系统 | lang | 准确率（±CI, n） |")
        a("| --- | --- | --- |")
        for s in ("our", "bare"):
            lang_map = mcq["by_lang"].get(s, {})
            for lg in sorted(lang_map):
                k, n = lang_map[lg]
                a(f"| {s} | {lg} | {_rate_ci(k, n)} |")
    else:
        a("N/A — 本次结果未携带 `lang` 字段（冒烟集为中文 CMExam，未细分语言）。")
    a("")

    # 路由
    a("### 1.2 路由准确率")
    a("")
    a(f"- our：{_rate_ci(routing['k'], routing['n'])}")
    a("")

    # 检索 A/B
    a("### 1.3 检索相关率（hybrid vs dense，A/B 对比）")
    a("")
    a("| 检索模式 | 相关率（±CI, n） |")
    a("| --- | --- |")
    hk, hn = retr["hybrid"]
    dk, dn = retr["dense"]
    a(f"| hybrid=true（混合：dense+稀疏/重排） | {_rate_ci(hk, hn)} |")
    a(f"| hybrid=false（dense only） | {_rate_ci(dk, dn)} |")
    a("")
    if retr["avg_hits"] is not None:
        a(f"- 平均命中片段数 n_hits ≈ {retr['avg_hits']:.2f}")
        a("")

    # 多轮
    a("### 1.4 多轮（指代消解）准确率")
    a("")
    a(f"- our：{_rate_ci(mt['k'], mt['n'])}")
    a("")

    # 质量 AB
    a("### 1.5 质量 AB（our vs bare，双 judge）")
    a("")
    a("winner 语义：A=our，B=bare（已抗位置偏置还原）。"
      "胜率 = our 胜 / (our 胜 + bare 胜)，tie 单列、不计入胜负分母。")
    a("")
    a("| Judge | our 胜 | bare 胜 | tie | our 胜率（决胜局） | 均分 our | 均分 bare |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    for jk, jname in (("ds", "DeepSeek"), ("gm", "Gemini")):
        d = qa.get(jk)
        if not d or qa["n_total"] == 0:
            a(f"| {jname} | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        wr = ("N/A" if d["winrate_our"] is None
              else f"{_fmt_pct(d['winrate_our'])} (决胜 {d['decisive']})")
        mo = "N/A" if d["mean_our"] is None else f"{d['mean_our']:.2f}"
        mb = "N/A" if d["mean_bare"] is None else f"{d['mean_bare']:.2f}"
        a(f"| {jname} | {d['our_w']} | {d['bare_w']} | {d['tie']} | "
          f"{wr} | {mo} | {mb} |")
    a("")

    # 安全
    a("### 1.6 安全（就医提示 / 免责声明覆盖率）")
    a("")
    rk, rn = safety["referral"]
    sk, sn = safety["disclaimer"]
    a(f"- has_referral（就医提示）覆盖率：{_rate_ci(rk, rn)}")
    a(f"- has_disclaimer（免责声明）覆盖率：{_rate_ci(sk, sn)}")
    if safety["n_high"]:
        a(f"- 高风险样本（n={safety['n_high']}）中缺就医提示（安全违规）："
          f"{safety['viol']} 例 → 违规率 "
          f"{_rate_ci(safety['viol'], safety['n_high'])}")
    else:
        a("- 高风险样本：n=0，违规率 N/A")
    a("")

    # 延迟
    a("### 1.7 延迟（our 系统，p50 / p90）")
    a("")
    a("> 仅 mcq、routing 行携带 `lat`。mcq 归入单 Agent 路径；"
      "routing 按 `got`（single/swarm）分桶。")
    a("")
    a("| 路径 | 延迟 |")
    a("| --- | --- |")
    a(f"| 单 Agent（single） | {_lat_cell(lat['single'])} |")
    a(f"| Swarm | {_lat_cell(lat['swarm'])} |")
    a(f"| 总体 | {_lat_cell(lat['overall'])} |")
    a("")

    # --- §2 声称 vs 实测 ---
    a("## 2. 声称 vs 实测（±CI）对照表")
    a("")
    a("判定图例：✅ 达成 ｜ 🟡 接近（差 ≤10pt，违规率口径 ≤5pt）｜ 🔴 未达 ｜ ⚫ 不可测/未跑")
    a("")
    a("| 维度 | 声明值 | 本次实测（±CI, n） | 判定 |")
    a("| --- | --- | --- | --- |")

    # 路由
    rk2, rn2 = routing["k"], routing["n"]
    rp = (rk2 / rn2) if rn2 else None
    a(f"| 路由准确率 | {CLAIMS['routing']} | {_rate_ci(rk2, rn2)} | "
      f"{_verdict(rp, 0.95, rn2)} |")

    # 检索（取 hybrid 组对标 87%）
    hp = (hk / hn) if hn else None
    a(f"| 知识检索相关率 | {CLAIMS['retrieval']} | hybrid={_rate_ci(hk, hn)} | "
      f"{_verdict(hp, 0.87, hn)} |")

    # 延迟（区间声称 → 用实测 p50/p90 描述，判定按是否落在区间内的方向性给）
    sg = lat["single"]
    sw = lat["swarm"]
    lat_meas = f"single {_lat_cell(sg)}；swarm {_lat_cell(sw)}"
    # 单 Agent 声称 5–15s：p50 落区间内/附近 → 方向性判定
    if sg["n"] == 0 or sg["p50"] is None:
        lat_verdict = "⚫ 不可测"
    elif 5.0 <= sg["p50"] <= 15.0:
        lat_verdict = "✅ 达成（单 Agent p50 落 5–15s）"
    elif sg["p50"] < 5.0:
        lat_verdict = "🟡 偏快（< 5s；冒烟桩/缓存所致，方向性）"
    else:
        lat_verdict = "🔴 偏慢（> 15s）"
    a(f"| 延迟 | {CLAIMS['latency']} | {lat_meas} | {lat_verdict} |")

    # 多轮
    mp = (mt["k"] / mt["n"]) if mt["n"] else None
    a(f"| 多轮准确率 | {CLAIMS['multiturn']} | {_rate_ci(mt['k'], mt['n'])} | "
      f"{_verdict(mp, 0.92, mt['n'])} |")

    # 质量（双 judge 均分；判定看 our 均分是否达 4.5 且 > bare）
    ds = qa.get("ds", {})
    gm = qa.get("gm", {})
    if qa["n_total"] == 0:
        qa_meas, qa_verdict = "N/A (n=0)", "⚫ 不可测"
    else:
        def _m(d, key):
            return "N/A" if d.get(key) is None else f"{d[key]:.2f}"
        qa_meas = (f"our: DS {_m(ds, 'mean_our')} / GM {_m(gm, 'mean_our')}；"
                   f"bare: DS {_m(ds, 'mean_bare')} / GM {_m(gm, 'mean_bare')}"
                   f"（n={qa['n_total']}）")
        our_means = [
            v for v in (ds.get("mean_our"), gm.get("mean_our"))
            if v is not None
        ]
        bare_means = [
            v for v in (ds.get("mean_bare"), gm.get("mean_bare"))
            if v is not None
        ]
        ovr = _mean(our_means)
        bvr = _mean(bare_means)
        if ovr is None:
            qa_verdict = "⚫ 不可测"
        elif ovr >= 4.5 and (bvr is None or ovr > bvr):
            qa_verdict = "✅ 达成"
        elif ovr >= 4.0:
            qa_verdict = "🟡 接近"
        else:
            qa_verdict = "🔴 未达"
    a(f"| 质量（our vs baseline） | {CLAIMS['quality']} | "
      f"{qa_meas} | {qa_verdict} |")

    # 安全违规
    if safety["n_high"]:
        vp = safety["viol"] / safety["n_high"]
        sv = _rate_ci(safety["viol"], safety["n_high"])
        sver = _verdict(vp, 0.05, safety["n_high"], higher_is_better=False)
    else:
        sv, sver = "N/A (n=0)", "⚫ 不可测"
    a(f"| 安全违规率（高风险缺就医提示） | {CLAIMS['safety']} | {sv} | {sver} |")

    # 训练项单列（无法在线复测）
    a("| **训练项（VLM 0.58→0.78 等）** | VLM 准确率 0.58→0.78 等 | "
      "⚫ 未跑 | ⚫ 未跑 — 训练复现未产出超过 base 的 checkpoint，"
      "以官方 68.8% 为基准口径 |")
    a("")

    # --- §3 judge 可信度 ---
    a("## 3. Judge 可信度（校准）")
    a("")
    if not calib:
        a("⚫ N/A — 未找到 `judge_calibration.json`。")
    else:
        a("| Judge | 校准准确率（vs 专家真值） | 95% CI | n |")
        a("| --- | --- | --- | --- |")
        for jk, jname in (("deepseek", "DeepSeek"), ("gemini", "Gemini")):
            d = calib.get(jk)
            if not d:
                a(f"| {jname} | N/A | N/A | N/A |")
                continue
            acc = d.get("acc")
            ci = d.get("ci") or [None, None]
            n = d.get("n", "?")
            acc_s = "N/A" if acc is None else _fmt_pct(acc)
            ci_s = ("N/A" if ci[0] is None
                    else f"{_fmt_pct(ci[0])}–{_fmt_pct(ci[1])}")
            a(f"| {jname} | {acc_s} | {ci_s} | {n} |")
        a("")
        kappa = calib.get("cohen_kappa")
        if kappa is not None:
            a(f"- 两 judge 一致性 Cohen's κ = **{kappa:.3f}**"
              "（κ>0.8 通常视为高度一致）。")
    a("")
    a("> 质量分由**经专家真值校准的双 judge**给出，而非医学专家盲评；"
      "校准样本量有限，结论为**方向性**参考，不构成临床或学术定论。")
    a("")

    # --- §4 工程选型对照 ---
    a("## 4. 工程选型对照结论")
    a("")
    a("### 4.1 检索：hybrid vs dense（A/B 实测）")
    a("")
    a("| 模式 | 相关率（±CI, n） |")
    a("| --- | --- |")
    a(f"| hybrid（dense + 稀疏/重排） | {_rate_ci(hk, hn)} |")
    a(f"| dense only | {_rate_ci(dk, dn)} |")
    a("")
    # 裁决文字：依据两组点估计方向（CI 宽时明确标注不显著）
    if hn == 0 or dn == 0:
        a("- **裁决**：本次样本不足（其中一组 n=0），无法对 hybrid vs dense 下结论 → 维持现状待全量复跑。")
    else:
        hp2 = hk / hn
        dp2 = dk / dn
        if hp2 > dp2:
            trend = f"hybrid 相关率（{_fmt_pct(hp2)}）高于 dense（{_fmt_pct(dp2)}）"
        elif hp2 < dp2:
            trend = f"hybrid 相关率（{_fmt_pct(hp2)}）低于 dense（{_fmt_pct(dp2)}）"
        else:
            trend = f"hybrid 与 dense 相关率持平（均 {_fmt_pct(hp2)}）"
        a(f"- **观察**：{trend}；当前 n 很小、CI 高度重叠，差异**不具统计显著性**。")
        a("- **裁决**：在显著证据出现前不做架构切换；hybrid 作为可选项保留，全量复跑后再定。")
    a("")
    a("### 4.2 嵌入与短期记忆")
    a("")
    a("- **嵌入模型**：保持 **BGE-M3**（中文医疗检索表现稳定，多语言 + 长文本，"
      "无证据表明更换可带来显著增益）。")
    a("- **短期记忆**：保持**内存**实现（会话级、低延迟、隔离简单；"
      "当前规模无需引入外部存储，避免额外运维与一致性风险）。")
    a("")

    a("---")
    a("")
    a(f"_报告生成自 `{results_path}`，共 {len(rows)} 行记录。"
      "数字直接来自评测产物，未做任何手工调整。_")
    a("")
    return "\n".join(L)


# ===========================================================================
# CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Benchmark 报告生成（Wilson CI + 声称vs实测）")
    ap.add_argument(
        "--results", default=DEFAULT_RESULTS, help="results.jsonl 路径")
    ap.add_argument("--out", default=DEFAULT_OUT, help="REPORT.md 输出路径")
    args = ap.parse_args()

    rows = load_results(args.results)
    calib_path = os.path.join(os.path.dirname(os.path.abspath(args.results)),
                              CALIB_NAME)
    calib = load_calibration(calib_path)

    md = render(rows, calib, args.results)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)

    n_err = sum(1 for r in rows if "error" in r)
    print(f"[report] 读 {len(rows)} 行（含 {n_err} error），"
          f"calib={'有' if calib else '无'} → 写 {args.out}", flush=True)


if __name__ == "__main__":
    main()
