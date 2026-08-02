"""检索确定性度量（topic 金标准，不依赖 LLM judge）。

动机：retrieval_ab.py 的 24 题 top-3 二值 + DeepSeek 单次判定对检索改动无分辨力
（hybrid=dense=54.2%）。本脚本用「每题期望 topic」与「chunk metadata.topic」做确定性匹配，
算 Recall@k / MRR / nDCG@k，秒级可复跑、零随机，作为检索改动的主度量。

评测矩阵：{rewrite on/off} × {hybrid on/off} 四格分别报，隔离 BM25 与改写各自的贡献
（rewrite=off 走干净的 1:1 融合路径，可复现）。

用法（容器内）：
  docker exec medix-fix bash -c "cd /workspace/MediTriage/agent && \
    python3 /workspace/MediTriage/diag/rag/retrieval_metrics.py [--quick] [--out PATH]"
  --quick 只跑 rewrite=off 两格（不调 vLLM，最快，用于逐项 A/B）。
输出：stdout 指标表 + log/benchmark/retrieval_metrics.jsonl（每题命中明细）。
"""
import argparse
import json
import math
import sys
from pathlib import Path

_ASK = next(
    p for p in Path(__file__).resolve().parents
    if (p / "agent" / "meditriage" / "paths.py").is_file()
)
for _p in (str(_ASK / "agent"), str(_ASK / "diag")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import meditriage.paths as _paths  # noqa: E402
from meditriage.knowledge.langchain_rag import LangChainRAG  # noqa: E402

OUT = str(_paths.LOG_DIR / "benchmark/retrieval_metrics.jsonl")

# 每题：(query, 期望 topic 金标准集合)。guideline 块的 topic 即疾病名，
# 糖尿病指南 topic 为 doc_id 形式 ada_diabetes_2025。命中=检回块 topic ∈ 金标准集。
GOLD = {
    "diabetes": {"ada_diabetes_2025", "21_guideline_diabetes",
                 "02_lifestyle_diabetes", "diabetes_type_2"},
    "acs": {"acs"},
    "vhd": {"vhd", "heart_valve_diseases"},
    "copd": {"copd", "chronic_bronchitis", "emphysema"},
    "afib": {"afib", "atrial_fibrillation", "arrhythmia"},
    "hypertension": {"hypertension", "20_guideline_hypertension",
                     "01_lifestyle_hypertension", "high_blood_pressure"},
    "ckd": {"ckd", "chronic_kidney_disease"},
    "cardiomyopathy": {"cardiomyopathy"},
    "lipids": {"lipids", "cholesterol", "ldl_the_bad_cholesterol"},
    "pe": {"pe", "pulmonary_embolism"},
    "asthma": {"asthma", "asthma_in_children"},
    "hf": {"hf", "heart_failure"},
}

QUESTIONS = [
    ("2型糖尿病患者血糖控制的一线药物和 HbA1c 目标是什么？", "diabetes"),
    ("What is the first-line pharmacologic therapy and HbA1c target for type 2 diabetes?", "diabetes"),
    ("急性冠脉综合征（ACS）患者的抗血小板治疗方案有哪些？", "acs"),
    ("What antiplatelet therapy is recommended for acute coronary syndrome?", "acs"),
    ("严重主动脉瓣狭窄的干预指征是什么？", "vhd"),
    ("What are the indications for intervention in severe aortic stenosis?", "vhd"),
    ("慢阻肺（COPD）急性加重的处理原则是什么？", "copd"),
    ("How is a COPD exacerbation managed according to GOLD?", "copd"),
    ("房颤患者的抗凝治疗如何根据 CHA2DS2-VASc 评分决定？", "afib"),
    ("How does CHA2DS2-VASc score guide anticoagulation in atrial fibrillation?", "afib"),
    ("高血压的诊断标准和一线降压药物有哪些？", "hypertension"),
    ("What are the first-line antihypertensive drug classes?", "hypertension"),
    ("慢性肾病（CKD）的分期和管理要点是什么？", "ckd"),
    ("What are the KDIGO staging and management principles for chronic kidney disease?", "ckd"),
    ("肥厚型心肌病的诊断和治疗要点是什么？", "cardiomyopathy"),
    ("What is the management approach for hypertrophic cardiomyopathy?", "cardiomyopathy"),
    ("血脂异常患者他汀治疗的 LDL-C 目标值是多少？", "lipids"),
    ("What is the LDL-C target for statin therapy in dyslipidemia?", "lipids"),
    ("急性肺栓塞的危险分层和抗凝治疗如何选择？", "pe"),
    ("How is acute pulmonary embolism risk-stratified and anticoagulated?", "pe"),
    ("哮喘的阶梯式药物治疗方案是怎样的？", "asthma"),
    ("What is the stepwise pharmacologic treatment for asthma per GINA?", "asthma"),
    ("射血分数降低的心力衰竭（HFrEF）的四联药物治疗是什么？", "hf"),
    ("What is the four-pillar therapy for heart failure with reduced ejection fraction?", "hf"),
]

# 域外盲区题：医学相关但不在语料（指南+MedlinePlus）覆盖范围内。期望检索 top1
# 低于阈值 → 系统应坦诚弃权（not_found），而非把弱命中误当证据。衡量"盲区被正确
# 识别"的能力（弃权率应高），与上面的域内召回互补。
OOD_QUESTIONS = [
    "蛇毒血清的批号管理与冷链规范",
    "航天员失重环境下骨密度丢失的对抗训练方案",
    "深海潜水减压病高压氧舱的治疗压力参数",
    "PPG 光电容积脉搏波与 EEG 脑电信号的本质区别",
    "法医毒理学中河豚毒素的定量检测方法",
    "兽医临床犬瘟热的疫苗免疫程序",
]


def _rels(hits, gold_set):
    """每个命中是否相关（topic ∈ 金标准集）→ 0/1 列表。"""
    out = []
    for h in hits:
        topic = (h.get("metadata") or {}).get("topic", "")
        out.append(1 if topic in gold_set else 0)
    return out


def _metrics(rels, k_recall=3):
    """从 0/1 相关列表算 Recall@1/@3、MRR、nDCG@3。
    Recall 这里是命中导向：top-k 内是否至少一个相关（题级 hit-rate）。"""
    hit1 = 1.0 if rels[:1] and rels[0] else 0.0
    hitk = 1.0 if any(rels[:k_recall]) else 0.0
    mrr = 0.0
    for i, r in enumerate(rels):
        if r:
            mrr = 1.0 / (i + 1)
            break
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k_recall]))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k_recall, sum(rels) or 1)))
    ndcg = dcg / ideal if ideal else 0.0
    return hit1, hitk, mrr, ndcg


def run_cell(rag, rewrite, hybrid, top_k, writer):
    rag.use_hybrid = hybrid
    agg = [0.0, 0.0, 0.0, 0.0]
    n = 0
    for i, (q, topic) in enumerate(QUESTIONS):
        hits = rag.search(q, top_k=top_k, rewrite=rewrite)
        rels = _rels(hits, GOLD[topic])
        m = _metrics(rels)
        for j in range(4):
            agg[j] += m[j]
        n += 1
        writer.write(json.dumps({
            "id": f"retr_{i:02d}", "topic": topic,
            "rewrite": rewrite, "hybrid": hybrid,
            "rels": rels,
            "hit_topics": [(h.get("metadata") or {}).get("topic", "") for h in hits[:3]],
            "hit1": m[0], "hit3": m[1], "mrr": round(m[2], 4), "ndcg3": round(m[3], 4),
        }, ensure_ascii=False) + "\n")
    return [a / n for a in agg], n


def run_ood(rag, threshold=0.30):
    """域外题弃权评测：top1 分数 < 阈值视为正确弃权（盲区被识别）。"""
    rag.use_hybrid = True
    abstain = 0
    rows = []
    for q in OOD_QUESTIONS:
        hits = rag.search(q, top_k=3, rewrite=False)
        top1 = hits[0].get("score", 0.0) if hits else 0.0
        ok = top1 < threshold
        abstain += int(ok)
        rows.append((q[:24], round(top1, 3), "弃权✓" if ok else "误召回✗"))
    return abstain, len(OOD_QUESTIONS), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="只跑 rewrite=off 两格")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rag = LangChainRAG(use_hybrid=True)
    combos = [(False, True), (False, False)]
    if not args.quick:
        combos = [(True, True), (True, False), (False, True), (False, False)]

    print(f"{'cell':24} {'Hit@1':>7} {'Hit@3':>7} {'MRR':>7} {'nDCG@3':>7}")
    print("-" * 58)
    with open(args.out, "w", encoding="utf-8") as w:
        for rewrite, hybrid in combos:
            (h1, h3, mrr, ndcg), n = run_cell(rag, rewrite, hybrid, args.top_k, w)
            label = f"rw={int(rewrite)} hybrid={int(hybrid)}"
            print(f"{label:24} {h1:7.3f} {h3:7.3f} {mrr:7.3f} {ndcg:7.3f}")
    print(f"\n明细写入 {args.out}（n={n} 题/格，top_k={args.top_k}）")

    ab, total, rows = run_ood(rag)
    print(f"\n=== 域外盲区弃权率：{ab}/{total} 正确识别（top1<0.30）===")
    for q, s, verdict in rows:
        print(f"  {verdict}  top1={s:<6} {q}")


if __name__ == "__main__":
    main()
