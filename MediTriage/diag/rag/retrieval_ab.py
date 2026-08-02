"""检索 A/B（语料对齐版）：hybrid vs dense-only 在「答案确在 RAG 语料内」的问题上的相关率。

检索题须贴合 RAG 语料主题：用 CMExam 等不在 13 指南语料覆盖范围内的考题评检索，相关率会虚低为 0。
本脚本用 24 道贴合语料主题（糖尿病/ACS/瓣膜病/COPD/房颤/高血压/CKD/心肌病/血脂/肺栓塞/哮喘/心衰，中英各半）
的检索题，对每题分别用 hybrid 与 dense-only 检索 top-3，由 DeepSeek 判"片段是否相关且可支撑作答"。

用法（容器内，注入 DEEPSEEK_API_KEY）:
  docker exec -e DEEPSEEK_API_KEY=$(cat ~/.config/deepseek_api_key) medix-fix \
    python3 /workspace/MediTriage/diag/retrieval_ab.py
输出: /workspace/log/benchmark/results_retrieval_corpus.jsonl + stdout 相关率对比
"""
import sys, json, os

# --- 路径引导：向上定位 MediTriage 根（含 agent/meditriage/paths.py），从任意目录可运行 ---
import sys as _sys
from pathlib import Path as _Path
_ASK = next(
    p for p in _Path(__file__).resolve().parents
    if (p / 'agent' / 'meditriage' / 'paths.py').is_file()
)
for _p in (str(_ASK / 'agent'), str(_ASK / 'diag')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import meditriage.paths as _paths
# --- end 引导 ---
from meditriage.knowledge.langchain_rag import LangChainRAG
from judge_lib import judge_deepseek, wilson_ci

OUT = str(_paths.LOG_DIR / "benchmark/results_retrieval_corpus.jsonl")
os.makedirs(str(_paths.LOG_DIR / "benchmark"), exist_ok=True)

# 语料对齐检索题（topic 取自 medical_knowledge_m3 实际覆盖；中英各 12）
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


def judge_relevant(question, hits):
    if not hits:
        return False
    snips = "\n".join(f"- {(h.get('content') or '')[:300]}" for h in hits)
    msg = [{"role": "user", "content": (
        f"医学问题：{question}\n\n检索到的资料片段：\n{snips}\n\n"
        "这些片段是否与该问题相关、且包含可用于回答它的医学信息？"
        '只输出 JSON：{"relevant": true} 或 {"relevant": false}'
    )}]
    try:
        r = (judge_deepseek(msg) or "").lower().replace(" ", "")
        if '"relevant":true' in r:
            return True
        if '"relevant":false' in r:
            return False
        return ("true" in r) and ("false" not in r)
    except Exception as e:
        print(f"  judge err: {e}")
        return None


def main():
    rag = LangChainRAG(use_hybrid=True)  # 一个实例，按 use_hybrid 开关切换两路
    if not getattr(rag, "use_hybrid", False):
        print("WARN: hybrid 未启用（BM25 索引可能未建），仍会跑 dense。")
    res = open(OUT, "w", encoding="utf-8")
    tallies = {True: [0, 0], False: [0, 0]}  # hybrid -> [relevant, total]
    for i, (q, topic) in enumerate(QUESTIONS):
        for hybrid in (True, False):
            rag.use_hybrid = hybrid
            hits = rag.search(q, top_k=3)
            rel = judge_relevant(q, hits)
            rec = {
                "dim": "retrieval",
                "id": f"retr_corpus_{i:02d}",
                "topic": topic,
                "hybrid": hybrid,
                "relevant": bool(rel),
                "n_hits": len(hits),
            }
            res.write(json.dumps(rec, ensure_ascii=False) + "\n")
            res.flush()
            if rel is not None:
                tallies[hybrid][0] += int(bool(rel))
                tallies[hybrid][1] += 1
        print(
            f"[{i+1}/{len(QUESTIONS)}] {topic:14} "
            f"hybrid={tallies[True]} dense={tallies[False]}",
            flush=True,
        )
    print("=" * 56)
    for hybrid, name in (
        (True, "hybrid(dense+BM25 RRF+rerank)"),
        (False, "dense-only"),
    ):
        k, n = tallies[hybrid]
        if n:
            lo, hi = wilson_ci(k, n)
            print(
                f"{name:34} 相关率 {k}/{n} = {k/n:.1%}  "
                f"(95% CI {lo:.1%}–{hi:.1%})"
            )
    print(f"\n结果写入 {OUT}")


if __name__ == "__main__":
    main()
