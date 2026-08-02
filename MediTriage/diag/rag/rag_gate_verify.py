"""Gate 验证：量化检索证据的质量。

对一组临床问题查询索引的 top-3，judge 每个检索块：① 是否有效（非无效
内容）② 是否与问题相关。报告检索块的有效率与相关率，并与旧基线 54%
（retrieval_ab）对比。

用法（容器内注入 key）：
  docker exec -e DEEPSEEK_API_KEY=$(cat ~/.config/deepseek_api_key) medix-fix \
    python3 /workspace/MediTriage/diag/rag_gate_verify.py
"""
import sys, json, re
# 路径引导：向上定位 MediTriage 根（含 agent/meditriage/paths.py），从任意目录可运行。
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
from meditriage.knowledge.langchain_rag import LangChainRAG
from judge_lib import judge_deepseek

QUESTIONS = [
    "2型糖尿病的一线降糖药和 HbA1c 目标", "急性冠脉综合征的抗血小板治疗",
    "严重主动脉瓣狭窄的干预指征", "慢阻肺急性加重如何处理",
    "房颤 CHA2DS2-VASc 评分与抗凝", "高血压一线降压药物分类",
    "慢性肾病 KDIGO 分期与管理", "肥厚型心肌病的治疗要点",
    "他汀治疗的 LDL-C 目标值", "急性肺栓塞危险分层与抗凝",
    "哮喘 GINA 阶梯治疗", "HFrEF 心衰四联药物",
    "What antiplatelet therapy for acute coronary syndrome?",
    "First-line antihypertensive drug classes",
    "Statin LDL-C target in dyslipidemia",
]


def judge(question, chunk):
    msg = [{"role": "user", "content": (
        f"医学问题：{question}\n检索到的知识块：\n{chunk[:700]}\n\n"
        "判断该块：useful=是否是有效医学知识(非目录/刊头/参考文献/碎片)；relevant=是否与问题相关可支撑作答。\n"
        '只输出 JSON：{"useful": true/false, "relevant": true/false}'
    )}]
    try:
        r = (judge_deepseek(msg) or "").lower().replace(" ", "")
        u = '"useful":true' in r
        rel = '"relevant":true' in r
        return u, rel
    except Exception:
        return None, None


def main():
    rag = LangChainRAG(use_hybrid=True)
    uu = rr = tot = 0
    for q in QUESTIONS:
        hits = rag.search(q, top_k=3)
        for h in hits:
            u, rel = judge(q, h.get("content", ""))
            if u is not None:
                uu += int(u); rr += int(rel); tot += 1
        print(
            f"  Q「{q[:24]}」hits={len(hits)} 累计 useful={uu} rel={rr} / {tot}",
            flush=True,
        )
    print("\n" + "=" * 56)
    print(f"检索块有效率: {uu}/{tot} = {uu/tot:.1%}" if tot else "no hits")
    print(
        f"检索块相关率: {rr}/{tot} = {rr/tot:.1%}（旧基线 retrieval_ab≈54%）"
        if tot else ""
    )


if __name__ == "__main__":
    main()
