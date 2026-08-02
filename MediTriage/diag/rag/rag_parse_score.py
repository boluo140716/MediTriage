"""RAG 解析器对比评测——同题对齐评分。

公平性约定：
- 以固定医学锚点定位同一语义段落，在每家输出里截取锚点附近同样大小的
  窗口，确保各解析器评的是同一题（避免按输出中点截取时各家长度不同而比到不同段落）。
- 多锚点、多篇取平均；并记录锚点是否在解析中丢失，作为中立的完整性信号。
- 启发式指标仅作描述性参考，不作裁决依据；这些指标基于已知解析缺陷设定，有偏。
- 评分对裁判隐藏解析器身份，仅提供文本。

用法（容器内，注入 key）：
  docker exec -e DEEPSEEK_API_KEY=$(cat ~/.config/deepseek_api_key) medix-fix \
    python3 /workspace/MediTriage/diag/rag_parse_score.py
"""
import re
import json
from pathlib import Path
import sys

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
from judge_lib import judge_deepseek

BAKE = (_paths.DATA_DIR / "rag_corpus/_bakeoff")
WIN = 380  # 锚点两侧窗口
SKIP_FRONT = 2500  # 跳过目录/关键词前言，落到正文

# 每篇的医学锚点（应出现在正文、三家都该有 → 同题对齐）
ANCHORS = {
    "esc_afib_2024_essential": ["anticoagulation", "CHA2DS2", "cardioversion"],
    "gold_copd_2025": ["exacerbation", "bronchodilator", "spirometry"],
    "ada_diabetes_2025": ["metformin", "HbA1c", "glycemic"],
}

RE_HYPHEN = re.compile(r"[A-Za-z]-\n[A-Za-z]")
RE_CLS = re.compile(r"^\s*(I|II|IIa|IIb|III|IV)\s+[ABC]\s*$", re.M)
RE_REF = re.compile(r"[a-z]\.\d{2,3}")


def heuristics(t):
    return {"chars": len(t), "hyphen": len(RE_HYPHEN.findall(t)),
            "cls": len(RE_CLS.findall(t)), "ref": len(RE_REF.findall(t))}


def window_at(t, anchor):
    """锚点正文首次出现（跳过前言）的对齐窗口；未找到返回 None。"""
    for m in re.finditer(re.escape(anchor), t, re.I):
        if m.start() >= SKIP_FRONT:
            return t[m.start() - WIN: m.start() + WIN]
    m = re.search(re.escape(anchor), t, re.I)  # 回退到任意位置匹配
    return t[m.start() - WIN: m.start() + WIN] if m else None


def judge(passage):
    msg = [{"role": "user", "content": (
        "下面是从医学指南 PDF 自动提取的一段文本。评估它作为'可检索知识片段'的质量："
        "语句是否通顺连贯、能否读懂相关医学信息，有无双栏交错串读、断词、参考文献编号乱入、表格碎片、乱码。\n\n"
        f"片段：\n{passage}\n\n"
        '只输出 JSON：{"score": <1-5整数>, "reason": "<20字内>"}。'
        "5=干净连贯可直接用；3=可读但有杂质；1=严重乱序不可用。"
    )}]
    try:
        r = judge_deepseek(msg) or ""
        mm = re.search(r"\{.*\}", r, re.S)
        if mm:
            o = json.loads(mm.group(0))
            return int(o.get("score", 0)), str(o.get("reason", ""))[:36]
    except Exception as e:
        return 0, f"err:{e}"
    return 0, "parse fail"


def main():
    parsers = sorted(
        [p.name for p in BAKE.iterdir() if p.is_dir()]
    ) if BAKE.exists() else []
    print("解析器：", parsers)
    agg = {p: [] for p in parsers}
    found = {p: 0 for p in parsers}
    total_anchor_slots = 0

    for stem, anchors in ANCHORS.items():
        # 至少有一家产出该篇才进行比较
        if not any((BAKE / p / (stem + ".md")).exists() for p in parsers):
            continue
        print("\n" + "=" * 80 + f"\n# {stem}")
        # 描述性启发式
        for p in parsers:
            f = BAKE / p / (stem + ".md")
            if f.exists():
                h = heuristics(f.read_text(encoding="utf-8"))
                print(
                    f"  [{p:11}] chars={h['chars']:>8} "
                    f"断词={h['hyphen']:>4} 等级块={h['cls']:>4} "
                    f"引用黏={h['ref']:>4}"
                )
        # 同题对齐评分
        for anchor in anchors:
            total_anchor_slots += 1
            row = {}
            for p in parsers:
                f = BAKE / p / (stem + ".md")
                if not f.exists():
                    row[p] = ("—", "无该篇产出"); continue
                w = window_at(f.read_text(encoding="utf-8"), anchor)
                if not w:
                    row[p] = ("✗", "锚点丢失"); continue
                found[p] += 1
                s, why = judge(w)
                agg[p].append(s)
                row[p] = (s, why)
            print(f"  锚点「{anchor}」同题评分:")
            for p in parsers:
                v, why = row[p]
                print(f"      {p:11} {str(v):>3}  {why}")

    print("\n" + "=" * 80 + "\n# 汇总（同题对齐，公平）")
    print(f"{'parser':12}{'平均连贯(↑)':>12}{'锚点命中':>10}{'样本n':>7}")
    for p in parsers:
        s = agg[p]
        avg = sum(s) / len(s) if s else 0
        print(
            f"{p:12}{avg:>12.2f}{found[p]:>7}/"
            f"{total_anchor_slots}{len(s):>7}"
        )


if __name__ == "__main__":
    main()
