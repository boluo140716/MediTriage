"""把 v2 清洗+分块的真实块导出到本地 markdown，供人工核验切块质量。

指南正文有版权，故写到本地文件而非打印到标准输出。

用法: python3 MediTriage/diag/dump_sample_chunks.py
输出: data/rag_corpus/_bakeoff/SAMPLE_CHUNKS_v2.md
"""
import sys, re
from pathlib import Path
# --- 路径引导：向上定位 MediTriage 根（含 agent/meditriage/paths.py），从任意目录可运行 ---
import sys as _sys
from pathlib import Path as _Path
_ASK = next(
    p for p in _Path(__file__).resolve().parents
    if (p / 'agent' / 'meditriage' / 'paths.py').is_file()
)
for _p in (str(_ASK / 'agent'),):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import meditriage.paths as _paths
# --- end 引导 ---
sys.path.insert(0, str(_paths.DATA_DIR / "rag_corpus"))
from clean_chunk import clean_markdown, chunk_markdown

BAKE = _paths.DATA_DIR / "rag_corpus/_bakeoff/docling"
OUT = _paths.DATA_DIR / "rag_corpus/_bakeoff/SAMPLE_CHUNKS_v2.md"

# 每篇选若干临床锚点块（应有实质内容）+ 若干随机块（如实展示分布）
PICKS = {
    "esc_afib_2024_essential": [
        "anticoagulation", "rate control", "catheter ablation"
    ],
    "gold_copd_2025": [
        "exacerbation", "bronchodilator", "inhaled corticosteroid"
    ],
    "ada_diabetes_2025": ["metformin", "glycemic target", "GLP-1"],
}


def main():
    lines = ["# v2 清洗+分块 真实样本（供核验是否有用文本）\n",
             "> 每篇：先按临床锚点取块（应有料），再附 2 个均匀随机块（诚实展示分布，含可能的残留垃圾）。\n"]
    for stem, anchors in PICKS.items():
        md = clean_markdown(
            (BAKE / (stem + ".md")).read_text(encoding="utf-8"), lang="en"
        )
        chunks = chunk_markdown(md, {"source": stem})
        lines.append(f"\n\n{'='*80}\n## {stem}  （清洗后 {len(chunks)} 块）\n")
        shown = set()
        # 锚点块：跳过前 25%（前置区），取正文区的命中，以观察真实临床内容
        start = len(chunks) // 4
        for anc in anchors:
            for j in range(start, len(chunks)):
                if j in shown:
                    continue
                if re.search(re.escape(anc), chunks[j]["content"], re.I):
                    lines.append(f"\n### [锚点:{anc}] 块#{j} (len={len(chunks[j]['content'])})\n```\n{chunks[j]['content']}\n```")
                    shown.add(j)
                    break
        # 随机均匀块
        for j in [len(chunks)//3, 2*len(chunks)//3]:
            if j not in shown:
                lines.append(f"\n### [随机] 块#{j} (len={len(chunks[j]['content'])})\n```\n{chunks[j]['content']}\n```")
                shown.add(j)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写 {OUT}  （{OUT.stat().st_size//1024} KB）")
    print("用: less", OUT, " 或在编辑器打开查看完整块")


if __name__ == "__main__":
    main()
