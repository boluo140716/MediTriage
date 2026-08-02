"""混合路由探针：判断哪些篇章应改用 marker 解析。

对已有 3 篇逐篇计算：
- 表密度（docling md 中表格行/推荐表标记占比）—— 作为是否表密集的廉价判据。
- 清洗后有效块率：docling_cleaned vs marker_cleaned —— 看 marker 在该篇是否更优。

用法（容器内注入 key）：
  docker exec -e DEEPSEEK_API_KEY=$(cat ~/.config/deepseek_api_key) medix-fix \
    python3 /workspace/MediTriage/diag/rag_hybrid_probe.py
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
for _p in (str(_ASK / 'agent'), str(_ASK / 'diag')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import meditriage.paths as _paths
# --- end 引导 ---
sys.path.insert(0, str(_paths.DATA_DIR / "rag_corpus"))
from judge_lib import judge_deepseek
from clean_chunk import clean_markdown, chunk_markdown

BAKE = (_paths.DATA_DIR / "rag_corpus/_bakeoff")
STEMS = ["esc_afib_2024_essential", "gold_copd_2025", "ada_diabetes_2025"]
N = 12

_RE_TABLE = re.compile(r"\|")
_RE_RECO = re.compile(
    r"\bClass\s+[Ia]+\b|\bLevel\s+[ABC]\b|Recommendation", re.I
)


def table_density(md: str) -> float:
    lines = [l for l in md.splitlines() if l.strip()]
    if not lines:
        return 0.0
    hit = sum(1 for l in lines if _RE_TABLE.search(l) or _RE_RECO.search(l))
    return hit / len(lines)


def judge_useful(chunk):
    msg = [{"role": "user", "content": (
        "下面是医疗 RAG 知识库中的一个检索块。判断它是否是**可用于回答临床问题的有效医学知识**，"
        "还是**垃圾块**（目录/页眉/刊头/广告/参考文献/双栏串读碎片/无意义符号）。\n\n"
        f"块内容：\n{chunk[:700]}\n\n"
        '只输出 JSON：{"useful": true} 或 {"useful": false}。'
    )}]
    try:
        r = (judge_deepseek(msg) or "").lower().replace(" ", "")
        if '"useful":true' in r:
            return True
        if '"useful":false' in r:
            return False
        return ("true" in r) and ("false" not in r)
    except Exception:
        return None


def even(n, k):
    return list(range(n)) if n <= k else [
        round(i*(n-1)/(k-1)) for i in range(k)
    ]


def useful_rate(parser, stem):
    f = BAKE / parser / (stem + ".md")
    if not f.exists():
        return None
    md = clean_markdown(f.read_text(encoding="utf-8"), lang="en")
    ch = chunk_markdown(md, {"source": stem})
    u = t = 0
    for i in even(len(ch), N):
        r = judge_useful(ch[i]["content"])
        if r is not None:
            u += int(r)
            t += 1
    return (u, t, u/t if t else 0, len(ch))


def main():
    print(
        f"{'doc':28}{'表密度':>8}{'docling有效率':>16}"
        f"{'marker有效率':>16}{'marker增益':>10}"
    )
    for stem in STEMS:
        dens = table_density(
            (BAKE / "docling" / (stem + ".md")).read_text(encoding="utf-8")
        )
        d = useful_rate("docling", stem)
        m = useful_rate("marker", stem)
        gain = (m[2] - d[2]) * 100 if (d and m) else 0
        print(f"{stem:28}{dens:>8.1%}{d[2]:>14.0%}({d[3]})"
              f"{m[2]:>13.0%}({m[3]}){gain:>+9.1f}pt")


if __name__ == "__main__":
    main()
