import re, sys
from pathlib import Path

# 路径引导：向上定位 MediTriage 根（含 agent/meditriage/paths.py），从任意目录可运行。
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

BAKE = _paths.DATA_DIR / "rag_corpus/_bakeoff"
stem = sys.argv[1] if len(sys.argv) > 1 else "esc_afib_2024_essential"
parsers = ["pdfplumber", "docling", "marker"]

RE_HYPHEN = re.compile(r"[A-Za-z]-\n[A-Za-z]")
RE_CLS = re.compile(r"^\s*(I|II|IIa|IIb|III|IV)\s+[ABC]\s*$", re.M)
RE_REF = re.compile(r"[a-z]\.\d{2,3}")
RE_HEADER = re.compile(r"^#{1,6}\s", re.M)
RE_TABLE = re.compile(r"^\s*\|.*\|\s*$", re.M)

print(f"# 对比文件: {stem}\n")
print(
    f"{'parser':12}{'chars':>9}{'断词':>6}{'等级孤块':>8}"
    f"{'引用黏连':>8}{'#标题':>6}{'|表格行':>8}{'噪声/千':>8}"
)
for p in parsers:
    f = BAKE / p / (stem + ".md")
    if not f.exists():
        print(f"{p:12}  (尚无输出)")
        continue
    t = f.read_text(encoding="utf-8")
    n = max(len(t), 1)
    hy, cl, rf = (
        len(RE_HYPHEN.findall(t)),
        len(RE_CLS.findall(t)),
        len(RE_REF.findall(t)),
    )
    hd, tb = len(RE_HEADER.findall(t)), len(RE_TABLE.findall(t))
    noise = round((hy + cl + rf) / n * 1000, 2)
    print(f"{p:12}{len(t):>9}{hy:>6}{cl:>8}{rf:>8}{hd:>6}{tb:>8}{noise:>8}")

# 抽取含 anticoagulation 的同一段落，对比各解析器输出的可读性
print("\n--- 同一主题段落（含 'anticoagulation'）样例 ---")
for p in parsers:
    f = BAKE / p / (stem + ".md")
    if not f.exists():
        continue
    t = f.read_text(encoding="utf-8")
    m = re.search(r"anticoagulation", t, re.I)
    seg = (
        t[m.start() - 120: m.start() + 260].replace("\n", "⏎")
        if m else "(未找到)"
    )
    print(f"\n[{p}]\n{seg}")
