"""路由决策：计算每篇 docling 清洗后的有效块率，低于阈值的篇标记为需改用 marker 重解析。

用法（容器内注入 key）：
  docker exec -e DEEPSEEK_API_KEY=$(cat ~/.config/deepseek_api_key) medix-fix \
    python3 /workspace/MediTriage/diag/rag_route_decide.py
输出：每篇有效率 + 低于阈值的 stem 列表（逗号分隔，作为
parse_corpus --parser marker --only ... 的输入）。
"""
import sys
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
# clean_chunk 居于 rag_corpus，引导块未覆盖
_sys.path.insert(0, str(_paths.DATA_DIR / "rag_corpus"))
from judge_lib import judge_deepseek
from clean_chunk import clean_markdown, chunk_markdown

DOCLING = (_paths.DATA_DIR / "rag_corpus/_parsed/docling")
N = 12
THRESH = 0.65


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
    return (list(range(n)) if n <= k
            else [round(i * (n - 1) / (k - 1)) for i in range(k)])


def main():
    rows, collapse = [], []
    for f in sorted(DOCLING.glob("*.md")):
        ch = chunk_markdown(
            clean_markdown(f.read_text(encoding="utf-8"), lang="en"),
            {"source": f.stem},
        )
        if not ch:
            rows.append((f.stem, 0.0, 0, 0))
            collapse.append(f.stem)
            continue
        u = t = 0
        for i in even(len(ch), N):
            r = judge_useful(ch[i]["content"])
            if r is not None:
                u += int(r)
                t += 1
        rate = u / t if t else 0
        rows.append((f.stem, rate, len(ch), t))
        if rate < THRESH:
            collapse.append(f.stem)
        print(f"  {f.stem:34} 有效率 {rate:4.0%}  ({len(ch)}块)", flush=True)
    print("\n" + "=" * 60)
    print(f"阈值 {THRESH:.0%}。需 marker 救的篇（{len(collapse)}）：")
    print(",".join(collapse) if collapse else "(无，docling 全部达标)")


if __name__ == "__main__":
    main()
