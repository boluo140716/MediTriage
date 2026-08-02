"""衡量清洗的真实价值：块级"垃圾率"。

针对"检索出来的块很多没用"这个问题，度量的核心指标是分块后垃圾块的占比：
语料切块后，有多少块是垃圾（目录/刊头/广告/参考文献/双栏串读碎片），多少是可用的临床知识块。
对比：raw docling 分块 vs cleaned docling 分块，各随机抽样，DeepSeek 判 useful y/n。

用法（容器内注入 key）：
  docker exec -e DEEPSEEK_API_KEY=$(cat ~/.config/deepseek_api_key) medix-fix \
    python3 /workspace/MediTriage/diag/rag_chunk_junkrate.py
"""
import sys, re, json
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

BAKE = (_paths.DATA_DIR / "rag_corpus/_bakeoff/docling")
STEMS = ["esc_afib_2024_essential", "gold_copd_2025", "ada_diabetes_2025"]
N_SAMPLE = 15  # 每篇均匀抽样块数（控制 API 调用量）


def _even_idx(n, k):
    """在 [0,n) 上均匀取 k 个下标。"""
    if n <= k:
        return list(range(n))
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def judge_useful(chunk):
    msg = [{"role": "user", "content": (
        "下面是医疗 RAG 知识库中的一个检索块。判断它是否是**可用于回答临床问题的有效医学知识**，"
        "还是**垃圾块**（目录/页眉页脚/刊头/广告/编辑名单/参考文献列表/双栏串读碎片/无意义符号）。\n\n"
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


def run(label, get_md):
    useful = total = 0
    for stem in STEMS:
        md = get_md(stem)
        chunks = chunk_markdown(md, {"source": stem})
        idxs = _even_idx(len(chunks), N_SAMPLE)
        for i in idxs:
            r = judge_useful(chunks[i]["content"])
            if r is not None:
                useful += int(r)
                total += 1
        print(f"  [{label}] {stem}: {len(chunks)} 块, 抽 {len(idxs)}", flush=True)
    rate = useful / total if total else 0
    print(f"==> {label}: 有效块率 {useful}/{total} = {rate:.1%}")
    return useful, total, rate


def main():
    raw = lambda s: (BAKE / (s + ".md")).read_text(encoding="utf-8")
    cleaned = lambda s: clean_markdown(raw(s), lang="en")
    print("# RAW docling 分块")
    r1 = run("raw", raw)
    print("\n# CLEANED docling 分块")
    r2 = run("cleaned", cleaned)
    print("\n" + "=" * 56)
    print(f"原始分块   有效块率: {r1[2]:.1%} ({r1[0]}/{r1[1]})")
    print(f"清洗后分块 有效块率: {r2[2]:.1%} ({r2[0]}/{r2[1]})")
    print(f">>> 提升: {(r2[2]-r1[2])*100:+.1f} 个百分点")


if __name__ == "__main__":
    main()
