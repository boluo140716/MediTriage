"""RAG badcase 聚类选题。

badcase.py 把线上低相关/空检索 miss 落到 log/badcase/rag_misses.jsonl；
本脚本把这些 query 用 BGE-M3 嵌入后按相似度聚类，输出高频缺口主题簇 + 代表 query
+ 命中分数分布，作为定向扩语料（中文指南 / MedlinePlus 主题）的选题依据。

纯只读分析：读 jsonl + 嵌入，不改任何索引/文件。query 可能含健康信息，结果只在
本机 log/ 留存（已 .gitignore），不外发。

用法（容器内）：
  docker exec medix-fix bash -c "cd /workspace/MediTriage/agent && \
    python3 /workspace/MediTriage/diag/rag/badcase_cluster.py [--sim 0.6] [--min-cluster 2]"
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ASK = next(
    p for p in Path(__file__).resolve().parents
    if (p / "agent" / "meditriage" / "paths.py").is_file()
)
if str(_ASK / "agent") not in sys.path:
    sys.path.insert(0, str(_ASK / "agent"))
import meditriage.paths as _paths  # noqa: E402

MISSES = _paths.LOG_DIR / "badcase" / "rag_misses.jsonl"


def _load():
    if not MISSES.is_file():
        return []
    out = []
    for line in MISSES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _cluster(queries, embeds, sim_thresh):
    """贪心单链聚类：与已有簇质心余弦相似度 ≥ 阈值则并入，否则新建簇。"""
    import numpy as np
    clusters = []  # [{idxs, centroid}]
    for i, v in enumerate(embeds):
        v = np.asarray(v, dtype="float32")
        placed = False
        for c in clusters:
            if float(np.dot(v, c["centroid"])) >= sim_thresh:
                c["idxs"].append(i)
                m = len(c["idxs"])
                c["centroid"] = (c["centroid"] * (m - 1) + v) / m
                placed = True
                break
        if not placed:
            clusters.append({"idxs": [i], "centroid": v})
    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=float, default=0.6, help="同簇余弦相似度阈值")
    ap.add_argument("--min-cluster", type=int, default=2, help="报告的最小簇大小")
    args = ap.parse_args()

    rows = _load()
    if not rows:
        print(f"无 badcase 记录（{MISSES}）。线上积累一定量 miss 后再跑。")
        return

    reasons = Counter(r.get("reason", "?") for r in rows)
    print(f"miss 总数 {len(rows)}；按原因：{dict(reasons)}\n")

    queries = [r.get("query", "") for r in rows]
    from meditriage.knowledge.langchain_rag import LangChainRAG
    rag = LangChainRAG(use_hybrid=False, use_query_rewrite=False)
    embeds = [rag.embeddings.embed_query(q) for q in queries]

    clusters = _cluster(queries, embeds, args.sim)
    clusters.sort(key=lambda c: len(c["idxs"]), reverse=True)

    print(f"=== 缺口主题簇（sim≥{args.sim}，≥{args.min_cluster} 条）===")
    shown = 0
    for c in clusters:
        if len(c["idxs"]) < args.min_cluster:
            continue
        shown += 1
        idxs = c["idxs"]
        scores = [
            (rows[i].get("top") or [{}])[0].get("score", 0.0)
            for i in idxs if rows[i].get("top")
        ]
        avg = sum(scores) / len(scores) if scores else 0.0
        rep = queries[idxs[0]]
        print(f"\n[{len(idxs)} 条 · top1 均分 {avg:.3f}] 代表：{rep[:50]}")
        for i in idxs[1:5]:
            print(f"    - {queries[i][:50]}")
    if not shown:
        print(f"（暂无 ≥{args.min_cluster} 条的簇；多为零散 miss）")
        for c in clusters[:8]:
            print(f"    · {queries[c['idxs'][0]][:50]}")


if __name__ == "__main__":
    main()
