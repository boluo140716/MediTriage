"""扩充入库（增量，不 drop）：把 _expanded/<源>/*.txt 清洗+结构化分块后追加到现有索引。
当前源：medlineplus（NLM 公共领域健康主题）。type=health_topic，source=medlineplus。
经 search_knowledge（不限 type）可检索到，补发热/感染/外伤/皮肤/急诊等盲区。

用法（容器内）：
  docker exec medix-fix bash -c "cd /workspace/MediTriage/agent && python3 /workspace/MediTriage/data/rag_corpus/ingest_expanded.py --source medlineplus"
"""
import argparse, sys
from pathlib import Path

# 路径引导：向上定位 MediTriage 根（含 agent/meditriage/paths.py），从任意目录可运行
import sys as _sys
from pathlib import Path as _Path
_ASK = next(
    p for p in _Path(__file__).resolve().parents
    if (p / "agent" / "meditriage" / "paths.py").is_file()
)
for _p in (str(_ASK / "agent"), str(_Path(__file__).resolve().parent)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import meditriage.paths as _paths
# 路径引导结束
from clean_chunk import clean_markdown, chunk_markdown

CORPUS = (_paths.DATA_DIR / "rag_corpus")
MILVUS_URI = _paths.MILVUS_URI
COLLECTION = "medical_knowledge_m3"

SRC_META = {
    "medlineplus": {
        "source": "medlineplus",
        "type": "health_topic",
        "lang": "en",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="medlineplus", choices=list(SRC_META))
    a = ap.parse_args()
    srcdir = CORPUS / "_expanded" / a.source
    base_meta = SRC_META[a.source]

    all_chunks = []
    files = sorted(
        p for p in srcdir.glob("*.txt") if not p.name.startswith("_")
    )
    for f in files:
        content = f.read_text(encoding="utf-8")
        if len(content.strip()) < 120:
            continue
        cleaned = clean_markdown(content, lang=base_meta.get("lang", "en"))
        meta = {**base_meta, "topic": f.stem, "doc_id": f"{a.source}:{f.stem}"}
        all_chunks += chunk_markdown(cleaned, meta)
    print(f"{a.source}: {len(files)} 文件 → {len(all_chunks)} 块")

    from meditriage.knowledge.milvus_kb import MedicalKnowledgeBase
    kb = MedicalKnowledgeBase(uri=MILVUS_URI, collection_name=COLLECTION)
    before = kb.count_documents()
    n = kb._rag.add_prechunked(all_chunks)
    from pymilvus import MilvusClient
    MilvusClient(uri=MILVUS_URI).flush(COLLECTION)
    print(f"追加 {n} 块；入库前 {before} → 现 {kb.count_documents()}")


if __name__ == "__main__":
    main()
