"""重建 RAG 索引 v2：docling/marker 解析 → clean_chunk 去样板+结构化分块 → 直接入库。

路由：某篇若存在 _parsed/marker/<stem>.md（docling 解析失败时改用 marker 的
产出），优先用 marker；否则用 docling。
中文 txt 本就是纯正文，轻量清洗后同样做结构化分块。

幂等：先 drop 旧 collection。
运行（容器内）：
  docker exec medix-fix bash -c "cd /workspace/MediTriage/agent && python3 /workspace/MediTriage/data/rag_corpus/build_rag_index_v2.py"
"""
import json, sys
from pathlib import Path

# 路径引导：向上定位 MediTriage 根（含 agent/meditriage/paths.py），从任意目录可运行。
import sys as _sys
from pathlib import Path as _Path
_ASK = next(
    p for p in _Path(__file__).resolve().parents
    if (p / 'agent' / 'meditriage' / 'paths.py').is_file()
)
for _p in (str(_ASK / 'agent'), str(_Path(__file__).resolve().parent)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import meditriage.paths as _paths
from clean_chunk import clean_markdown, chunk_markdown, zh_type_from_stem

CORPUS = (_paths.DATA_DIR / "rag_corpus")
PARSED = CORPUS / "_parsed"
ZH = CORPUS / "local_zh"
MILVUS_URI = _paths.MILVUS_URI
COLLECTION = "medical_knowledge_m3"

# 元数据 manifest（org/topic/year）
meta_by_id = {}
for name in ("source_manifest.json",):
    p = CORPUS / "guidelines" / name
    if p.exists():
        for e in json.load(open(p)).get("entries", []):
            meta_by_id[e["id"]] = e

all_chunks = []
docling_dir, marker_dir = PARSED / "docling", PARSED / "marker"

# 指南：marker 优先（仅 docling 失败的篇目有 marker 产出），否则 docling
for md_path in sorted(docling_dir.glob("*.md")):
    stem = md_path.stem
    mk = marker_dir / (stem + ".md")
    src = mk if mk.exists() else md_path
    parser_used = "marker" if mk.exists() else "docling"
    cleaned = clean_markdown(src.read_text(encoding="utf-8"), lang="en")
    m = meta_by_id.get(stem, {})
    if not m:
        # 文件名 stem 与 manifest id 失配会静默丢掉机构/年份元数据
        #（如 ada_diabetes_2025.pdf vs manifest id ada_standards_2025，
        # 失配的整篇 chunk 引用信息都会为空）——必须显式告警
        print(f"  [WARN] manifest 无 id={stem}：该篇 org/year 将为空，"
              f"请核对文件名与 source_manifest.json 的 id", flush=True)
    meta = {
        "source": m.get("org", "guideline"),
        "topic": m.get("topic", stem),
        "year": m.get("year", ""),
        "lang": "en",
        "type": "clinical_guideline",
        "doc_id": stem,
        "parser": parser_used,
    }
    ch = chunk_markdown(cleaned, meta)
    all_chunks += ch
    print(f"  [{parser_used}] {stem}: {len(ch)} chunks", flush=True)

# 中文本地文档
for txt in sorted(ZH.glob("*.txt")):
    content = txt.read_text(encoding="utf-8")
    if len(content.strip()) < 50:
        continue
    _zt = zh_type_from_stem(txt.stem)
    cleaned = clean_markdown(content, lang="zh")
    meta = {
        "source": "local_zh",
        "topic": txt.stem,
        "lang": "zh",
        "type": _zt,
        "doc_id": txt.stem,
    }
    ch = chunk_markdown(cleaned, meta)
    all_chunks += ch
    print(f"  [zh ] {txt.stem}: {len(ch)} chunks", flush=True)

print(f"\nTOTAL chunks: {len(all_chunks)}")

# 幂等重建
from pymilvus import MilvusClient
_c = MilvusClient(uri=MILVUS_URI)
if _c.has_collection(COLLECTION):
    _c.drop_collection(COLLECTION)
    print(f"Dropped existing collection: {COLLECTION}")
del _c

from meditriage.knowledge.milvus_kb import MedicalKnowledgeBase
kb = MedicalKnowledgeBase(uri=MILVUS_URI, collection_name=COLLECTION)
n = kb._rag.add_prechunked(all_chunks)
print(f"\nINDEXED: {n} chunks; count_documents={kb.count_documents()}")

# 写 manifest
with open(CORPUS / "manifest_v2.jsonl", "w", encoding="utf-8") as f:
    for c in all_chunks:
        f.write(
            json.dumps(
                {"len": len(c["content"]), **c.get("metadata", {})},
                ensure_ascii=False,
            )
            + "\n"
        )
print(f"Manifest: {CORPUS}/manifest_v2.jsonl")
