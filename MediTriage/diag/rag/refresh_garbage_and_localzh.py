"""一次性维护：清理索引垃圾块 + 刷新 local_zh 语料。

做两件事（涉及删除 Milvus 记录，须经用户确认后运行）：
1. 删除 medical_knowledge_m3 中 9 个期刊 front-matter 块（刊头/编委名单，
   检索出口已兜底不返回，此处做物理清理）；
2. 删除 local_zh 两个"诊疗要点"文档的现有 chunk，按当前文件
   用与 build_rag_index_v2 完全相同的管线（clean_markdown → chunk_markdown
   → add_prechunked）重新入库，保证 chunk 形态与全库一致。

用法（容器内）：
  docker exec medix-fix python3 /workspace/MediTriage/diag/rag/refresh_garbage_and_localzh.py
跑完重启 web（BM25 索引在进程启动时构建，需重启才感知刷新）。
"""
import json
import re
import sys
from pathlib import Path

_ASK = next(
    p for p in Path(__file__).resolve().parents
    if (p / 'agent' / 'meditriage' / 'paths.py').is_file()
)
sys.path.insert(0, str(_ASK / 'agent'))
# clean_chunk 与 build 脚本同目录，按文件名直接 import
sys.path.insert(0, str(_ASK / 'data' / 'rag_corpus'))

from pymilvus import MilvusClient  # noqa: E402

from meditriage.paths import MILVUS_URI  # noqa: E402
from clean_chunk import (  # noqa: E402
    clean_markdown, chunk_markdown, zh_type_from_stem,
)

COL = "medical_knowledge_m3"
FRONT = re.compile(
    r"DEPUTY EDITORS|EDITOR[\s-]?IN[\s-]?CHIEF|EDITORIAL BOARD"
    r"|ASSOCIATE EDITORS|PRINT ISSN|ONLINE ISSN"
    r"|THE JOURNAL OF CLINICAL AND APPLIED RESEARCH",
    re.I,
)
LOCAL_REFRESH = ("20_guideline_hypertension", "21_guideline_diabetes")
ZH_DIR = _ASK / "data" / "rag_corpus" / "local_zh"


def main():
    c = MilvusClient(uri=MILVUS_URI)

    # 1) 扫出要删的两类 chunk
    front_ids, local_ids = [], []
    it = c.query_iterator(
        collection_name=COL, batch_size=1000,
        output_fields=["content", "metadata"],
    )
    while True:
        rows = it.next()
        if not rows:
            break
        for r in rows:
            meta = json.loads(r["metadata"])
            if FRONT.search(r["content"][:300]):
                front_ids.append(r["id"])
            elif meta.get("doc_id") in LOCAL_REFRESH:
                local_ids.append(r["id"])
    it.close()
    print(f"待删 front-matter 垃圾块={len(front_ids)}  "
          f"local_zh 旧 chunk={len(local_ids)}")

    to_delete = front_ids + local_ids
    if to_delete:
        c.delete(collection_name=COL, ids=to_delete)
        c.flush(COL)
        print(f"deleted {len(to_delete)} chunks")

    # 2) 用 build 的同一管线重切 local_zh 两篇
    new_chunks = []
    for stem in LOCAL_REFRESH:
        content = (ZH_DIR / f"{stem}.txt").read_text(encoding="utf-8")
        cleaned = clean_markdown(content, lang="zh")
        meta = {
            "source": "local_zh", "topic": stem, "lang": "zh",
            "type": zh_type_from_stem(stem), "doc_id": stem,
        }
        ch = chunk_markdown(cleaned, meta)
        new_chunks += ch
        print(f"  [zh] {stem}: {len(ch)} chunks")

    from meditriage.knowledge.milvus_kb import MedicalKnowledgeBase
    kb = MedicalKnowledgeBase(uri=MILVUS_URI, collection_name=COL)
    n = kb._rag.add_prechunked(new_chunks)
    print(f"re-ingested {n} chunks; collection count={kb.count_documents()}")
    print("注意：BM25 索引在进程启动时构建，web 服务需重启以感知刷新。")


if __name__ == "__main__":
    main()
