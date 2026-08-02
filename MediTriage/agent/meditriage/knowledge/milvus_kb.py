"""医学知识库门面（Milvus + BGE-M3 + reranker，混合检索编排）。

委托 LangChainRAG（命名为历史遗留，实现已不依赖 langchain）。对外接口
（search / delete_collection / count_documents）保持稳定，被 search-knowledge /
clinical-guideline 等 skill 与 medical_memory 复用。检索编排细节见
knowledge/langchain_rag.py。
"""
import threading
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from meditriage.knowledge.langchain_rag import LangChainRAG, MILVUS_URI, DEFAULT_COLLECTION
except ImportError:
    from langchain_rag import LangChainRAG, MILVUS_URI, DEFAULT_COLLECTION


class MedicalKnowledgeBase:
    """医学知识库（单例；内部委托 LangChainRAG）。"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        uri: str = MILVUS_URI,
        collection_name: str = DEFAULT_COLLECTION,
        device: str = "cuda:2",
    ):
        if getattr(self, "_initialized", False):
            return
        with self._lock:
            if getattr(self, "_initialized", False):
                return
            self._rag = LangChainRAG(
                uri=uri,
                collection_name=collection_name,
                device=device,
            )
            self.collection_name = collection_name
            self.uri = uri
            self._initialized = True
            logger.info("MedicalKnowledgeBase ready (BGE-M3 + Milvus)")

    # ---- 对外接口 ----

    def search(
        self, query: str, top_k: int = 3, filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """语义检索 + rerank。返回 [{id, content, metadata, score}]。"""
        return self._rag.search(query, top_k=top_k, filter_type=filter_type)

    def delete_collection(self):
        """删除 collection（用于重建/测试）。"""
        try:
            if self._rag.client.has_collection(self.collection_name):
                self._rag.client.drop_collection(self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"delete_collection failed: {e}")

    def count_documents(self) -> int:
        """统计 chunk 数量。"""
        try:
            return self._rag.client.get_collection_stats(
                self.collection_name
            ).get("row_count", 0)
        except Exception as e:
            logger.warning(f"count_documents failed: {e}")
            return 0
