"""本地长期记忆：Milvus agent_memory collection + BGE-M3。

实现：
  - 复用 knowledge.langchain_rag 已加载的 BGE-M3 embeddings（通过
    MedicalKnowledgeBase 单例的内部 LangChainRAG._rag.embeddings），避免二次加载
    1024 维模型占用 GPU。
  - 独立 collection `agent_memory`（与 RAG 语料的 `medical_knowledge_m3` 分开），
    用原生 MilvusClient 直接 create_collection(dim=1024, COSINE) / insert / search。
  - user_id 隔离：检索时 over-fetch 后在 Python 侧按 metadata.user_id 过滤，简单
    稳妥。

向量存储地址：http://medical-milvus:19530（MediTriage/infra/start_medical_milvus.sh）
"""
import json
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from pymilvus import MilvusClient

from meditriage.paths import MILVUS_URI

MEMORY_COLLECTION = "agent_memory"
EMBED_DIM = 1024  # BGE-M3


def _composite_rank(score: float, trust: float = 0.5,
                    age_days: int = 0) -> float:
    """信任感知检索综合分：相似度 × 信任权重 × 时间衰减。"""
    import math
    decay = math.exp(-max(int(age_days), 0) / 90.0)  # ~90 天半衰期量级
    return float(score) * (0.5 + 0.5 * float(trust)) * (0.7 + 0.3 * decay)


class MedicalMemory:
    """本地长期记忆（单例）：Milvus `agent_memory` collection + BGE-M3 embedding。

    方法：
      - add_memory(user_id, content, mtype='episodic', session_id='') -> bool
      - search_memory(user_id, query, top_k=3)
        -> List[{content, score, metadata, id}]

    embedding 复用 MedicalKnowledgeBase 内部的 BGE-M3；若复用失败则自建
    LangChainRAG（会再载一次 BGE-M3，可接受）作为兜底。
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, uri: str = MILVUS_URI,
                 collection_name: str = MEMORY_COLLECTION):
        if getattr(self, "_initialized", False):
            return

        with self._lock:
            if getattr(self, "_initialized", False):
                return

            self.uri = uri
            self.collection_name = collection_name
            self.enabled = False
            self._embeddings = None
            self.client = None

            try:
                self._embeddings = self._load_embeddings()
                self.client = MilvusClient(uri=uri)
                self._ensure_collection()
                self.enabled = True
                self._initialized = True
                logger.info(
                    f"MedicalMemory ready "
                    f"(Milvus {collection_name} + BGE-M3 @ {uri})"
                )
            except Exception as e:
                logger.warning(
                    f"MedicalMemory init failed, long-term memory disabled: {e}"
                )
                self._initialized = True  # 不再重试初始化，避免反复刷日志

    # ---- 内部 ----

    def _load_embeddings(self):
        """优先复用 MedicalKnowledgeBase 已加载的 BGE-M3，避免重复占用 GPU。"""
        try:
            try:
                from meditriage.knowledge.milvus_kb import MedicalKnowledgeBase
            except ImportError:
                from milvus_kb import MedicalKnowledgeBase
            kb = MedicalKnowledgeBase()  # 单例：若已初始化则零开销
            emb = kb._rag.embeddings
            logger.info(
                "MedicalMemory: reusing BGE-M3 embeddings "
                "from MedicalKnowledgeBase"
            )
            return emb
        except Exception as e:
            logger.warning(
                f"MedicalMemory: reuse embeddings failed ({e}), "
                f"loading own LangChainRAG"
            )
            try:
                from meditriage.knowledge.langchain_rag import LangChainRAG
            except ImportError:
                from langchain_rag import LangChainRAG
            # 自建 RAG 实例（会加载一次 BGE-M3）；只取其 embeddings，
            # 存储仍走本类自己的 client
            rag = LangChainRAG(
                collection_name=self.collection_name, use_reranker=False
            )
            return rag.embeddings

    def _ensure_collection(self):
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=EMBED_DIM,
                metric_type="COSINE",
                auto_id=True,
            )
            logger.info(f"Created collection: {self.collection_name}")
        self.client.load_collection(self.collection_name)

    # 不同来源的默认可信度（tool 检索验证 > 用户自述 > Agent 自答）
    TRUST_BY_SOURCE = {
        "tool_verified": 0.9,
        "user_stated": 0.6,
        "agent_generated": 0.3,
    }

    @staticmethod
    def _is_low_value(content: str) -> bool:
        """写入门控：过滤过短/寒暄/无信息内容，不存入长期记忆。"""
        t = (content or "").strip()
        if len(t) < 12:
            return True
        if t.lower() in (
            "你好", "谢谢", "再见", "hi", "hello", "ok", "好的", "嗯", "在吗"
        ):
            return True
        return False

    # ---- 对外 ----

    def add_memory(
        self,
        user_id: str,
        content: str,
        mtype: str = "episodic",
        session_id: str = "",
        source: str = "agent_generated",
        trust: Optional[float] = None,
    ) -> bool:
        """存入一条长期记忆（带写入门控 + 来源/可信度）。

        Args:
            user_id: 用户标识（隔离检索用）
            content: 记忆文本（会话摘要 / 最终答案等）
            mtype: 记忆类型（episodic / semantic ...）
            session_id: 来源会话 ID
            source: 来源信任级别（tool_verified / user_stated / agent_generated）
            trust: 可信度 [0,1]；None 时按 source 取默认

        Returns:
            是否成功写入（被门控跳过返回 False）
        """
        if not self.enabled:
            return False
        if not content or not content.strip():
            return False
        if self._is_low_value(content):
            logger.info(
                f"MedicalMemory: skip low-value content (user={user_id})"
            )
            return False
        if trust is None:
            trust = self.TRUST_BY_SOURCE.get(source, 0.3)

        try:
            vector = self._embeddings.embed_query(content)
            # 去重：同 user_id 下近重复（sim>0.9）跳过，避免同一事实堆叠
            try:
                dup = self.client.search(
                    self.collection_name,
                    data=[vector],
                    limit=10,
                    output_fields=["metadata"],
                )
                for h in (dup[0] if dup else []):
                    try:
                        md = json.loads(
                            h.get("entity", {}).get("metadata", "{}")
                        )
                    except Exception as e:
                        md = {}
                        logger.debug(f"近重复检查 metadata 解析失败: {e}")
                    if (md.get("user_id") == user_id
                            and float(h.get("distance", 0.0)) > 0.9):
                        logger.info(
                            f"MedicalMemory: skip near-duplicate "
                            f"(user={user_id})"
                        )
                        return True  # 视为已知，不新增
            except Exception as e:
                logger.debug(f"近重复检查失败，继续写入: {e}")
            metadata = {
                "user_id": user_id,
                "type": mtype,
                "session_id": session_id,
                "source": source,
                "trust": float(trust),
                "timestamp": datetime.now().isoformat(),
            }
            self.client.insert(
                self.collection_name,
                [{
                    "vector": vector,
                    "content": content,
                    "metadata": json.dumps(metadata, ensure_ascii=False),
                }],
            )
            logger.info(
                f"MedicalMemory: saved (user={user_id}, source={source}, "
                f"trust={trust:.2f}, session={session_id})"
            )
            return True
        except Exception as e:
            logger.error(f"MedicalMemory.add_memory failed: {e}")
            return False

    def search_memory(self, user_id: str, query: str, top_k: int = 3,
                      min_score: float = 0.45) -> List[Dict[str, Any]]:
        """语义检索本用户的长期记忆（按 user_id 隔离 + 相似度阈值过滤）。

        Args:
            user_id: 仅返回该用户的记忆
            query: 查询文本
            top_k: 返回条数
            min_score: 相似度下限（COSINE 归一，低于此值不召回，防低质记忆注入）

        Returns:
            [{id, content, score, metadata}]，metadata 含
            user_id/type/source/trust/session_id/timestamp
        """
        if not self.enabled:
            return []
        if not query or not query.strip():
            return []

        try:
            qv = self._embeddings.embed_query(query)
            # over-fetch 后在 Python 侧按 user_id 过滤，保证隔离且实现简单
            fetch_k = max(top_k * 6, 30)
            res = self.client.search(
                self.collection_name,
                data=[qv],
                limit=fetch_k,
                output_fields=["content", "metadata"],
            )
            hits = res[0] if res else []
            results: List[Dict[str, Any]] = []
            for h in hits:
                ent = h.get("entity", {})
                try:
                    meta = json.loads(ent.get("metadata", "{}"))
                except Exception as e:
                    meta = {}
                    logger.debug(f"记忆检索 metadata 解析失败: {e}")
                if meta.get("user_id") != user_id:
                    continue  # user 隔离
                # COSINE：distance 越大越相似（normalize 后约等于余弦相似度）
                score = float(h.get("distance", 0.0))
                if score < min_score:
                    continue  # 低相关不召回（防低质/无关记忆注入 prompt）
                results.append({
                    "id": str(h.get("id", "")),
                    "content": ent.get("content", ""),
                    "score": score,
                    "metadata": meta,
                })
            # 信任感知重排：综合分 = 相似度 × 信任 × 时间衰减，再截断 top_k
            now = datetime.now()

            def _age_days(md):
                try:
                    return max((now - datetime.fromisoformat(
                        md.get("timestamp", now.isoformat()))).days, 0)
                except Exception:
                    return 0

            results.sort(
                key=lambda r: _composite_rank(
                    r["score"],
                    r["metadata"].get("trust", 0.5),
                    _age_days(r["metadata"]),
                ),
                reverse=True,
            )
            results = results[:top_k]
            logger.info(
                f"MedicalMemory: {len(results)} memories "
                f"(user={user_id}, min_score={min_score}) q={query[:40]!r}"
            )
            return results
        except Exception as e:
            logger.error(f"MedicalMemory.search_memory failed: {e}")
            return []

    def count(self) -> int:
        """collection 行数（调试/自测用）。"""
        if not self.enabled:
            return 0
        try:
            return self.client.get_collection_stats(
                self.collection_name).get("row_count", 0)
        except Exception as e:
            logger.warning(f"MedicalMemory.count failed: {e}")
            return 0
