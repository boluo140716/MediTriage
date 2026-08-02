"""长期记忆：跨会话相似案例检索。

委托 memory.medical_memory.MedicalMemory，复用本项目 Milvus
（medical-milvus:19530）+ BGE-M3 的本地长期记忆（collection: agent_memory）。
写入统一由 swarm 落库（_persist_long_term）负责，本类只提供检索入口
（search_similar_sessions / .enabled）。
"""
import os
from typing import Dict, List, Any, Optional
from loguru import logger

# user_id（env MEDIX_MEMORY_USER 可覆盖以隔离测试/生产；默认单租户 medix_user）
DEFAULT_USER_ID = os.environ.get("MEDIX_MEMORY_USER", "medix_user")

try:
    from .medical_memory import MedicalMemory
except ImportError:
    from medical_memory import MedicalMemory


class LongTermMemory:
    """长期记忆检索门面（委托本地 MedicalMemory：Milvus + BGE-M3）。

    会话开始前语义检索相似历史案例；写入由 swarm 落库统一处理。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化长期记忆管理器（config 参数保留以兼容旧签名）。
        """
        self.user_id = (config or {}).get("user_id", DEFAULT_USER_ID)
        try:
            self._memory = MedicalMemory()
            self.enabled = self._memory.enabled
            if self.enabled:
                logger.info(
                    "LongTermMemory initialized with local Milvus "
                    "(agent_memory) + BGE-M3"
                )
            else:
                logger.warning(
                    "LongTermMemory disabled: MedicalMemory unavailable"
                )
        except Exception as e:
            logger.warning(f"Failed to initialize local long-term memory: {e}")
            self._memory = None
            self.enabled = False

    def search_similar_sessions(
        self,
        query: str,
        limit: int = 5,
        exclude_session: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        语义检索相似的历史会话（按 user_id 隔离）。

        Args:
            query: 查询文本（通常是用户问题）
            limit: 返回结果数量
            exclude_session: 排除该会话自己的记忆——本会话内容已在短期
                记忆里，再经长期记忆召回等于重复注入还可能盖过新话题
            user_id: 登录身份（隔离各用户记忆）；无则退回实例默认值

        Returns:
            相似会话列表，每个含 memory_id、content、score、metadata、timestamp
        """
        if not self.enabled or self._memory is None:
            return []

        try:
            hits = self._memory.search_memory(
                user_id=user_id or self.user_id, query=query,
                top_k=limit + 2 if exclude_session else limit,
            )
            formatted_results = []
            for h in hits:
                meta = h.get("metadata", {})
                if (exclude_session
                        and meta.get("session_id") == exclude_session):
                    continue
                formatted_results.append({
                    "memory_id": h.get("id", "unknown"),
                    "content": h.get("content", ""),
                    "score": h.get("score", 0.0),
                    "metadata": meta,
                    "timestamp": meta.get("timestamp"),
                })
            formatted_results = formatted_results[:limit]
            logger.info(
                f"Found {len(formatted_results)} similar sessions for "
                f"query: {query[:50]}..."
            )
            return formatted_results
        except Exception as e:
            logger.error(f"Failed to search similar sessions: {e}")
            return []
