"""深度研究工作流。

编排多步骤研究流程：查询规划 → 搜索 → 检索 → 综合 → 验证。
"""
from typing import List, Dict, Any, Optional
from loguru import logger
import asyncio

from meditriage.core import LLMClient
from meditriage.knowledge.milvus_kb import MedicalKnowledgeBase
from meditriage.research.evidence_synthesizer import EvidenceSynthesizer, ResearchReport
from meditriage.research.pubmed import search_pubmed

# 全局知识库实例（单例）
_kb_instance = None


def get_knowledge_base():
    """获取知识库单例。"""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = MedicalKnowledgeBase()
    return _kb_instance


class DeepResearchWorkflow:
    """深度研究工作流。

    功能：
    - 多步骤研究流程编排
    - 查询规划和优化
    - 并行知识库检索
    - 证据综合和质量控制

    检索后端为本地 Milvus 知识库（medical-milvus + BGE-M3）与 PubMed 文献
    （NCBI E-utilities）；网络不可达时降级为仅本地。
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        use_knowledge_base: bool = True,
        use_pubmed: bool = True
    ):
        """初始化工作流。

        Args:
            llm_client: LLM 客户端
            use_knowledge_base: 是否使用 Milvus 知识库
            use_pubmed: 是否检索 PubMed 文献（失败自动降级为仅本地）
        """
        self.llm_client = llm_client or LLMClient()
        self.use_knowledge_base = use_knowledge_base
        self.use_pubmed = use_pubmed

        # 使用 Milvus 知识库单例（和其他 Skills 共享，避免重复加载模型）
        self.knowledge_base = (
            get_knowledge_base() if use_knowledge_base else None
        )
        self.synthesizer = EvidenceSynthesizer(llm_client=self.llm_client)

    async def run(
        self,
        question: str,
        max_kb_results: int = 5
    ) -> ResearchReport:
        """执行深度研究。

        Args:
            question: 研究问题
            max_kb_results: 最大知识库检索结果数

        Returns:
            研究报告
        """
        logger.info(f"Starting DeepResearch for: {question}")

        # Step 1: 查询规划
        sub_queries = await self._plan_queries(question)
        logger.info(f"Planned {len(sub_queries)} sub-queries")

        # Step 2: 并行检索 Milvus 知识库
        kb_results: List[Dict[str, Any]] = []

        search_tasks = []

        if self.use_knowledge_base and self.knowledge_base:
            # 从 Milvus 知识库检索
            for query in sub_queries[:3]:
                search_tasks.append(
                    self._search_milvus(
                        query, top_k=max_kb_results // len(sub_queries)
                    )
                )

        # 并行执行
        if search_tasks:
            results = await asyncio.gather(
                *search_tasks, return_exceptions=True
            )

            # 收集结果
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Search task failed: {result}")
                    continue

                if isinstance(result, list):
                    # Milvus 返回的是字典列表
                    kb_results.extend(result)

        logger.info(f"Collected {len(kb_results)} KB results")

        # Step 2b: PubMed 文献检索（权威可引；无网/失败静默降级为仅本地）
        # PubMed 以英文为主，中文问题先转英文检索词，否则常召回为空。
        pubmed_results: List[Dict[str, Any]] = []
        if self.use_pubmed:
            try:
                pq = await self._pubmed_query(question)
                pubmed_results = await asyncio.to_thread(
                    search_pubmed, pq, max_kb_results
                )
            except Exception as e:
                logger.warning(f"PubMed step failed: {e}")
            logger.info(f"Collected {len(pubmed_results)} PubMed results")

        # Step 3: 证据综合
        report = await self.synthesizer.synthesize(
            query=question,
            kb_results=kb_results,
            pubmed_results=pubmed_results
        )
        if not report.key_findings:
            logger.warning("Report has no key findings")

        if not report.summary:
            logger.warning("Report has no summary")

        logger.info("DeepResearch completed")
        return report

    async def _search_milvus(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """从 Milvus 知识库搜索。

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            文档列表（字典格式）
        """
        try:
            results = self.knowledge_base.search(
                query=query, top_k=top_k, filter_type=None
            )
            logger.debug(
                f"Milvus search returned {len(results)} results "
                f"for: {query[:50]}..."
            )
            return results
        except Exception as e:
            logger.error(f"Milvus search failed: {e}")
            return []

    async def _pubmed_query(self, question: str) -> str:
        """把（可能是中文的）问题转成简洁英文 PubMed 检索词。

        PubMed 以英文为主，直接用中文常召回为空；失败则退回原问题。
        """
        try:
            resp = await self.llm_client.chat([{
                "role": "user",
                "content": (
                    "Convert this medical question into a concise English "
                    "PubMed search query (keywords only, no explanation):\n"
                    + question
                ),
            }])
            line = (resp or "").strip().splitlines()
            return line[0].strip() if line and line[0].strip() else question
        except Exception:
            return question

    async def _plan_queries(self, question: str) -> List[str]:
        """查询规划：将复杂问题拆解为多个子查询。

        Args:
            question: 原始问题

        Returns:
            子查询列表
        """
        prompt = f"""你是医学研究助手。请将以下问题拆解为 2-3 个更具体的子查询，以便进行深度研究。

原始问题：{question}

要求：
1. 每个子查询应该聚焦一个特定方面
2. 子查询应该互补，覆盖问题的不同角度
3. 子查询应该简洁明确

输出格式：
每行一个子查询，不需要编号。

示例：
原始问题：2型糖尿病如何治疗？
子查询1：2型糖尿病的药物治疗方案
子查询2：2型糖尿病的生活方式管理
子查询3：2型糖尿病的并发症预防
"""

        try:
            response = await self.llm_client.chat([
                {"role": "user", "content": prompt}
            ])

            # 解析子查询
            lines = response.strip().split('\n')
            sub_queries = []

            for line in lines:
                line = line.strip()
                # 移除可能的编号
                line = line.lstrip('0123456789.-:：）) ')
                if line and len(line) > 5:  # 过滤太短的行
                    sub_queries.append(line)

            # 至少包含原始问题
            if not sub_queries:
                sub_queries = [question]

            # 限制数量
            sub_queries = sub_queries[:3]

            return sub_queries

        except Exception as e:
            logger.error(f"Query planning error: {e}")
            # 降级：返回原始问题
            return [question]
