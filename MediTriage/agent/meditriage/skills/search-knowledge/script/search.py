"""搜索医学知识库 Skill（自包含）。"""
from typing import Dict, Any
from loguru import logger

# 全局知识库实例（避免重复加载模型）
_kb_instance = None


def get_knowledge_base():
    """获取知识库单例。"""
    global _kb_instance
    if _kb_instance is None:
        from meditriage.knowledge.milvus_kb import MedicalKnowledgeBase
        _kb_instance = MedicalKnowledgeBase()
    return _kb_instance


async def search_knowledge(
    query: str, max_results: int = 5
) -> Dict[str, Any]:
    """搜索医学知识库。

    Args:
        query: 查询内容。
        max_results: 最多返回结果数（默认 5）。

    Returns:
        {
            "answer": "格式化的知识库检索结果",
            "total_found": 检索到的结果数,
            "query": "原始查询"
        }
    """
    import asyncio

    logger.info(
        f"Searching knowledge base: query={query}, "
        f"max_results={max_results}"
    )

    # KB 冷启动与检索都是重型同步操作（GPU embedding / rerank / BM25），
    # 放线程池执行，避免卡死 FastAPI 事件循环（并发会话 SSE 全局冻结）。
    # Milvus 故障时优雅降级——不把原始连接异常串透传给 LLM。
    try:
        kb = await asyncio.to_thread(get_knowledge_base)
        results = await asyncio.to_thread(
            kb.search, query=query, top_k=max_results, filter_type=None
        )
    except Exception as e:
        logger.warning(f"知识库不可用，降级回答: {e}")
        from meditriage.knowledge.badcase import log_rag_miss
        log_rag_miss(query, None, route="search", reason="unavailable")
        return {
            "answer": (
                "【知识库暂不可用】本次无法完成检索。请基于通用医学常识"
                "谨慎作答，并明确告知用户'本次回答未经知识库证据支持'；"
                "严禁编造具名指南、机构或年份。"
            ),
            "total_found": 0,
            "query": query,
            "not_found": True,
        }

    from meditriage.knowledge.badcase import log_rag_miss

    # 相关性阈值：reranker 概率分（实测相关≈0.99 / 无关≈0.002）低于阈值视为
    # 弱命中不喂给 LLM；全军覆没则坦诚弃权，禁止编造来源。阈值统一定义在 knowledge
    # 层单一常量（env 可调）
    from meditriage.knowledge.langchain_rag import RAG_MIN_SCORE
    min_score = RAG_MIN_SCORE
    strong = [d for d in results if d.get("score", 0) >= min_score]
    if results and not strong:
        log_rag_miss(query, results, route="search", reason="low_relevance")
        return {
            "answer": (
                f"【知识库未收录】检索到的内容与'{query}'相关性都很低，"
                "不足以作为回答依据。请据此处理：① 不要再以相近措辞重复"
                "检索本工具；② 可基于通用医学常识谨慎作答，但必须注明"
                "'此处未检索到可靠证据'；③ 严禁编造具名指南、机构或年份。"
            ),
            "total_found": 0,
            "query": query,
            "not_found": True,
        }
    results = strong

    # 格式化结果（来源标注用索引里真实存在的 metadata 字段组装）
    from meditriage.knowledge.citations import build_citations, source_label

    formatted_results = []
    for doc in results:
        formatted_results.append({
            "source": source_label(doc["metadata"]),
            "section": doc["metadata"].get("section", ""),
            "content": doc["content"],
            "score": doc["score"],
        })

    if formatted_results:
        # 灰区采样：top1 落在 [0.30, 0.45) 时结果照常返回，但留痕——
        # 这类"勉强命中"是排序/语料问题的早期信号
        if results and results[0].get("score", 0) < 0.45:
            log_rag_miss(query, results, route="search", reason="borderline")
        return {
            "answer": format_results(formatted_results),
            "total_found": len(formatted_results),
            "query": query,
            "citations": build_citations(results, query=query),
        }
    else:
        log_rag_miss(query, results, route="search", reason="empty")
        return {
            "answer": f"未找到关于'{query}'的相关医学知识，请尝试更具体的查询。",
            "total_found": 0,
            "query": query,
            "not_found": True,
        }


def format_results(results: list) -> str:
    """格式化知识库检索结果。

    Args:
        results: 检索结果列表。

    Returns:
        格式化的字符串。
    """
    if not results:
        return "未找到相关信息。"

    output = []
    for i, doc in enumerate(results, 1):
        head = f"【结果 {i} · 来源：{doc.get('source', '医学知识库')}"
        section = doc.get("section") or ""
        if section:
            head += f" · {section}"
        output.append(head + "】")
        output.append(doc.get("content", "无内容"))

        # 显示相关度分数（如果有）
        score = doc.get("score", 0)
        if score > 0:
            output.append(f"相关度: {score:.2%}")

        output.append("")  # 空行分隔

    return "\n".join(output)
