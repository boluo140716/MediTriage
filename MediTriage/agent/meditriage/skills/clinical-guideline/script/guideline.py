"""临床指南检索 Skill。

自包含实现。
"""
from typing import Dict, Any
from loguru import logger

# 全局知识库实例
_kb_instance = None


def get_knowledge_base():
    global _kb_instance
    if _kb_instance is None:
        from meditriage.knowledge.milvus_kb import MedicalKnowledgeBase
        _kb_instance = MedicalKnowledgeBase()
    return _kb_instance


async def clinical_guideline(
    query: str, max_results: int = 1
) -> Dict[str, Any]:
    """检索临床指南。

    Args:
        query: 查询内容（疾病名称或治疗主题）。
        max_results: 最大结果数（默认 1，仅返回最相关的指南）。

    Returns:
        {
            "answer": "格式化的临床指南信息",
            "guideline_title": "指南标题",
            "organization": "发布机构"
        }
    """
    logger.info(
        f"Searching clinical guidelines for: {query} "
        f"(max_results={max_results})"
    )

    import asyncio

    # 重型同步检索放线程池，避免阻塞事件循环（SSE 全局冻结）。
    # Milvus 故障时优雅降级——不把原始连接异常串透传给 LLM。
    try:
        kb = await asyncio.to_thread(get_knowledge_base)
        results = await asyncio.to_thread(
            kb.search,
            query=f"{query} 临床指南 诊疗规范",
            top_k=max_results,
            filter_type="clinical_guideline",
        )
    except Exception as e:
        logger.warning(f"指南库不可用，降级回答: {e}")
        from meditriage.knowledge.badcase import log_rag_miss
        log_rag_miss(query, None, route="guideline", reason="unavailable")
        return {
            "answer": (
                "【知识库暂不可用】本次无法检索临床指南。请基于通用医学"
                "常识谨慎作答，并明确告知用户'此处未检索到对应临床指南'；"
                "严禁编造具名指南、机构或年份。"
            ),
            "guideline_title": "",
            "organization": "",
            "source": "知识库不可用",
            "not_found": True,
        }

    # reranker 概率分（相关≈0.99 / 无关≈0.002），低于阈值视为未命中。
    # 阈值统一定义在 knowledge 层单一常量（guideline 面窄宜严，env 可调）
    from meditriage.knowledge.langchain_rag import RAG_MIN_SCORE_GUIDELINE
    if results and results[0]["score"] > RAG_MIN_SCORE_GUIDELINE:
        from meditriage.knowledge.citations import build_citations, source_label

        doc = results[0]
        metadata = doc["metadata"]

        return {
            "answer": format_guideline(doc["content"], metadata),
            "guideline_title": source_label(metadata),
            "organization": metadata.get("source", "N/A"),
            "year": metadata.get("year", "N/A"),
            "source": "向量数据库",
            "citations": build_citations([doc], query=query),
        }
    else:
        # 未找到相关内容：明确告知"语料未收录"，引导模型转通用知识作答而非反复重试
        # 工程取舍：语料缺失时硬查会诱发"自信幻觉"，应坦诚弃权并接地，
        # 见 arXiv 2401.14887 / 2508.02296
        logger.warning(f"No clinical guidelines found in vector DB for {query}")
        from meditriage.knowledge.badcase import log_rag_miss
        log_rag_miss(
            query, results, route="guideline",
            reason="low_relevance" if results else "empty",
        )
        return {
            "answer": (
                f"【知识库未收录】本地指南库（聚焦心血管/内分泌/肾脏/呼吸等专科的 13 部权威指南）"
                f"中未检索到与'{query}'相关的内容。\n"
                f"请据此处理：① 不要再以相近措辞重复检索本工具；"
                f"② 可基于通用医学常识谨慎作答，但必须注明'此处未检索到对应临床指南'；"
                f"③ 严禁编造具名指南、机构或年份（没检索到就不得写出具体来源）。"
            ),
            "guideline_title": "",
            "organization": "",
            "source": "知识库未收录",
            "not_found": True,
        }


def format_guideline(content: str, metadata: Dict[str, Any]) -> str:
    """格式化临床指南信息（字段取索引里真实存在的 metadata）。"""
    from meditriage.knowledge.citations import source_label

    org = metadata.get("source") or "N/A"
    if org == "local_zh":
        org = "本地知识库"
    output = [
        "【临床诊疗指南】\n",
        f"指南名称：{source_label(metadata)}",
        f"发布机构：{org}",
        f"发布年份：{metadata.get('year') or 'N/A'}",
    ]
    section = metadata.get("section") or ""
    if section:
        output.append(f"所在章节：{section}")
    output.append(f"\n内容：\n{content}")

    return "\n".join(output)
