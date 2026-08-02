"""搜索相似历史案例 Skill。

基于本地长期记忆（Milvus agent_memory + BGE-M3）检索相似案例。
"""
import os
from typing import Dict, Any
from loguru import logger

# user_id：与 LongTermMemory/swarm_coordinator 一致；
# env MEDIX_MEMORY_USER 可覆盖（测试隔离）
DEFAULT_USER_ID = os.environ.get("MEDIX_MEMORY_USER", "medix_user")


async def search_similar_cases(
    query: str, max_results: int = 3
) -> Dict[str, Any]:
    """搜索相似的历史案例。

    Args:
        query: 查询内容。
        max_results: 最多返回结果数（默认 3）。

    Returns:
        {
            "answer": "格式化的相似案例",
            "total_found": 找到的案例数,
            "query": "原始查询"
        }
    """
    logger.info(
        f"Searching similar cases: query={query}, "
        f"max_results={max_results}"
    )

    try:
        # 本地长期记忆（Milvus agent_memory + BGE-M3）
        from meditriage.memory.medical_memory import MedicalMemory

        memory = MedicalMemory()

        if not memory.enabled:
            return {
                "answer": "长期记忆功能未启用。无法搜索历史案例。",
                "total_found": 0,
                "query": query
            }

        # 搜索相似会话（按 user_id 隔离）
        results = memory.search_memory(
            user_id=DEFAULT_USER_ID, query=query, top_k=max_results
        )

        if not results:
            return {
                "answer": f"未找到与'{query}'相关的历史案例。",
                "total_found": 0,
                "query": query
            }

        # 格式化输出
        formatted_cases = format_cases(results)

        return {
            "answer": formatted_cases,
            "total_found": len(results),
            "query": query
        }

    except Exception as e:
        logger.error(f"Failed to search similar cases: {e}")
        return {
            "answer": f"抱歉，搜索历史案例时出错：{str(e)}",
            "total_found": 0,
            "query": query
        }


def format_cases(results: list) -> str:
    """格式化相似案例。

    Args:
        results: 检索结果列表。

    Returns:
        格式化的字符串。
    """
    if not results:
        return "未找到相似案例。"

    output = ["【相似历史案例】\n"]

    for i, case in enumerate(results, 1):
        content = case.get("content", "")
        score = case.get("score", 0.0)
        metadata = case.get("metadata", {})

        output.append(f"【案例 {i}】（相似度: {score:.2%}）")
        output.append(
            content[:300] + "..." if len(content) > 300 else content
        )

        # 显示时间戳（如果有）
        timestamp = metadata.get("timestamp", "")
        if timestamp:
            output.append(f"时间: {timestamp}")

        output.append("")  # 空行分隔

    return "\n".join(output)
