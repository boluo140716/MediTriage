"""短期记忆卫生：精确去重 + 预算窗口。

防多轮对话历史膨胀的两件实事：按 (role, content) 哈希去掉完全重复的消息；按字符
预算只保留最近若干条（更早的整条丢弃，不做摘要——摘要式压缩在本项目尺度下收益低
且易降质）。
"""
from typing import Dict, List
import hashlib

from loguru import logger


class MemoryHygiene:
    """短期记忆卫生：精确去重 + 预算窗口。"""

    @staticmethod
    def deduplicate(messages: List[Dict]) -> List[Dict]:
        """按 (role, content) 哈希去掉完全重复的消息，保留首次出现。"""
        if not messages:
            return []
        seen = set()
        unique = []
        for msg in messages:
            key = hashlib.md5(
                f"{msg.get('role', '')}:{msg.get('content', '')}".encode()
            ).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(msg)
        removed = len(messages) - len(unique)
        if removed:
            logger.info(f"去重 {removed} 条重复消息")
        return unique

    @staticmethod
    def budget_window(
        messages: List[Dict],
        max_chars: int = 6000,
        per_msg_cap: int = 2000,
    ) -> List[Dict]:
        """按字符预算保留最近消息，防多轮上下文膨胀溢出。

        单条超 per_msg_cap 截断；从最近往前累计超 max_chars 则丢弃更早的。
        """
        out, total = [], 0
        for m in reversed(messages or []):
            c = (m.get("content") or "")
            if len(c) > per_msg_cap:
                c = c[:per_msg_cap] + " …[历史已截断]"
            if total + len(c) > max_chars and out:
                break
            out.append({**m, "content": c})
            total += len(c)
        return list(reversed(out))
