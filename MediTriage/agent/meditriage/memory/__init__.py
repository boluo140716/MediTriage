"""
记忆系统：Agent 的持久化学习和记忆管理

包含：
- ShortTermMemory：会话级对话历史（内存/Redis）
- LongTermMemory：跨会话相似案例检索（Milvus + BGE-M3）
- MemoryHygiene：短期记忆卫生（精确去重 + 预算窗口）
"""

# 短期和长期记忆
from .short_term import (
    ShortTermMemory,
    ConversationHistory
)
from .long_term import (
    LongTermMemory
)

# 短期记忆卫生
from .hygiene import MemoryHygiene

# 本地 Markdown 持久化（会话总结）
from .session_summary import (
    SessionSummary,
    SessionSummaryManager,
    AgentParticipation,
    KeyFinding,
    PerformanceMetrics
)

__all__ = [
    # 短期和长期记忆
    'ShortTermMemory',
    'ConversationHistory',
    'LongTermMemory',
    # 短期记忆卫生
    'MemoryHygiene',
    # 本地持久化类（会话总结）
    'SessionSummary',
    'SessionSummaryManager',
    'AgentParticipation',
    'KeyFinding',
    'PerformanceMetrics',
]
