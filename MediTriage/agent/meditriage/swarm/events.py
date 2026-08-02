"""事件系统：Agent 之间的异步通信机制。"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any
import uuid


class EventType(Enum):
    """事件类型枚举（仅 swarm 粗粒度生命周期；前端细粒度可视化事件走字符串）。"""
    TASK_DECOMPOSED = "task_decomposed"  # LeadAgent 分解了任务
    SUBTASK_STARTED = "subtask_started"  # Agent 开始执行子任务
    SUBTASK_COMPLETED = "subtask_completed"  # Agent 完成子任务
    SWARM_STARTED = "swarm_started"  # Swarm 开始处理
    SWARM_COMPLETED = "swarm_completed"  # Swarm 完成处理


@dataclass
class Event:
    """事件数据类。

    Agent 通过发布事件到 SharedContext 来通信，而非直接调用其他 Agent。
    """
    type: EventType
    source_agent: str
    data: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
