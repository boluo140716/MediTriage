"""Agent 执行状态：AgentState 状态机（PENDING→IN_PROGRESS→COMPLETED/FAILED）。"""
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentState:
    """Agent状态"""
    task_id: str
    agent_id: str
    status: TaskStatus = TaskStatus.PENDING
    iteration: int = 0
    max_iterations: int = 5
    input_data: Dict[str, Any] = field(default_factory=dict)
    final_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def is_completed(self) -> bool:
        """检查任务是否完成"""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]

    def should_continue(self) -> bool:
        """检查是否应该继续迭代"""
        return (
            self.status == TaskStatus.IN_PROGRESS
            and self.iteration < self.max_iterations
            and not self.is_completed()
        )

    def mark_completed(self, result: Dict[str, Any]):
        """标记为完成"""
        self.status = TaskStatus.COMPLETED
        self.final_result = result
        self.updated_at = datetime.now()

    def mark_failed(self, error: str):
        """标记为失败"""
        self.status = TaskStatus.FAILED
        self.error = error
        self.updated_at = datetime.now()
