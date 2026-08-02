"""运行时护栏：约束校验 + 违规自动修复。

ConstraintValidator 检测违规、AutoFixer 修复，二者是同一条"检测→修复"
流水线，由 core.agent_loop 背靠背调用。
"""
from .validator import ConstraintValidator
from .auto_fixer import AutoFixer

__all__ = ['ConstraintValidator', 'AutoFixer']
