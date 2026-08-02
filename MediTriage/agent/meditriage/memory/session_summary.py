"""
SessionSummary：会话总结和经验提取

每次 Swarm 协作后自动生成会话总结，记录：
- 问题和背景
- 参与的 Agent
- 协作过程
- 关键发现
- 性能指标
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from loguru import logger


@dataclass
class AgentParticipation:
    """Agent 参与记录"""
    agent_id: str
    role: str  # lead/worker
    subtasks_handled: List[str]
    tool_calls: int
    execution_time: float  # 秒
    contribution_quality: float = 1.0  # 0-1


@dataclass
class KeyFinding:
    """关键发现"""
    category: str  # diagnosis/risk/evidence/treatment
    finding: str
    source_agent: str
    confidence: float = 1.0


@dataclass
class PerformanceMetrics:
    """性能指标

    注：parallel_efficiency / information_coverage / redundancy /
    speedup_vs_single 目前缺乏实测手段，默认 None（展示时标注"未实测"），仅在有实测数据时填充。
    """
    total_time: float  # 总耗时（秒）
    agent_count: int  # 参与 Agent 数量
    parallel_efficiency: Optional[float] = None  # 并行效率（0-1），未实测时为 None
    information_coverage: Optional[float] = None  # 信息覆盖度（0-1），未实测时为 None
    redundancy: Optional[float] = None  # 信息冗余度（0-1），未实测时为 None
    speedup_vs_single: Optional[float] = None  # 相比单 Agent 的加速比，未实测时为 None


@dataclass
class SessionSummary:
    """
    会话总结数据类

    记录一次完整的 Swarm 协作过程
    """
    session_id: str
    question: str
    context: Dict[str, Any]
    timestamp: datetime

    # 参与者
    agents_participated: List[AgentParticipation]

    # 过程
    subtasks_created: int
    subtasks_completed: int
    events_count: int

    # 结果
    final_answer: str
    key_findings: List[KeyFinding]

    # 性能
    performance: PerformanceMetrics

    # 元数据
    swarm_enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        date_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"# Session Summary: {self.session_id}",
            "",
            f"**时间**: {date_str}",
            "",
            "## 问题",
            self.question,
            ""
        ]

        if self.context:
            lines.extend([
                "## 背景",
                "```json",
                json.dumps(self.context, ensure_ascii=False, indent=2),
                "```",
                ""
            ])

        lines.extend([
            "## 参与 Agent",
            ""
        ])

        for agent in self.agents_participated:
            lines.append(f"### {agent.agent_id} ({agent.role})")
            lines.append(f"- 处理子任务：{len(agent.subtasks_handled)} 个")
            lines.append(f"- 工具调用：{agent.tool_calls} 次")
            lines.append(f"- 执行时间：{agent.execution_time:.2f} 秒")
            lines.append("")

        lines.extend([
            "## 协作过程",
            "",
            f"- 创建子任务：{self.subtasks_created} 个",
            f"- 完成子任务：{self.subtasks_completed} 个",
            f"- 发布事件：{self.events_count} 个",
            ""
        ])

        if self.key_findings:
            lines.extend([
                "## 关键发现",
                ""
            ])

            for finding in self.key_findings:
                lines.append(f"### {finding.category.upper()}")
                lines.append(f"**来源**: {finding.source_agent}")
                lines.append(f"**发现**: {finding.finding}")
                lines.append(f"**置信度**: {finding.confidence:.1%}")
                lines.append("")

        lines.extend([
            "## 最终答案",
            "",
            self.final_answer[:500]
            + ("..." if len(self.final_answer) > 500 else ""),
            ""
        ])

        def _pct(v: Optional[float]) -> str:
            return f"{v:.1%}" if v is not None else "未实测"

        def _speedup(v: Optional[float]) -> str:
            return f"{v:.2f}x" if v is not None else "未实测"

        lines.extend([
            "## 性能指标",
            "",
            f"- 总耗时：{self.performance.total_time:.2f} 秒",
            f"- 参与 Agent：{self.performance.agent_count} 个",
            f"- 并行效率：{_pct(self.performance.parallel_efficiency)}",
            f"- 信息覆盖度：{_pct(self.performance.information_coverage)}",
            f"- 信息冗余度：{_pct(self.performance.redundancy)}",
            f"- 加速比：{_speedup(self.performance.speedup_vs_single)}",
            ""
        ])

        return "\n".join(lines)

    @classmethod
    def from_shared_context(
        cls,
        session_id: str,
        question: str,
        shared_context: Any,
        final_answer: str,
        start_time: datetime,
        end_time: datetime
    ) -> "SessionSummary":
        """从 SharedContext 构建 SessionSummary"""

        # 计算性能指标
        total_time = (end_time - start_time).total_seconds()

        # 提取 Agent 参与信息
        agents_participated = []
        for agent_id, contributions in shared_context.agent_contributions.items():
            tool_calls = sum(
                1 for c in contributions
                if c.result.get('success', True)
            )
            agents_participated.append(AgentParticipation(
                agent_id=agent_id,
                role="worker",
                subtasks_handled=[c.subtask_id for c in contributions],
                tool_calls=tool_calls,
                execution_time=total_time / len(shared_context.agent_contributions)
            ))

        # 提取关键发现
        key_findings = []
        for contrib in shared_context.get_contributions():
            if "risk_level" in contrib.result:
                key_findings.append(KeyFinding(
                    category="risk",
                    finding=f"风险等级：{contrib.result['risk_level']}",
                    source_agent=contrib.agent_id,
                    confidence=contrib.confidence
                ))

        # 性能指标（并行效率/信息覆盖度/信息冗余度/加速比暂无实测手段，留空标注"未实测"）
        performance = PerformanceMetrics(
            total_time=total_time,
            agent_count=len(shared_context.agent_contributions),
        )

        return cls(
            session_id=session_id,
            question=question,
            context={},
            timestamp=start_time,
            agents_participated=agents_participated,
            subtasks_created=len(shared_context.task_decomposition),
            subtasks_completed=len(shared_context.get_all_completed_subtasks()),
            events_count=len(shared_context.events),
            final_answer=final_answer,
            key_findings=key_findings,
            performance=performance
        )


class SessionSummaryManager:
    """
    会话总结管理器

    负责保存和检索会话总结
    """

    def __init__(self, base_dir: str = None):
        # 默认锚到 paths.CACHE_DIR 下，不随进程 CWD 走——不同 CWD 起的进程会把同一份缓存散落到多处
        if base_dir is None:
            from meditriage.paths import CACHE_DIR
            base_dir = str(CACHE_DIR / "session_summaries")
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_summary_path(self, session_id: str) -> Path:
        """获取会话总结文件路径。

        session_id 来自请求侧（server 入口已白名单校验），这里再防御性
        过滤一层——它拼进落盘路径，含 / 或 .. 即可写出 base_dir 之外。
        """
        import re
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(session_id))[:64] or "x"
        # 按日期组织
        date_str = safe_id.split("-")[0] if "-" in safe_id else "unknown"
        date_dir = self.base_dir / date_str
        date_dir.mkdir(parents=True, exist_ok=True)
        return date_dir / f"{safe_id}.md"

    def save_summary(self, summary: SessionSummary):
        """保存会话总结"""
        summary_path = self._get_summary_path(summary.session_id)

        try:
            content = summary.to_markdown()
            summary_path.write_text(content, encoding="utf-8")
            logger.info(f"Saved session summary: {summary.session_id}")
        except Exception as e:
            logger.error(f"Error saving session summary: {e}")
