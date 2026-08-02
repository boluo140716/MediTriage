"""Swarm 编排的 LangGraph 实现（默认引擎）。

与旧版 SwarmCoordinator 等价的多 Agent 拓扑——记忆增强 → LeadAgent 动态
分解 → 多 Worker 并行执行 → LeadAgent 二次汇总 → 持久化——用 LangGraph
的 StateGraph 显式建模为图：

    START → enrich(记忆注入) → route(分解)
              ├─(≥2 子任务)→ [Send 扇出] → worker(并行) → synthesize ─┐
              └─(单任务/降级)→ single ────────────────────────────────┤
                                                                      ▼
                                                            persist → END

复用 LeadAgent / worker.process_subtask / SharedContext / 长短期记忆 /
出口护栏 _guard_final_answer 等组件，worker 节点跑完整的 RAG +
ReAct 循环；SSE 事件（session_started / lead_routing / swarm_* / 子任务与
工具事件）与 SwarmCoordinator 对齐，两个引擎对前端接口一致。并行靠 Send API 动态扇出
+ worker_results 的 operator.add reducer 扇入，单 worker 超时/异常折叠为
结构化结果，不拖垮整图。

接入：默认引擎（process_with_swarm 未设 SWARM_ENGINE 即走本路径）；
SWARM_ENGINE=legacy 回退旧版 SwarmCoordinator（保留对照与回退能力）。
graph.compile() 可挂 checkpointer 获得断点续跑，当前未启用。
"""
import asyncio
import operator
import os
import uuid
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from loguru import logger

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from meditriage.agents import (
    ConsultationAgent, DiagnosticAgent, ResearchAgent
)
from meditriage.agents.extractors import extract_suggestions
from meditriage.memory import (
    SessionSummaryManager, SessionSummary, ShortTermMemory, LongTermMemory
)
from meditriage.memory.long_term import DEFAULT_USER_ID
from meditriage.swarm.events import Event, EventType
from meditriage.swarm.lead_agent import LeadAgent
from meditriage.swarm.shared_context import SharedContext

# 单 worker 超时（秒）：按 worker 独立计时（legacy 协调器为所有 worker 整体 90s）；
# 单个慢任务只折叠自己，不拖垮同批其余 worker 的产出
_WORKER_TIMEOUT = float(os.environ.get("MEDITRIAGE_WORKER_TIMEOUT", "90.0"))

_DEFAULT_DISCLAIMER = (
    "以上信息仅供参考，不能替代专业医生的诊断和治疗。如有疑虑，请及时就医。"
)


class _SwarmState(TypedDict, total=False):
    """图的共享状态。worker_results 用 operator.add 做并行扇入的 reducer。"""
    question: str
    context: Optional[Dict[str, Any]]
    session_id: str
    user_id: str
    enable_swarm: bool
    emit: Any                      # None 安全的事件发射包装
    emit_raw: Any                  # 原始 emitter（可为 None），传给 AgentLoop
    lead: Any                      # 本请求的 LeadAgent
    agents: Dict[str, Any]         # 本请求的 worker 池（按 agent_id）
    start_time: Any
    enhanced_context: Dict[str, Any]
    assessment: Dict[str, Any]
    mode: str                      # single_agent / swarm / fallback / disabled_swarm
    sc: Any                        # SharedContext（仅 swarm 模式）
    subtasks: List[Any]            # SubTask 列表（仅 swarm 模式）
    worker_results: Annotated[List[Dict[str, Any]], operator.add]
    result: Dict[str, Any]


class LangGraphSwarm:
    """LangGraph 版多 Agent 编排（行为与 SwarmCoordinator 对齐）。

    图与记忆管理器常驻复用；worker/lead 每请求新建（与 SwarmCoordinator 一致，
    避免并发请求共享 Agent 实例的循环状态）。
    """

    def __init__(self):
        self.short_term_memory = ShortTermMemory(storage_type="memory")
        self.long_term_memory = LongTermMemory()
        self.session_manager = SessionSummaryManager()
        self.graph = self._build()
        logger.info("LangGraphSwarm initialized (default swarm engine)")

    # ---- 图构建 ----
    def _build(self):
        g = StateGraph(_SwarmState)
        g.add_node("enrich", self._enrich)
        g.add_node("route", self._route)
        g.add_node("single", self._single)
        g.add_node("worker", self._worker)
        g.add_node("synthesize", self._synthesize)
        g.add_node("persist", self._persist)
        g.add_edge(START, "enrich")
        g.add_edge("enrich", "route")
        g.add_conditional_edges("route", self._dispatch, ["single", "worker"])
        g.add_edge("worker", "synthesize")
        g.add_edge("synthesize", "persist")
        g.add_edge("single", "persist")
        g.add_edge("persist", END)
        return g.compile()

    # ---- 节点 ----
    async def _enrich(self, state: _SwarmState) -> Dict[str, Any]:
        """检索短/长期记忆构建增强上下文（对齐协调器 _enrich_context）。"""
        state["emit"]("session_started", {
            "session_id": state["session_id"],
            "question": state["question"],
            "timestamp": state["start_time"].isoformat(),
        })
        # 长期记忆检索含 GPU 嵌入 + Milvus IO（同步重型），放线程池防卡事件循环
        enhanced = await asyncio.to_thread(
            self._build_enhanced_context,
            state["question"], state.get("context"),
            state["session_id"], state["user_id"],
        )
        return {"enhanced_context": enhanced}

    def _build_enhanced_context(
        self, question, context, session_id, user_id
    ) -> Dict[str, Any]:
        """短期取最近 6 条（单条截断 600 字）；长期取 top 2（截断 400 字
        并按不可信文本消毒）。与协调器同一套裁剪参数。"""
        from meditriage.swarm.swarm_coordinator import _sanitize_memory_text
        recent_history = self.short_term_memory.get_recent_messages(
            session_id=session_id, limit=10
        )
        similar = self.long_term_memory.search_similar_sessions(
            query=question, limit=3,
            exclude_session=session_id, user_id=user_id,
        )
        enhanced = dict(context or {})
        if recent_history:
            enhanced["recent_history"] = [
                {
                    "role": m.get("role", ""),
                    "content": (m.get("content", "") or "")[:600],
                }
                for m in recent_history[-6:]
            ]
            logger.info(
                f"Loaded {len(recent_history)} recent messages "
                f"from short-term memory"
            )
        if similar:
            enhanced["historical_cases"] = [
                {
                    "summary": _sanitize_memory_text(
                        (m["content"] or "")[:400]
                    ),
                    "score": m["score"],
                }
                for m in similar[:2]
            ]
            logger.info(
                f"Found {len(similar)} similar historical cases "
                f"from long-term memory"
            )
        return enhanced

    async def _route(self, state: _SwarmState) -> Dict[str, Any]:
        """LeadAgent 动态分解（LLM 路由），决定单 Agent 或 Swarm。"""
        lead, emit = state["lead"], state["emit"]
        assessment = await lead.assess_and_decompose(
            state["question"], state["enhanced_context"]
        )
        subtasks_data = assessment.get("subtasks", []) or []
        use_swarm = (
            len(subtasks_data) >= 2 and state.get("enable_swarm", True)
        )
        logger.info(
            f"[langgraph] LeadAgent 分解任务：{len(subtasks_data)} 个 "
            f"-> {'swarm' if use_swarm else 'single'}"
        )
        emit("lead_routing", {
            "session_id": state["session_id"],
            "num_subtasks": len(subtasks_data),
            "route": "swarm" if use_swarm else "single_agent",
            "subtasks": [
                {
                    "type": t.get("type"),
                    "assigned_agent": t.get("assigned_agent"),
                    "description": t.get("description", "")[:200],
                }
                for t in subtasks_data
            ],
        })

        if not use_swarm:
            if len(subtasks_data) == 1:
                mode = "single_agent"
            elif len(subtasks_data) == 0:
                logger.warning(
                    "No subtasks generated, fallback to ConsultationAgent"
                )
                mode = "fallback"
            else:
                logger.info("Swarm disabled, fallback to ConsultationAgent")
                mode = "disabled_swarm"
            return {"assessment": assessment, "mode": mode}

        # Swarm 模式：建共享黑板，事件转发给前端，注册子任务
        sc = SharedContext(session_id=state["session_id"])

        def _forward(evt):
            emit(evt.type.value, evt.data)

        sc.subscribe(_forward)
        for w in state["agents"].values():
            w.attach_shared_context(sc)
        sc.publish_event(Event(
            type=EventType.SWARM_STARTED,
            source_agent="langgraph_swarm",
            data={
                "question": state["question"],
                "num_subtasks": len(subtasks_data),
            },
        ))
        subtasks = lead.create_subtasks(
            assessment, sc,
            original_question=state["question"],
            original_context=state["enhanced_context"],
        )
        return {
            "assessment": assessment, "mode": "swarm",
            "sc": sc, "subtasks": subtasks,
        }

    def _dispatch(self, state: _SwarmState):
        """单任务/降级走 single；多任务按子任务 Send 扇出并行 worker。"""
        if state["mode"] != "swarm":
            return "single"
        return [
            Send("worker", {
                "sc": state["sc"],
                "subtask": st,
                "agents": state["agents"],
                "emit_raw": state.get("emit_raw"),
            })
            for st in state["subtasks"]
        ]

    async def _worker(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """单个 worker：调用所分配 agent 的 process_subtask（完整 RAG + ReAct 循环）。

        超时与异常折叠为结构化结果，不让单个 worker 拖垮整个图。
        """
        sc, st = payload["sc"], payload["subtask"]
        entry: Dict[str, Any] = {
            "agent_id": st.assigned_agent, "subtask_id": st.id,
        }
        agent = payload["agents"].get(st.assigned_agent)
        if agent is None:
            logger.warning(
                f"[langgraph] unknown agent_id: {st.assigned_agent}, "
                f"skip subtask"
            )
            entry["error"] = f"unknown agent {st.assigned_agent}"
            return {"worker_results": [entry]}

        sc.start_subtask(st.id)
        try:
            result = await asyncio.wait_for(
                agent.process_subtask(
                    st, event_emitter=payload.get("emit_raw")
                ),
                timeout=_WORKER_TIMEOUT,
            )
            sc.complete_subtask(
                st.id, agent.agent_id,
                result if isinstance(result, dict)
                else {"answer": str(result)},
            )
            logger.info(f"{agent.agent_id}: Completed {st.type}")
        except asyncio.TimeoutError:
            logger.warning(
                f"[langgraph] worker {agent.agent_id} timeout "
                f"({_WORKER_TIMEOUT:.0f}s): {st.type}"
            )
            entry["timeout"] = True
        except Exception as e:
            logger.error(f"[langgraph] worker {agent.agent_id} failed: {e}")
            entry["error"] = str(e)
        return {"worker_results": [entry]}

    async def _synthesize(self, state: _SwarmState) -> Dict[str, Any]:
        """LeadAgent 二次汇总 + 出口护栏 + 会话摘要 + 短期记忆落盘。"""
        from meditriage.swarm.swarm_coordinator import _guard_final_answer
        sc, lead = state["sc"], state["lead"]
        question, session_id = state["question"], state["session_id"]
        timeout_occurred = any(
            r.get("timeout") for r in state.get("worker_results", [])
        )
        if timeout_occurred:
            logger.warning("[langgraph] 部分 worker 超时，汇总已完成部分")

        final_answer = await lead.synthesize_results(
            question=question,
            shared_context=sc,
            timeout_occurred=timeout_occurred,
        )
        # 汇总文本是新生成的，worker 层护栏管不到它：出口再过一遍
        # validator + auto_fixer（高危未就医 / 越界表述在此兜底补救）
        final_answer = _guard_final_answer(final_answer)
        end_time = datetime.now()

        # 会话摘要缓存（与协调器同一套 SessionSummary）
        try:
            summary = SessionSummary.from_shared_context(
                session_id=session_id,
                question=question,
                shared_context=sc,
                final_answer=final_answer,
                start_time=state["start_time"],
                end_time=end_time,
            )
            self.session_manager.save_summary(summary)
        except Exception as e:
            logger.error(f"Failed to generate session summary: {e}")

        # Swarm 路由下 worker 子任务循环不带 session_id，整轮对话不会经
        # Agent Loop 落短期记忆——必须在此显式保存，否则追问全部失忆
        if self.short_term_memory and session_id:
            self.short_term_memory.add_message(
                session_id=session_id, role="user", content=question
            )
            if final_answer:
                self.short_term_memory.add_message(
                    session_id=session_id, role="assistant",
                    content=final_answer,
                )

        sc.publish_event(Event(
            type=EventType.SWARM_COMPLETED,
            source_agent="langgraph_swarm",
            data={
                "duration": (
                    end_time - state["start_time"]
                ).total_seconds(),
                "agents_count": len(sc.agent_contributions),
            },
        ))

        completed_agents = list(sc.agent_contributions.keys())
        result = {
            "answer": final_answer,
            "swarm_enabled": True,
            "engine": "langgraph",
            "session_id": session_id,
            "agents_involved": completed_agents,
            "subtasks_completed": len(sc.get_all_completed_subtasks()),
            "total_time": (
                end_time - state["start_time"]
            ).total_seconds(),
            "swarm_metadata": sc.get_summary(),
            "timeout_occurred": timeout_occurred,
            "suggestions": (
                extract_suggestions(final_answer)
                or ["请遵循医嘱，注意休息和营养"]
            ),
        }
        if timeout_occurred and not completed_agents:
            result["disclaimer"] = (
                "由于系统超时，未能提供完整分析。建议简化问题重试，"
                "或在紧急情况下立即就医。"
            )
        elif timeout_occurred:
            result["disclaimer"] = (
                f"以上分析基于 {len(completed_agents)} 个 Agent 的部分协作"
                f"结果（部分分析模块超时未完成），仅供参考，不能替代医生诊断。"
            )
        else:
            result["disclaimer"] = (
                "以上分析基于多个专业 Agent 的协作，仅供参考，"
                "不能替代医生诊断。"
            )
        return {"result": result}

    async def _single(self, state: _SwarmState) -> Dict[str, Any]:
        """单任务直接调用对应 Agent（对齐协调器单 Agent 路由）。"""
        mode = state["mode"]
        agents = state["agents"]
        agent, agent_id = agents["consultation_agent"], "consultation_agent"
        if mode == "single_agent":
            task = (state["assessment"].get("subtasks") or [{}])[0]
            cand = task.get("assigned_agent")
            if cand in agents:
                agent, agent_id = agents[cand], cand
            else:
                logger.warning(
                    f"Unknown agent_id: {cand}, fallback to ConsultationAgent"
                )
        logger.info(f"Route: Single Agent ({agent_id}) [mode={mode}]")

        result = await agent.process({
            "question": state["question"],
            "context": state["enhanced_context"],
            "session_id": state["session_id"],
            # 注入 emitter 让 AgentLoop 发出细粒度事件
            "_event_emitter": state.get("emit_raw"),
        })
        result.update({
            "swarm_enabled": False,
            "engine": "langgraph",
            "session_id": state["session_id"],
            "agents_involved": [agent_id],
        })
        if mode == "single_agent":
            result["route_reason"] = f"单任务路由到 {agent_id}"
        result.setdefault("disclaimer", _DEFAULT_DISCLAIMER)
        result.setdefault("suggestions", [])
        return {"result": result}

    async def _persist(self, state: _SwarmState) -> Dict[str, Any]:
        """长期记忆落库 + 完成事件（单/多路径共用出口）。"""
        from meditriage.swarm.swarm_coordinator import _persist_long_term
        result = state["result"]
        mode = state["mode"]
        result.setdefault(
            "total_time",
            (datetime.now() - state["start_time"]).total_seconds(),
        )
        # 失败/空答案不写（_persist_long_term 内判定），避免污染 agent_memory
        await asyncio.to_thread(
            _persist_long_term,
            state["question"], result.get("answer", ""),
            state["session_id"],
            "swarm" if mode == "swarm" else f"mode={mode}",
            None if mode == "swarm" else result,
            state["user_id"],
        )
        state["emit"]("session_completed", {
            "session_id": state["session_id"],
            "mode": mode,
            "answer_preview": (result.get("answer") or "")[:500],
        })
        return {"result": result}

    # ---- 入口 ----
    async def run(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        event_emitter: Optional[Any] = None,
        user_id: Optional[str] = None,
        enable_swarm: bool = True,
    ) -> Dict[str, Any]:
        start_time = datetime.now()
        sid = session_id or (
            f"{start_time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        )
        uid = user_id or DEFAULT_USER_ID

        def _safe_emit(etype, data):
            if event_emitter:
                try:
                    event_emitter(etype, data)
                except Exception:
                    pass

        # worker/lead 每请求新建（与 SwarmCoordinator 一致），短期记忆注入各 worker 循环
        agents = {
            "consultation_agent": ConsultationAgent(),
            "diagnostic_agent": DiagnosticAgent(),
            "research_agent": ResearchAgent(),
        }
        for w in agents.values():
            if hasattr(w, "loop"):
                w.loop.short_term_memory = self.short_term_memory

        logger.info(
            f"[langgraph] Processing question (session={sid}): "
            f"{question[:50]}..."
        )
        out = await self.graph.ainvoke({
            "question": question,
            "context": context,
            "session_id": sid,
            "user_id": uid,
            "enable_swarm": enable_swarm,
            "emit": _safe_emit,
            "emit_raw": event_emitter,
            "lead": LeadAgent(),
            "agents": agents,
            "start_time": start_time,
            "worker_results": [],
            "result": {},
        })
        result = out.get("result") or {}
        result.setdefault("answer", "")
        result.setdefault("session_id", sid)
        return result


_INSTANCE: Optional[LangGraphSwarm] = None


def _get_swarm() -> LangGraphSwarm:
    """模块级单例（图与记忆管理器常驻；worker 每请求新建见 run）。"""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = LangGraphSwarm()
    return _INSTANCE


def swarm_engine() -> str:
    """当前 swarm 引擎：默认 langgraph；SWARM_ENGINE=legacy 回退旧版协调器。"""
    return os.environ.get("SWARM_ENGINE", "langgraph").strip().lower()


def langgraph_enabled() -> bool:
    return swarm_engine() != "legacy"


async def run_langgraph_swarm(
    question: str,
    context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    event_emitter: Optional[Any] = None,
    user_id: Optional[str] = None,
    enable_swarm: bool = True,
) -> Dict[str, Any]:
    """便捷入口：用 LangGraph 图跑一次多 Agent 编排。"""
    return await _get_swarm().run(
        question, context=context, session_id=session_id,
        event_emitter=event_emitter, user_id=user_id,
        enable_swarm=enable_swarm,
    )
