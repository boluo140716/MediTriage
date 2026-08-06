"""Agent 循环引擎（ReAct：思考 → 行动 → 观察）。

控制流用 LangGraph StateGraph 显式建模为图：

    START → reason ──(tool_calls)──→ act ──(迭代有余)──→ reason
              │(文本答复)                │(迭代耗尽)
              ▼                         ▼
           finalize → END           wrapup(强制总结) → END

reason/act/finalize/wrapup 分别对应：LLM 决策、Skill 执行回灌、答复收尾
（校验+修复+落记忆）、迭代耗尽或上下文溢出后的强制总结。节点内部复用处理单元
（_handle_tool_calls / _finalize_answer / _force_summary /
_shrink_messages_for_overflow），整体行为：
- 上下文超限：截短中间观察 → wrapup（确定性失败，不重试）
- 其他异常：迭代有余额则回到 reason 重试，否则标记失败
- 支持短期记忆集成与约束验证
"""
import asyncio
import os
import uuid
import json
import re
from typing import Dict, Any, List, Optional, TypedDict
from loguru import logger

from langgraph.graph import StateGraph, START, END

from .state_manager import AgentState, TaskStatus
from .llm_client import LLMResponse

# 上下文预算预警线：prompt 占用超过窗口 80% 时告警留痕。
# 窗口大小与 vLLM --max-model-len 一致，换部署经 env 覆盖。
_CTX_WARN_TOKENS = int(
    int(os.environ.get("MEDITRIAGE_MAX_MODEL_LEN", "262144")) * 0.8
)

# 约束验证和自动修复
try:
    from meditriage.guardrails import ConstraintValidator, AutoFixer
    CONSTRAINTS_ENABLED = True
except ImportError:
    logger.warning(
        "Guardrails module not found, running without constraint validation"
    )
    CONSTRAINTS_ENABLED = False


_TOOL_NARRATION_RE = re.compile(
    r'^\s*(调用工具|我将调用|我会调用|正在调用|准备调用|现在调用)[:：][^\n]*\n?',
    re.MULTILINE,
)


def _strip_tool_narration(text):
    """去除模型把"调用工具：X"之类内部动作当文本输出的前导噪声。

    若整条被清空则保留原文。
    """
    if not text:
        return text
    cleaned = _TOOL_NARRATION_RE.sub('', text).lstrip('\n　 ')
    return cleaned if cleaned.strip() else text


class _LoopState(TypedDict, total=False):
    """图状态：单次 run 的全部运行时数据。

    messages 列表在节点内就地变更（与处理单元的既有签名一致），
    channel 始终持同一引用；agent_state 是 AgentState 状态机
    （迭代计数 / 完成标记），路由依据它判定走向。
    """
    agent: Any
    agent_state: Any
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]]
    session_id: Optional[str]
    emit: Any                   # callable(etype, data)，已绑定 agent/task 元数据
    llm_response: Any           # 本轮 LLM 响应（LLMResponse）
    overflow: bool              # 上下文超限（确定性失败 → 直接收尾）
    errored: bool               # 本轮异常（迭代有余额则重试）


class AgentLoop:
    """Agent 循环引擎。

    LLM 自主决策 Skill 调用，循环直到任务完成。

    功能：
    - 支持短期记忆（ShortTermMemory）
    - 自动记录每轮的 user/assistant 消息
    """

    def __init__(
        self,
        max_iterations: int = 10,
        short_term_memory: Optional[Any] = None,
        max_tool_calls: int = 4,
    ):
        """初始化 Agent 循环引擎。

        Args:
            max_iterations: 最大迭代次数（防止无限循环）。
            short_term_memory: 短期记忆管理器（可选）。
            max_tool_calls: 最大 Skill 调用次数（硬性限制，默认 4 次；
                可经 agent config 覆盖）。
        """
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.short_term_memory = short_term_memory
        # 工具并行开关（env 可关）：同一轮多个 Skill 并行执行 + 保序回灌。
        # 质量保障：结果按原始调用顺序回灌，模型看到的内容/顺序与串行一致。
        self._tool_parallel = os.environ.get(
            "MEDITRIAGE_TOOL_PARALLEL", "1"
        ).strip().lower() not in ("0", "false", "no", "")
        try:
            self._tool_max_concurrent = max(
                1, int(os.environ.get("MEDITRIAGE_TOOL_MAX_CONCURRENT", "3"))
            )
        except (TypeError, ValueError):
            self._tool_max_concurrent = 3

        # 约束验证器和自动修复器
        self.validator = ConstraintValidator() if CONSTRAINTS_ENABLED else None
        self.auto_fixer = AutoFixer() if CONSTRAINTS_ENABLED else None
        if CONSTRAINTS_ENABLED:
            logger.debug("Constraint validation enabled")

        # 控制流图：节点复用下方处理单元；图本身无跨请求状态，可复用
        self._graph = self._build_graph()

    async def run(
        self,
        agent,
        input_data: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """执行 Agent 循环（LangGraph 图驱动）。

        编排：初始化消息 → 图执行（reason ⇄ act 交替，finalize / wrapup
        收尾）。逐轮逻辑见 _node_* 节点与 _handle_tool_calls /
        _finalize_answer / _force_summary。

        Args:
            agent: Agent 实例。
            input_data: 输入数据（可包含 '_event_emitter' callable，
                用于流式可视化）。

        Returns:
            最终结果。
        """
        task_id = str(uuid.uuid4())
        state = AgentState(
            task_id=task_id,
            agent_id=agent.agent_id,
            input_data=input_data,
            max_iterations=self.max_iterations,
        )

        # 每次 run 的私有计数挂在本次 AgentState 上，而非实例属性——
        # 同一 Agent 的多个子任务并发进入同一 loop 时互不串扰
        # （工具预算 / 去重守卫各算各的）
        self._ensure_run_counters(state)

        # 取出可选的事件发射器（用于前端流式可视化）
        # emitter 签名：emit(event_type_str, data_dict)
        # 优先级：input_data 显式传入 > self._pending_emitter（Swarm 模式注入）
        event_emitter = None
        if isinstance(input_data, dict):
            event_emitter = input_data.get('_event_emitter')
        if event_emitter is None:
            event_emitter = getattr(self, '_pending_emitter', None)

        def _emit(etype, data):
            if event_emitter:
                try:
                    event_emitter(
                        etype,
                        {
                            **data,
                            "agent_id": agent.agent_id,
                            "task_id": task_id,
                        },
                    )
                except Exception:
                    pass

        logger.info(
            f"Starting Agent Loop for {agent.agent_id}, task_id={task_id}"
        )

        try:
            state.status = TaskStatus.IN_PROGRESS

            # 初始化消息历史（包含历史对话）
            messages = self._initialize_messages(agent, input_data, session_id)

            # 记录用户消息到短期记忆——存原始问题，不存 format_user_input
            # 的格式化产物（后者带背景信息/历史渲染，会在下一轮被再次注入，
            # 历史套历史地递归膨胀，且把他话题记忆以用户口吻回放）
            if self.short_term_memory and session_id:
                raw_q = None
                if isinstance(input_data, dict):
                    raw_q = (
                        input_data.get('question')
                        or input_data.get('query')
                    )
                user_message = raw_q or (
                    messages[-1]["content"] if messages else str(input_data)
                )
                self.short_term_memory.add_message(
                    session_id=session_id,
                    role="user",
                    content=user_message
                )
                logger.debug(
                    f"Recorded user message to short-term memory "
                    f"(session={session_id})"
                )

            # 获取 Agent 的 Skills (OpenAI format)
            tools_openai_format = agent.get_tools_for_llm()

            logger.debug(
                f"Agent has "
                f"{len(tools_openai_format) if tools_openai_format else 0} "
                f"skills available"
            )

            # 图执行主体（recursion_limit 给足
            # reason/act 交替 + 入口与收尾节点的余量）
            await self._graph.ainvoke(
                {
                    "agent": agent,
                    "agent_state": state,
                    "messages": messages,
                    "tools": tools_openai_format,
                    "session_id": session_id,
                    "emit": _emit,
                    "llm_response": None,
                    "overflow": False,
                    "errored": False,
                },
                config={"recursion_limit": self.max_iterations * 4 + 10},
            )

            # LLM 连续失败到 FAILED：给出明确错误答复，
            # 不让空 {} 静默流向前端（用户会看到"完成"但无任何内容）
            if state.status == TaskStatus.FAILED and not state.final_result:
                state.final_result = {
                    'answer': (
                        "抱歉，本次处理遇到内部错误，未能生成有效回答。"
                        "请稍后重试或换一种问法；如症状紧急，"
                        "请直接就医或拨打 120。"
                    ),
                    'iterations': state.iteration,
                    'agent_id': agent.agent_id,
                    'error': True,
                }

            logger.info(
                f"Agent Loop finished: status={state.status.value}, "
                f"iterations={state.iteration}"
            )
            return state.final_result or {}

        except Exception as e:
            logger.error(f"Agent Loop failed: {e}")
            state.mark_failed(str(e))
            raise

    # ---- 控制流图：节点与路由 ----

    def _build_graph(self):
        """把 ReAct 循环建为 StateGraph。"""
        g = StateGraph(_LoopState)
        g.add_node("reason", self._node_reason)
        g.add_node("act", self._node_act)
        g.add_node("finalize", self._node_finalize)
        g.add_node("wrapup", self._node_wrapup)
        g.add_conditional_edges(START, self._route_entry, ["reason", "wrapup"])
        g.add_conditional_edges(
            "reason", self._route_after_reason,
            ["reason", "act", "finalize", "wrapup"],
        )
        g.add_conditional_edges(
            "act", self._route_after_act, ["reason", "wrapup"]
        )
        g.add_conditional_edges(
            "finalize", self._route_after_finalize, ["reason", "wrapup", END]
        )
        g.add_edge("wrapup", END)
        return g.compile()

    async def _node_reason(self, s: _LoopState) -> Dict[str, Any]:
        """一轮 LLM 决策：返回 tool_calls（→act）或文本答复（→finalize）。"""
        state, agent, emit = s["agent_state"], s["agent"], s["emit"]
        state.iteration += 1
        logger.debug(
            f"=== Iteration {state.iteration}/{state.max_iterations} ==="
        )

        emit("agent_thinking", {
            "iteration": state.iteration,
            "max_iterations": state.max_iterations,
            "tools_available": [
                t["function"]["name"]
                for t in (s.get("tools") or [])
                if "function" in t
            ],
        })

        try:
            llm_response: LLMResponse = (
                await agent.llm_client.chat_with_tools(
                    messages=s["messages"],
                    tools=s.get("tools"),
                    tool_choice="auto",
                    temperature=agent.config.get('temperature', 0.7)
                )
            )
        except Exception as e:
            return self._on_node_error(agent, state, s["messages"], e)

        emit("llm_response", {
            "iteration": state.iteration,
            "finish_reason": llm_response.finish_reason,
            "has_tool_calls": llm_response.has_tool_calls(),
            "content_preview": (llm_response.content or "")[:300],
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in llm_response.tool_calls
            ],
            "prompt_tokens": llm_response.prompt_tokens,
            "total_tokens": llm_response.total_tokens,
        })

        # 主动预算预警：超过 80% 窗口先留痕，别等 vLLM 400 才被动兜底
        if (llm_response.prompt_tokens
                and llm_response.prompt_tokens > _CTX_WARN_TOKENS):
            logger.warning(
                f"上下文占用 {llm_response.prompt_tokens} "
                f"tokens，已超过预警线 {_CTX_WARN_TOKENS}（窗口的 80%）"
            )

        return {
            "llm_response": llm_response, "overflow": False, "errored": False,
        }

    async def _node_act(self, s: _LoopState) -> Dict[str, Any]:
        """执行本轮 Skill 调用并回灌观察结果。"""
        state, agent = s["agent_state"], s["agent"]
        try:
            await self._handle_tool_calls(
                agent, s["llm_response"], s["messages"],
                s.get("session_id"), state, s["emit"],
            )
        except Exception as e:
            return self._on_node_error(agent, state, s["messages"], e)
        return {"overflow": False, "errored": False}

    async def _node_finalize(self, s: _LoopState) -> Dict[str, Any]:
        """文本答复收尾：校验 + 自动修复 + 落记忆，标记完成。"""
        state, agent = s["agent_state"], s["agent"]
        try:
            await self._finalize_answer(
                agent, s["llm_response"], s["messages"],
                s.get("session_id"), state, s["emit"],
            )
        except Exception as e:
            return self._on_node_error(agent, state, s["messages"], e)
        return {"overflow": False, "errored": False}

    async def _node_wrapup(self, s: _LoopState) -> Dict[str, Any]:
        """迭代耗尽 / 溢出后的收尾：未完成则强制总结。"""
        state = s["agent_state"]
        if not state.is_completed():
            await self._force_summary(
                s["agent"], s["messages"], s.get("session_id"), state
            )
        return {}

    def _on_node_error(self, agent, state, messages, e) -> Dict[str, Any]:
        """节点异常的统一归类。

        上下文超限是确定性失败：同一 messages 重试必然再超，截短历史观察
        后直接转入强制总结；其他异常在迭代耗尽时标记失败，否则留给路由重试。
        """
        logger.error(f"Error in iteration {state.iteration}: {e}")
        if agent.llm_client.is_ctx_overflow(e):
            logger.warning("上下文超限：截短工具回灌后转入强制总结")
            self._shrink_messages_for_overflow(messages)
            return {"llm_response": None, "overflow": True, "errored": False}
        if state.iteration >= state.max_iterations:
            state.mark_failed(str(e))
        return {"llm_response": None, "overflow": False, "errored": True}

    def _route_entry(self, s: _LoopState) -> str:
        """入口守卫（等价于 while 的首次条件判断）。"""
        return "reason" if s["agent_state"].should_continue() else "wrapup"

    def _route_after_reason(self, s: _LoopState) -> str:
        state = s["agent_state"]
        if s.get("overflow"):
            return "wrapup"
        if s.get("errored"):
            return "reason" if state.should_continue() else "wrapup"
        if s["llm_response"].has_tool_calls():
            return "act"
        return "finalize"

    def _route_after_act(self, s: _LoopState) -> str:
        if s.get("overflow"):
            return "wrapup"
        return "reason" if s["agent_state"].should_continue() else "wrapup"

    def _route_after_finalize(self, s: _LoopState) -> str:
        if s.get("overflow"):
            return "wrapup"
        if s.get("errored"):
            return (
                "reason" if s["agent_state"].should_continue() else "wrapup"
            )
        return END

    @staticmethod
    def _shrink_messages_for_overflow(
        messages: List[Dict[str, Any]], keep_chars: int = 400
    ) -> None:
        """上下文超限后的自救：就地截短中间观察结果。

        只裁工具回灌与中间 assistant 文本，不动 system 提示和最后一条
        user 消息——裁完交给 _force_summary 用剩余信息收尾。
        """
        last_user_idx = max(
            (i for i, m in enumerate(messages)
             if m.get("role") == "user"),
            default=-1,
        )
        for i, m in enumerate(messages):
            if i == last_user_idx or m.get("role") == "system":
                continue
            content = m.get("content")
            if isinstance(content, str) and len(content) > keep_chars:
                m["content"] = content[:keep_chars] + "…（超长截断）"

    @staticmethod
    def _ensure_run_counters(state) -> None:
        """初始化挂在 state 上的本次 run 计数（工具预算 + 去重签名集）。

        计数随 state 走而非实例属性：同一 loop 被并发复用（同一 worker
        领多个子任务）时，各 run 的预算与去重互不串扰。
        """
        if not hasattr(state, 'tool_call_count'):
            state.tool_call_count = 0
        if not hasattr(state, 'tool_failure_count'):
            state.tool_failure_count = 0
        if not hasattr(state, 'called_signatures'):
            state.called_signatures = set()

    async def _handle_tool_calls(
        self, agent, llm_response, messages, session_id, state, emit
    ):
        """执行本轮 LLM 请求的 Skill 调用，将结果回灌到 messages。

        达到最大调用次数时改为要求模型直接作答；对相同 (工具, 参数) 去重。
        """
        self._ensure_run_counters(state)
        # 硬性限制：检查是否已达到最大调用次数
        if state.tool_call_count >= self.max_tool_calls:
            logger.warning(
                f"已达到最大 Skill 调用次数限制 "
                f"({self.max_tool_calls})，强制生成最终答案"
            )
            # 强制要求 LLM 提供最终答案
            messages.append({
                'role': 'user',
                'content': (
                    f'已完成 {self.max_tool_calls} 次信息检索。'
                    f'请基于已获取的信息提供最终答复。'
                )
            })
            return

        logger.info(
            f"LLM requested {len(llm_response.tool_calls)} "
            f"tool calls (当前已调用 "
            f"{state.tool_call_count}/{self.max_tool_calls})"
        )

        # 添加 assistant 消息（包含 tool_calls）
        messages.append(
            self._create_assistant_message_with_tools(llm_response)
        )

        # 记录 assistant 消息到短期记忆
        if self.short_term_memory and session_id:
            tool_names = [tc.name for tc in llm_response.tool_calls]
            self.short_term_memory.add_message(
                session_id=session_id,
                role="assistant",
                content=f"调用工具：{', '.join(tool_names)}"
            )

        # 执行每个 Skill 调用：并行执行 + 保序回灌。
        # 并发安全已排查：全部 Skill 只读（RAG 检索/规则/常量表），无写共享状态。
        # 质量保障：asyncio.gather 按输入顺序返回结果 -> 模型看到的工具结果
        # 内容与顺序和串行完全一致，最终输出质量不变。
        _any_failed = False
        _parallel = self._tool_parallel and len(llm_response.tool_calls) > 1

        # ---- 1) 预检：search_history 注入 + 去重 + 约束验证（主协程、执行前
        #          统一检查，消除并发下 called_signatures 竞态）----
        _prepared: List[tuple] = []
        for tool_call in llm_response.tool_calls:
            if tool_call.name == "search_history" and session_id:
                tool_call.arguments["session_id"] = session_id
            try:
                _sig = (
                    f"{tool_call.name}:"
                    f"{json.dumps(tool_call.arguments, sort_keys=True, ensure_ascii=False)}"
                )
            except Exception:
                _sig = f"{tool_call.name}:{tool_call.arguments}"
            _skip = _sig in state.called_signatures
            if not _skip:
                state.called_signatures.add(_sig)
                # 验证调用（无副作用，可预检）
                if self.validator:
                    validation_result = (
                        self.validator.validate_tool_call(
                            agent.agent_id, tool_call.name
                        )
                    )
                    if not validation_result.get("valid"):
                        logger.warning(
                            f"约束警告: "
                            f"{validation_result.get('reason')}"
                        )
            else:
                logger.warning(
                    f"跳过重复调用：{tool_call.name}"
                    f"（相同参数已执行过）"
                )
            _prepared.append((tool_call, _skip))

        # ---- 1.5) 先发所有 tool_call_started（执行前，前端可实时看到在调用哪些工具）
        for tool_call, _skip in _prepared:
            state.tool_call_count += 1
            logger.debug(
                f"Executing: "
                f"{tool_call.name}({tool_call.arguments}) - "
                f"第 {state.tool_call_count} 次调用"
            )
            emit("tool_call_started", {
                "tool_name": tool_call.name,
                "arguments": tool_call.arguments,
                "call_index": state.tool_call_count,
                "iteration": state.iteration,
            })

        # ---- 2) 执行（并行/串行），结果按输入顺序对齐（gather 保序）----
        _tool_results: Dict[int, Any] = {}
        _pending = [tc for tc, _skip in _prepared if not _skip]
        if _parallel:
            _sem = asyncio.Semaphore(self._tool_max_concurrent)

            async def _execute_one(_tc):
                async with _sem:
                    return await agent.execute_tool(
                        tool_name=_tc.name, arguments=_tc.arguments
                    )

            _futures = [
                asyncio.ensure_future(_execute_one(tc)) for tc in _pending
            ]
            for _tc, _fut in zip(_pending, _futures):
                try:
                    _tool_results[id(_tc)] = await _fut
                except Exception as e:
                    # 失败隔离：单个工具异常不拖垮其他并行工具
                    logger.error(f"tool {_tc.name} parallel exec error: {e}")
                    _tool_results[id(_tc)] = {
                        "success": False,
                        "answer": f"工具执行异常：{e}",
                    }
        else:
            for _tc in _pending:
                _tool_results[id(_tc)] = await agent.execute_tool(
                    tool_name=_tc.name, arguments=_tc.arguments
                )

        # ---- 3) 保序回灌：按原始调用顺序 emit completed / 回灌 messages /
        #          落短期记忆，与串行时模型看到的上下文完全一致 ----
        for tool_call, _skip in _prepared:
            if _skip:
                tool_result = {
                    "answer": (
                        "（已对相同请求检索过，不再重复执行。"
                        "请基于已获取的信息直接作答，"
                        "勿重复调用同一工具。）"
                    ),
                    "duplicate_skipped": True,
                }
            else:
                tool_result = _tool_results[id(tool_call)]
                if (isinstance(tool_result, dict)
                        and tool_result.get("success") is False):
                    _any_failed = True
                    state.tool_failure_count += 1

            # 预览给前端看：优先用 skill 格式化好的 answer 文本，而非 dict repr
            if isinstance(tool_result, dict):
                _preview = str(
                    tool_result.get("answer") or tool_result
                )[:400]
                _citations = tool_result.get("citations") or None
                # 未命中/重复跳过时 answer 是写给模型的内部指令，
                # 不构成可展示的检索证据——标记下发，前端据此不渲染进证据面板
                _no_evidence = bool(
                    tool_result.get("not_found")
                    or tool_result.get("duplicate_skipped")
                )
            else:
                _preview = str(tool_result)[:400]
                _citations = None
                _no_evidence = False

            emit("tool_call_completed", {
                "tool_name": tool_call.name,
                "call_index": state.tool_call_count,
                "iteration": state.iteration,
                "result_preview": _preview,
                **({"citations": _citations} if _citations else {}),
                **({"not_found": True} if _no_evidence else {}),
            })

            # 添加结果消息（citations 是给前端的展示数据，来源信息已在
            # answer 文本内，不再随工具消息回灌 LLM 占上下文）
            _llm_result = tool_result
            if isinstance(tool_result, dict) and "citations" in tool_result:
                _llm_result = {
                    k: v for k, v in tool_result.items() if k != "citations"
                }
            messages.append(
                agent.llm_client.create_tool_message(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=_llm_result
                )
            )

            # 记录结果到短期记忆
            if self.short_term_memory and session_id:
                result_summary = str(tool_result)[:200]
                self.short_term_memory.add_message(
                    session_id=session_id,
                    role="tool",
                    content=(
                        f"{tool_call.name}: "
                        f"{result_summary}"
                    )
                )

        # 失败重规划引导：裸 {"success": false} 回灌时模型常原样重试
        # 或带病作答，给一条明确的下一步指引
        if _any_failed:
            messages.append({
                'role': 'user',
                'content': (
                    '提示：上面有工具调用执行失败。请勿以相同参数重试该调用；'
                    '可更换关键词或改用其他工具再试一次，'
                    '或基于已获取的信息直接作答。'
                ),
            })

    async def _finalize_answer(
        self, agent, llm_response, messages, session_id, state, emit
    ):
        """LLM 给出文本答复：校验+自动修复+落记忆+后处理，标记任务完成。"""
        logger.info("LLM provided final response (no tool calls)")

        # 验证和修复输出
        final_answer = _strip_tool_narration(llm_response.content)

        if self.validator and final_answer:
            validation_result = self.validator.validate_output(
                agent.agent_id,
                final_answer
            )

            if not validation_result.get("valid"):
                logger.warning(
                    f"输出约束违规: "
                    f"{validation_result.get('violations')}"
                )

                # 自动修复
                if self.auto_fixer and validation_result.get("auto_fixable"):
                    fixed_answer = self.auto_fixer.fix_output(
                        final_answer,
                        validation_result.get("auto_fixable", [])
                    )
                    if fixed_answer != final_answer:
                        logger.info("输出已自动修复")
                        final_answer = fixed_answer

        # 记录最终回答到短期记忆
        if self.short_term_memory and session_id:
            self.short_term_memory.add_message(
                session_id=session_id,
                role="assistant",
                content=final_answer or "(empty response)"
            )
            logger.debug(
                f"Recorded final answer to short-term memory "
                f"(session={session_id})"
            )

        result = {
            'answer': final_answer,
            'iterations': state.iteration,
            'agent_id': agent.agent_id,
            'tool_failure_count': getattr(state, "tool_failure_count", 0),
        }

        # 让 Agent 进行结果后处理（如提取建议等）
        if hasattr(agent, 'post_process_result'):
            result = await agent.post_process_result(result, final_answer)

        emit("final_answer", {
            "iterations": state.iteration,
            "answer_preview": (final_answer or "")[:500],
            "tool_call_count": getattr(state, "tool_call_count", 0),
            "tool_failure_count": getattr(state, "tool_failure_count", 0),
        })

        state.mark_completed(result)

    async def _force_summary(self, agent, messages, session_id, state):
        """达到最大迭代仍未完成：禁用工具，强制 LLM 给出最终总结。"""
        logger.warning("Max iterations reached without completion")

        # 强制调用 LLM 生成最终总结
        try:
            logger.info("Forcing LLM to provide final answer")

            # 添加强制总结的提示
            messages.append({
                'role': 'user',
                'content': '请基于以上信息，提供最终的答复。'
            })

            # 调用 LLM（禁用 function calling）
            final_response = await agent.llm_client.chat_with_tools(
                messages=messages,
                tools=None,
                temperature=0.7
            )

            result = {
                'answer': (
                    _strip_tool_narration(final_response.content)
                    or '抱歉，未能完成任务'
                ),
                'iterations': state.iteration,
                'warning': 'max_iterations_reached'
            }

            # 记录最终回答到短期记忆
            if self.short_term_memory and session_id:
                self.short_term_memory.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=result['answer']
                )

            state.mark_completed(result)
            logger.info("Generated fallback answer after max iterations")

        except Exception as e:
            logger.error(f"Failed to generate fallback answer: {e}")
            # 降级到简单提取
            result = {
                'answer': '抱歉，系统在处理您的问题时遇到了问题。建议您简化问题或稍后重试。',
                'iterations': state.iteration,
                'warning': 'max_iterations_reached',
                'error': str(e)
            }
            state.mark_completed(result)

    def _initialize_messages(
        self,
        agent,
        input_data: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """初始化消息列表，包含历史对话上下文。"""
        messages = []

        # 系统提示词
        system_prompt = agent.get_system_prompt()
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })

        # 加载历史对话（短期记忆）
        if self.short_term_memory and session_id:
            # 取最近 5 轮对话
            # get_history 已做去重 + 预算窗口（见 memory.hygiene）
            history = self.short_term_memory.get_history(session_id, limit=5)
            if history:
                logger.info(
                    f"Loaded {len(history)} historical messages "
                    f"(budgeted) from short-term memory"
                )
                messages.extend(history)

        # 用户输入
        user_message = agent.format_user_input(input_data)
        messages.append({
            'role': 'user',
            'content': user_message
        })

        return messages

    def _create_assistant_message_with_tools(
        self, llm_response: LLMResponse
    ) -> Dict[str, Any]:
        """创建包含 tool_calls 的 assistant 消息。"""
        message = {
            'role': 'assistant',
            'content': llm_response.content or None
        }

        # 添加 tool_calls（OpenAI 格式）
        if llm_response.tool_calls:
            message['tool_calls'] = [
                {
                    'id': tc.id,
                    'type': 'function',
                    'function': {
                        'name': tc.name,
                        'arguments': json.dumps(
                            tc.arguments, ensure_ascii=False
                        )
                    }
                }
                for tc in llm_response.tool_calls
            ]

        return message
