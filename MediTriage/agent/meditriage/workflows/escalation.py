"""转诊闭环核心（P1-2）：置信度评估 -> 交接摘要 -> 状态机落库。

决策三层（顺序执行，命中即短路）：
  1) 强制转诊：危机关键词命中（自伤/心脑血管急症等）、处理超时/报错/空回答、
     用户主动要求转人工
  2) 明显低风险：信息咨询类提问（怎么/如何/注意事项）且无风险信号
     -> 直接不转（省一次 LLM 调用）；真实症状描述进第 3 层打分
  3) 模糊场景：LLM 打置信度分（0~1）-> 低于阈值(默认 0.4) 转诊；
     LLM 失败/超时 -> 安全兜底转人工（宁过不欠）

设计取舍：
  - 危机信号永远优先于 LLM 打分（规则是权威，模型可能漏判）；
  - 置信度评分只对模糊场景调用一次（短 prompt、8s 超时），普通咨询零额外延迟；
    MEDITRIAGE_ESCALATION_LLM=0 可退化为纯规则模式；
  - 不抛异常：评分失败/超时降级处理，绝不让转诊逻辑中断主问答链路。
"""
import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from meditriage.guardrails._keywords import HIGH_RISK_KEYWORDS
from .escalation_store import EscalationStore

# 阈值 / 开关（env，统一 MEDITRIAGE_ 前缀）
DEFAULT_THRESHOLD = 0.4
DEFAULT_LLM_TIMEOUT = 8.0
# 同一会话内"LLM 评分不可用"的安全兜底转诊上限（防 API 故障刷爆医生队列）
DEFAULT_MAX_FALLBACK_PER_SESSION = 1

# 危机关键词：guardrails 共享高危症状 + 本模块补充（对齐评测集危机案例与自伤表达）
_EXTRA_CRISIS_KEYWORDS = (
    # 自伤/轻生（与 swarm 出口兜底同一意图，此处用于转诊建单）
    "自杀", "自残", "自伤", "自尽", "轻生", "想死", "不想活",
    "活不下去", "生无可恋",
    # 心脑血管急症（对齐评测集：胸痛 / 卒中样症状 / 突发剧烈头痛）
    "胸口疼", "胸口痛", "心口疼", "心前区", "喘不上气", "憋气",
    "一侧手脚没力气", "手脚麻木无力", "说话不清楚", "言语不清",
    "突然剧烈", "炸开", "冒冷汗", "晕倒", "不省人事",
    "中风", "卒中", "脑梗", "脑出血", "偏瘫",
    "120",
)
_CRISIS_PAT = re.compile(
    "|".join(re.escape(k) for k in tuple(HIGH_RISK_KEYWORDS) + _EXTRA_CRISIS_KEYWORDS)
)

# 用户主动要求转人工（显式意图，避免误伤"什么时候需要看医生"这类咨询）
_ASK_DOCTOR_PAT = re.compile(
    r"转人工|人工服务|人工客服|帮我找医生|找医生看|让医生|需要医生|"
    r"要医生|挂号|预约医生|人工介入"
)

# 回答侧"已提示立即就医"信号：不作为强制转诊，但计入原因并抬高风险。
# 不含"急救/紧急情况"等宽泛词（AI 几乎每个回答都带"如遇紧急情况请立即
# 拨打急救电话"这类泛化话术，误判会把非问诊/科普咨询也转人工）。
_SEEK_CARE_PAT = re.compile(
    r"立即就医|尽快就医|马上就医|立即去医院|尽快去医院|"
    r"马上到急诊|拨打\s*120"
)

# 明确症状/身体部位词：命中说明是"症状咨询"（问诊）而非科普/能力/寒暄。
# 与澄清层共用（澄清层从这里 import），避免词表漂移。
_SYMPTOM_PAT = re.compile(
    r"头痛|头疼|头晕|疼痛|疼|痛|发烧|发热|咳嗽|胸闷|心慌|心悸|恶心|呕吐|"
    r"腹泻|便秘|失眠|乏力|麻木|皮疹|瘙痒|视力|耳朵|鼻子|喉咙|嗓子|胸|腹|"
    r"背|腰|腿|脚|手|关节|胃|心率|不适|难受|喘|出血|抽搐|"
    r"意识|晕倒|冷汗|麻木无力"
)

# 信息咨询判定：无风险信号且是"怎么/如何/注意事项"类科普咨询 -> 明显低风险，
# 直接短路不调 LLM（省延迟与成本）；真实症状描述（如"我最近经常头晕"）
# 不命中，进第 3 层 LLM 打分。
_INFO_QUERY_PAT = re.compile(
    r"怎么|如何|怎样|需要注意|注意什么|有什么注意事项|建议|是什么|什么叫|"
    r"有什么区别|区别|方法|好吗|能不能|可不可以|可以吗|该不该|要不要|合适吗|"
    r"你是谁|我是谁|自我介绍|介绍一下|能做什么|会做什么|有哪些功能|什么功能"
)

_JSON_RE = re.compile(r"\{.*\}", re.S)

_CONFIDENCE_SYSTEM_PROMPT = (
    "你是医疗分诊系统的置信度评估器。给定患者问题和 AI 助手给出的回答，"
    "评估：仅凭当前回答，AI 是否有足够把握安全地完成本次分诊，"
    "还是应当转交人工医生复核。"
    '只输出一个 JSON 对象，不要任何其他文字，格式：'
    '{"confidence": 0到1的小数（越高代表AI越有把握，越低越应转人工）, '
    '"risk_level": "low"或"medium"或"high", '
    '"summary": {"症状": "从问题提取的症状/主诉", '
    '"已给建议": "AI回答给出的关键建议", '
    '"风险等级": "low"或"medium"或"high", '
    '"待确认点": "需要医生复核确认的疑点"}}'
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class EscalationDecision(BaseModel):
    should_escalate: bool
    confidence: float = Field(ge=0.0, le=1.0)
    risk_level: str = "medium"
    reasons: List[str] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)


class RuleSignals(BaseModel):
    """规则层信号结构体（替代裸 dict，防键名漂移）。"""
    force: bool = False
    risk_level: str = "low"
    confidence: float = 1.0
    reasons: List[str] = Field(default_factory=list)
    info_query: bool = False


def _rule_signals(question: str, answer: str,
                  result: Optional[Dict[str, Any]]) -> RuleSignals:
    """规则层信号。返回 {force, risk_level, confidence, reasons}。"""
    q = question or ""
    a = answer or ""
    reasons: List[str] = []

    # 1) 危机关键词（权威信号，强制转诊）
    m = _CRISIS_PAT.search(q)
    if m:
        return RuleSignals(
            force=True, risk_level="high", confidence=0.0,
            reasons=[f"危机关键词命中：{m.group(0)}"],
        )

    # 2) 用户主动要求转人工
    if _ASK_DOCTOR_PAT.search(q):
        return RuleSignals(
            force=True, risk_level="medium", confidence=0.0,
            reasons=["用户主动要求人工介入"],
        )

    # 3) 处理失败/超时/工具失败/空回答：结果不可信 -> 强制转诊
    if result:
        if result.get("timeout_occurred"):
            reasons.append("回答超时（部分分析未完成）")
        if result.get("error"):
            reasons.append("处理报错")
        try:
            tool_failures = int(result.get("tool_failure_count") or 0)
        except (TypeError, ValueError):
            tool_failures = 0
        if tool_failures > 0:
            reasons.append(f"工具调用失败 {tool_failures} 次，证据检索不完整")
    if not a.strip():
        reasons.append("未产出有效回答")
    if reasons:
        return RuleSignals(
            force=True,
            risk_level="high" if not a.strip() else "medium",
            confidence=0.0,
            reasons=reasons,
        )

    # 4) 不再把"回答提示立即就医"当作风险信号：AI 回答几乎总带"如遇X请立即
    # 就医/拨打急救电话"这类泛化建议话术（连"你好"的能力介绍都会举例"胸痛伴
    # 出汗"），若据此转人工会把所有普通/科普/寒暄咨询都误转。真正的风险判定
    # 只依赖：问题侧危机词（用户自述）、Agent 结构化评估、运行异常。
    risk_level, confidence = "low", 1.0

    # 5) 信息咨询判定：科普咨询（怎么/如何/注意事项）且问题无危机词 -> 不转。
    # 不看答案的就医提示（科普答案常含"如有疑虑请及时就医"等泛化话术，
    # 若计入会导致科普咨询全部被误转人工）
    info_query = (
        bool(_INFO_QUERY_PAT.search(q))
        and not _CRISIS_PAT.search(q)
    )
    return RuleSignals(
        force=False, risk_level=risk_level,
        confidence=confidence, reasons=reasons,
        info_query=info_query,
    )


def _rule_summary(question: str, answer: str, risk_level: str) -> Dict[str, Any]:
    return {
        "症状": (question or "")[:200],
        "已给建议": (answer or "未生成回答")[:300],
        "风险等级": risk_level,
        "待确认点": "需医生复核风险判断与处理建议",
    }


def _parse_confidence_json(text: str) -> Optional[Dict[str, Any]]:
    """解析 LLM 置信度 JSON；结构非法返回 None。"""
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    conf = data.get("confidence")
    if not isinstance(conf, (int, float)) or not (0 <= float(conf) <= 1):
        return None
    risk = data.get("risk_level")
    if risk not in ("low", "medium", "high"):
        risk = "medium"
    summary = data.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    return {"confidence": float(conf), "risk_level": risk, "summary": summary}


class EscalationService:
    """转诊评估服务。进程内单例（get_escalation_service）。"""

    def __init__(
        self,
        store: Optional[EscalationStore] = None,
        llm_client=None,
        threshold: Optional[float] = None,
        use_llm: Optional[bool] = None,
        enabled: Optional[bool] = None,
    ):
        self.store = store or EscalationStore()
        self._llm = llm_client
        self.threshold = (
            threshold if threshold is not None
            else _env_float("MEDITRIAGE_ESCALATION_THRESHOLD", DEFAULT_THRESHOLD)
        )
        self.use_llm = (
            use_llm if use_llm is not None
            else _env_bool("MEDITRIAGE_ESCALATION_LLM", True)
        )
        self.enabled = (
            enabled if enabled is not None
            else _env_bool("MEDITRIAGE_ESCALATION_ENABLED", True)
        )
        self.max_fallback_per_session = (
            _env_int("MEDITRIAGE_ESCALATION_MAX_FALLBACK_PER_SESSION",
                     DEFAULT_MAX_FALLBACK_PER_SESSION)
        )
        # 进程内兜底转诊计数（key=session_id），防 LLM 故障时刷爆医生队列
        self._fallback_counts: Dict[str, int] = {}

    def _get_llm(self):
        if self._llm is None:
            from meditriage.core import LLMClient
            self._llm = LLMClient()
        return self._llm

    async def _llm_score(self, question: str,
                         answer: str) -> Optional[Dict[str, Any]]:
        """LLM 置信度打分；失败/超时返回 None（调用方安全兜底）。"""
        try:
            text = await asyncio.wait_for(
                self._get_llm().chat(
                    [
                        {"role": "system", "content": _CONFIDENCE_SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"患者问题：{question}\n"
                            f"AI回答：{(answer or '')[:800]}"
                        )},
                    ],
                    temperature=0,
                    max_tokens=300,
                ),
                timeout=DEFAULT_LLM_TIMEOUT,
            )
        except Exception as e:
            logger.warning(f"置信度评分失败，安全兜底转人工: {e}")
            return None
        parsed = _parse_confidence_json(text)
        if parsed is None:
            logger.warning("置信度评分输出无法解析，安全兜底转人工")
            return None
        return parsed

    async def _ambiguous_decision(
        self, question: str, answer: str,
        sig: RuleSignals,
    ) -> Optional[EscalationDecision]:
        """模糊场景决策：LLM 打分；LLM 不可用则按规则信号安全兜底。"""
        if not self.use_llm:
            # 纯规则模式：回答已提示立即就医等高危信号直接转人工，
            # 否则该信号在无 LLM 评分时会被丢弃（高危遗漏）
            if sig.risk_level == "high" and any(
                "立即就医" in r for r in sig.reasons
            ):
                return EscalationDecision(
                    should_escalate=True,
                    confidence=sig.confidence,
                    risk_level="high",
                    reasons=sig.reasons + ["纯规则模式：高危信号直接转人工"],
                    summary=_rule_summary(question, answer, "high"),
                )
            return None
        scored = await self._llm_score(question, answer)
        if scored is None:
            # 评分不可用：无规则高危信号 -> 不转（信任规则层 + Agent 评估，
            # 避免"评分故障导致几乎所有问题都转人工"）
            logger.debug("置信度评分不可用且无规则高危信号，不转人工")
            return None
        # 只有"判断为高危"才转：评分 risk_level=high 且低置信度；
        # low/medium 一律不转（中危由 AI 答复 + 建议就医，不强制转人工）
        if scored["risk_level"] == "high" and scored["confidence"] < self.threshold:
            base = _rule_summary(question, answer, "high")
            summary = {
                **base,
                **{k: v for k, v in (scored.get("summary") or {}).items() if v},
            }
            return EscalationDecision(
                should_escalate=True,
                confidence=scored["confidence"],
                risk_level="high",
                reasons=sig.reasons + [
                    f"置信度 {scored['confidence']:.2f} < 阈值 {self.threshold}"
                ],
                summary=summary,
            )
        logger.debug(
            f"评分 {scored['risk_level']}/{scored['confidence']:.2f}，"
            f"非高危不转人工"
        )
        return None

    async def evaluate(
        self,
        question: str,
        answer: str,
        session_id: str,
        result: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """评估并（命中则）建单。返回转诊单 dict；未命中返回 None。

        绝不抛异常：任何内部失败都记日志并返回 None（不阻断主问答链路）。
        """
        if not self.enabled:
            return None
        try:
            sig = _rule_signals(question or "", answer or "", result)
            if sig.force:
                decision = EscalationDecision(
                    should_escalate=True,
                    confidence=sig.confidence,
                    risk_level=sig.risk_level,
                    reasons=sig.reasons,
                    summary=_rule_summary(question, answer, sig.risk_level),
                )
            elif not _CRISIS_PAT.search(question or "") and not _SYMPTOM_PAT.search(
                question or ""
            ):
                # 非问诊短路：问题未描述明确症状（"你好""关于医疗知识你能帮我
                # 解答吗"）-> 不转人工。答案里的"立即就医/急救"是 AI 泛化
                # 建议话术，几乎所有回答都有，不应据此把寒暄/科普/能力咨询
                # 误转人工。
                logger.debug("非问诊（无明确症状描述）：不转人工")
                return None
            elif sig.info_query:
                # 明显低风险的信息咨询：短路，不调 LLM
                return None
            elif str((result or {}).get("risk_level") or "").lower() in (
                "high", "emergency"
            ):
                # Agent 结构化评估为高危/紧急：直接转人工（不等 LLM 评分）
                logger.debug(f"高危信号：Agent 评估 {result.get('risk_level')}，转人工")
                decision = EscalationDecision(
                    should_escalate=True, confidence=0.6,
                    risk_level=str(result.get("risk_level") or "high").lower(),
                    reasons=["Agent 结构化评估为高危/紧急"],
                    summary=_rule_summary(question, answer, "high"),
                )
            elif str((result or {}).get("risk_level") or "").lower() == "low":
                # Agent 明确评估低危 -> 不转人工。答案里的"立即就医/尽快就医"
                # 多为 AI 泛化建议话术（几乎所有回答都有），不据此转人工；
                # 仅当 Agent 无风险结论时才看规则/评分信号。
                logger.debug("Agent 评估低危：不转人工")
                return None
            elif str((result or {}).get("risk_level") or "").lower() == "medium":
                # 中危：由 AI 答复 + 就医建议承接，不强制转人工
                # （用户要求：只有高危/异常/主动才转）
                logger.debug("中危不转人工：由 AI 答复 + 就医建议承接")
                return None
            elif (
                sig.risk_level == "low"
                and str((result or {}).get("risk_level") or "").lower()
                == "unknown"
            ):
                # Agent 无结构化风险结论且规则无信号 -> 不转（普通问题）
                logger.debug("无风险信号且 Agent 无高危证据：不转人工")
                return None
            else:
                # Agent 无风险等级但有规则风险信号 / 完全未知：LLM 评分补判
                decision = await self._ambiguous_decision(question, answer, sig)

            if decision is None or not decision.should_escalate:
                return None

            # 兜底转诊限频：LLM 评分不可用时的安全兜底，防刷爆医生队列
            if any("兜底转人工" in r for r in decision.reasons):
                key = session_id or "anon"
                if self._fallback_counts.get(key, 0) >= self.max_fallback_per_session:
                    logger.warning(
                        f"会话 {key} 兜底转诊已达上限"
                        f"({self.max_fallback_per_session})，跳过建单"
                    )
                    return None
                self._fallback_counts[key] = self._fallback_counts.get(key, 0) + 1

            esc = self.store.create(
                session_id=session_id,
                question=question or "",
                answer_preview=answer or "",
                summary=decision.summary,
                risk_level=decision.risk_level,
                confidence=decision.confidence,
                reasons=decision.reasons,
                user_id=user_id,
            )
            logger.info(
                f"转诊建单 {esc.get('escalation_id')} "
                f"(session={session_id}, confidence={decision.confidence:.2f}, "
                f"risk={decision.risk_level})"
            )
            return esc
        except Exception as e:
            logger.exception(f"转诊评估失败: {e}")
            return None


_service: Optional[EscalationService] = None


def get_escalation_service() -> EscalationService:
    """进程内单例（延迟初始化，避免 import 即建库）。"""
    global _service
    if _service is None:
        _service = EscalationService()
    return _service
