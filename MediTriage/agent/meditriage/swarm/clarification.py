"""澄清式多轮问诊：信息不足时主动追问（P1-5）。

设计：
- 规则预筛（零延迟）：危机 / 科普咨询 / 信息已完整 -> 不追问；
  仅"症状模糊、信息不足"（缺关键分诊维度）进入追问链路。
- 逐轮追问：一次只问 1 个问题（按优先级取最关键的缺失维度：
  时长 > 程度 > 伴随 > 既往），用户回答后下一轮基于会话累计信息
  重新评估还缺什么、继续追问，直到关键维度补全或达到追问轮上限
  （更贴近真人医生问诊节奏；SUFFICIENT 不再短路，缺维度即继续问）。
- 判定规则化：是否追问由规则确定（缺维度），不依赖 LLM 自由裁量，
  保证同一主诉行为稳定（实测 LLM 判定 should_clarify 会随机波动）。
- LLM 只负责生成问题：围绕当前缺失维度生成 1 个追问；失败/非法
  时用模板问题兜底（信息不足必追问，不落空）。
- 追问轮上限（默认 3 轮/会话）：用短期记忆 [CLARIFY-ROUND] 标记计数。
- 降级安全：LLM 完全不可用时模板兜底仍可追问；任何异常都不阻断主链路。

与转诊判定的关系：危机信号永远优先——预筛命中危机即不追问，
由 escalation 层强制转人工（绝不因追问延误危机处理）。
"""
import asyncio
import json
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger

# 复用转诊判定同一套危机/症状规则，避免正则漂移
from meditriage.workflows.escalation import _CRISIS_PAT, _SYMPTOM_PAT


class PrecheckResult(str, Enum):
    """规则预筛结果（str Enum：与字符串比较兼容，且防键名漂移）。"""
    CRISIS = "crisis"          # 危机信号：不追问，交转诊评估拦截
    INFO_QUERY = "info_query"  # 泛指科普：不追问，直接回答
    SUFFICIENT = "sufficient"  # 信息已较完整：不追问
    UNCLEAR = "unclear"        # 信息不足：进入轻量 LLM 判定


# ---- env 开关与上限 ----
DEFAULT_MAX_CLARIFY_ROUNDS = 3
CLARIFY_ENABLED = os.environ.get("MEDITRIAGE_CLARIFY", "1").strip().lower() in (
    "1", "true", "yes", "on"
)
CLARIFY_LLM_TIMEOUT = 8.0

_CLARIFY_MARKER = "[CLARIFY-ROUND]"
_JSON_RE = re.compile(r"\{.*\}", re.S)

# 个人主体（我/家人...）——用于区分"个人问诊"与"泛指科普"（已弃用，
# 见 rule_precheck：问诊判据只看是否描述症状词，不带"我"字）
_PERSONAL_SUBJECT_PAT = re.compile(r"我|我的|家人|孩子|老人|爸爸|妈妈|儿子|女儿|老婆|老公")

# 信息完整性维度：命中 >=2 个维度视为"信息已较完整，无需追问"
_SUFFICIENT_DIMENSIONS = (
    # 时长/频率/时间点
    re.compile(
        r"多久|几天|几周|几个月|几年|一周|两周|一个月|半年|长期|反复|"
        r"经常|偶尔|今天|昨天|前天|早晨|晚上|早上|睡前|饭后|运动后|运动时|休息时|"
        r"[一二两三四五六七八九十\d]+\s*天|[一二两三四五六七八九十\d]+\s*周|[一二两三四五六七八九十\d]+\s*个月"
    ),
    # 程度/性质
    re.compile(
        r"厉害|严重|剧烈|轻微|轻度|中度|重度|加重|减轻|缓解|越来越|"
        r"刺痛|钝痛|胀痛|烧灼|阵发性|持续性"
    ),
    # 伴随/诱因/并发（症状体征词在此：发热/呕吐/腹泻/皮疹等；
    # 含否定式回答："没有发热""无伴随症状"也算已澄清）
    re.compile(
        r"伴随|同时|还有|伴有|以及|加上|合并|诱因|"
        r"发烧|发热|体温|呕吐|腹泻|皮疹|"
        r"没有.{0,8}(症状|不适|发热|麻木|疼痛|恶心|呕吐|头晕|胸闷|咳嗽|乏力|其他|异常)|"
        r"无.{0,8}(上述|其他|伴随|症状|不适)"
    ),
    # 既往史/用药/基础病（症状体征词已归入伴随维度）
    re.compile(
        r"既往|病史|以前|在吃|服用|吃药|用药|过敏|血压|血糖"
    ),
)

# 维度名（与 _SUFFICIENT_DIMENSIONS 一一对应，用于缺失维度计算与兜底问题）
_DIMENSION_NAMES = ("症状持续时间", "严重程度", "伴随症状", "既往史/用药")


def _missing_dimensions(question: str) -> List[str]:
    """返回患者描述中缺失的关键分诊维度名（时长/程度/伴随/既往）。"""
    q = question or ""
    return [
        name for name, pat in zip(_DIMENSION_NAMES, _SUFFICIENT_DIMENSIONS)
        if not pat.search(q)
    ]


_CLARIFY_SYSTEM_PROMPT = (
    "你是医疗分诊系统的信息采集助手。医生接诊时需要先收集必要信息才能安全判断。"
    "患者描述缺少以下关键分诊信息：__MISSING__。\n"
    "请围绕这个缺失信息生成 1 个最关键的追问问题，一句话、具体、可回答，"
    "不要询问患者已经提供的信息。\n"
    "只输出一个 JSON 对象，不要任何其他文字，格式：\n"
    "{\"questions\": [\"问题1\"]}"
)


def rule_precheck(question: str, answer: str) -> PrecheckResult:
    """规则预筛。返回 PrecheckResult。

    区分科普与问诊的关键：未描述明确症状词 -> 科普/能力/一般咨询，直接回答
    （如"高血压饮食注意什么""你能帮我吗""我没说我生病"）；否则按信息完整性
    判定（真正需要追问的是用户描述了自己/家人的症状）。
    危机判定只看问题侧（患者自述）：危机词命中 -> 不追问，交给转诊评估拦截
    （绝不因追问延误危机处理）。注意不能查 AI 答案侧——答案几乎总会科普
    "拨打120"等危险信号提示（_CRISIS_PAT 含 "120"），查答案侧会误伤，
    导致所有模糊主诉都跳过追问；答案侧风险由转诊评估（escalation）独立兜底。
    """
    q = question or ""
    if _CRISIS_PAT.search(q):
        return PrecheckResult.CRISIS
    if not _SYMPTOM_PAT.search(q):
        # 未描述明确症状（疼/头晕/发烧等）-> 科普/能力/一般咨询，直接回答不追问。
        # 注意不能以"是否含个人主体（我/家人）"来区分：能力咨询（"你能帮我吗"）、
        # 澄清话术（"我没说我生病"）都带"我"，但并非问诊，不应被追问症状。
        return PrecheckResult.INFO_QUERY
    if sum(1 for pat in _SUFFICIENT_DIMENSIONS if pat.search(q)) >= 2:
        return PrecheckResult.SUFFICIENT
    return PrecheckResult.UNCLEAR


def count_clarify_rounds(messages: List[Dict[str, Any]]) -> int:
    """统计会话内已发生的澄清轮数（按 [CLARIFY-ROUND] 标记）。"""
    return sum(
        1 for m in (messages or [])
        if str(m.get("content") or "").startswith(_CLARIFY_MARKER)
    )


def _parse_clarify_json(text: str) -> Optional[List[str]]:
    """解析 LLM 生成的问题 JSON；结构非法返回 None（由调用方兜底补齐）。

    兼容旧格式（带 should_clarify/reason 字段），只取 questions 去重截断。
    """
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    qs = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(qs, list):
        return None
    seen, uniq = set(), []
    for x in qs:
        st = str(x).strip()
        if st and st not in seen:
            seen.add(st)
            uniq.append(st)
    return uniq[:3] if uniq else None


# ---- 模板兜底问题（LLM 失败/不足时逐维度兜底）----
_FALLBACK_BY_DIMENSION = {
    "症状持续时间": "这个症状持续多久了？是第一次出现还是反复发作？",
    "严重程度": "症状有多严重？会影响日常活动或睡眠吗？",
    "伴随症状": "除了这个症状，还有没有其他伴随表现（如发热、麻木、乏力、恶心等）？",
    "既往史/用药": "以前有过类似情况吗？有没有高血压、糖尿病等基础疾病或长期用药？",
}


def _fallback_questions(missing: List[str]) -> List[str]:
    """按缺失维度生成模板追问问题（每个缺失维度一个问题）。"""
    return [
        _FALLBACK_BY_DIMENSION[d] for d in missing
        if d in _FALLBACK_BY_DIMENSION
    ]


async def _generate_question(llm_client, question: str,
                             dim: str) -> Optional[str]:
    """LLM 围绕单个缺失维度生成 1 个追问问题；失败/超时/非法返回 None（模板兜底）。"""
    if llm_client is None:
        return None
    try:
        prompt = _CLARIFY_SYSTEM_PROMPT.replace("__MISSING__", dim)
        text = await asyncio.wait_for(
            llm_client.chat(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"患者描述：{question}"},
                ],
                temperature=0,
                max_tokens=200,
            ),
            timeout=CLARIFY_LLM_TIMEOUT,
        )
        qs = _parse_clarify_json(text) or []
        return qs[0] if qs else None
    except Exception as exc:
        logger.warning(f"澄清问题生成失败，使用模板兜底: {exc}")
        return None


async def maybe_clarify(
    question: str,
    session_id: str,
    final_answer: str,
    llm_client,
    memory,
    max_rounds: int = DEFAULT_MAX_CLARIFY_ROUNDS,
    enabled: bool = CLARIFY_ENABLED,
) -> Optional[Dict[str, Any]]:
    """信息不足时逐轮追问。返回 None（不追问）或
    {"questions": [1 个问题], "reason": "..."}。绝不抛异常。

    一次只问 1 个问题（按优先级取最关键的缺失维度），用户回答后下一轮
    重新评估还缺什么继续追问，直到信息足够或达到追问轮上限。
    判定规则化：UNCLEAR（缺关键维度）-> 必追问，不依赖 LLM 判定；
    LLM 只负责生成问题，失败/不足用模板兜底。
    """
    if not enabled:
        return None
    # 用户主动要求转人工：跳过澄清追问（直接走回答+转诊，不因信息不足被拦截）
    from meditriage.workflows.escalation import _ASK_DOCTOR_PAT
    if _ASK_DOCTOR_PAT.search(question or ""):
        logger.info("用户主动要求转人工：跳过澄清追问")
        return None
    try:
        msgs = memory.get_recent_messages(session_id, limit=200) if memory else []
        rounds = count_clarify_rounds(msgs)
        if rounds >= max_rounds:
            return None
        # 追问是累积的：用"历史用户消息 + 当前输入"评估缺失维度，
        # 避免重复问用户已回答过的维度（如历史已给时长、本轮只答程度）
        hist = " ".join(
            str(m.get("content") or "") for m in (msgs or [])
            if m.get("role") == "user"
        )
        combined = f"{hist} {question}".strip()
        pre = rule_precheck(combined, final_answer)
        # 硬短路：危机（转人工）/ 泛指科普（直接回答）不追问；
        # SUFFICIENT 不再短路——逐轮追问模式下，缺任意关键维度都继续问
        # （用户回答完再追问，直到补全或达追问轮上限）
        if pre is PrecheckResult.CRISIS or pre is PrecheckResult.INFO_QUERY:
            return None
        missing = _missing_dimensions(combined)
        if not missing:
            return None
        # 追问轮中用户未提供任何可识别维度信息（纯回应性回答："会""有一点"
        # "不太清楚"等）-> 视为已回答当前最优先缺失维度，跳过它问下一个，
        # 避免卡在同一问题死循环（真实回答往往不含关键词：如"有点影响
        # 日常活动"不含"严重/轻度"等程度词）。首轮主诉（rounds==0）不跳过，
        # 永远从最高优先级缺失维度问起。
        if (
            rounds > 0
            and not any(pat.search(question or "") for pat in _SUFFICIENT_DIMENSIONS)
        ):
            missing = missing[1:]
            if not missing:
                return None
        # 只问最关键的缺失维度（时长 > 程度 > 伴随 > 既往）
        dim = missing[0]
        q = await _generate_question(llm_client, question, dim)
        if not q:
            q = _FALLBACK_BY_DIMENSION.get(dim) or "请补充一下相关情况，以便更准确地判断。"
        logger.info(f"澄清追问 1 题（维度：{dim}）")
        return {
            "questions": [q],
            "reason": f"还缺少{dim}等信息，补充后给出更准确判断",
        }
    except Exception as exc:
        logger.warning(f"澄清判定失败，降级为直接回答: {exc}")
        return None


def clarify_memory_content(questions: List[str]) -> str:
    """把追问写入短期记忆的固定格式（供下一轮上下文识别）。"""
    return f"{_CLARIFY_MARKER}\n" + "\n".join(questions)
