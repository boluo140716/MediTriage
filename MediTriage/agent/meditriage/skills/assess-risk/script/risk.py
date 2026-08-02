"""风险评估 Skill（依赖 RAG 知识库）。"""
from typing import Dict, Any
from loguru import logger

# 全局知识库实例（避免重复加载模型）
_kb_instance = None


def get_knowledge_base():
    """获取知识库单例。"""
    global _kb_instance
    if _kb_instance is None:
        from meditriage.knowledge.milvus_kb import MedicalKnowledgeBase
        _kb_instance = MedicalKnowledgeBase()
    return _kb_instance


# ──────────────────────────────────────────────────────────────────────
# 确定性风险分级（临床分诊"红旗"清单式）
# 宁可多报不可漏报：紧急/危及生命的判定用确定性规则，不交给 LLM 概率判断。
# 本结果是给诊断 Agent 的"风险先验"，Agent 的 LLM 会在此之上做细化（可下调到中危并给升级标准）。
# ──────────────────────────────────────────────────────────────────────

# 口语→临床术语归一（让红旗匹配不漏口语输入，如"发烧"→"发热"）
_SYNONYM = {
    "发烧": "发热", "发高烧": "高热", "拉肚子": "腹泻", "喘不上气": "呼吸困难",
    "喘不过气": "呼吸困难", "上不来气": "呼吸困难", "脖子硬": "颈强直", "脖子僵": "颈强直",
    "叫不醒": "昏迷", "不省人事": "昏迷", "不醒人事": "昏迷", "抽风": "抽搐", "嘴歪": "口角歪斜",
    "胸口痛": "胸痛", "胸口疼": "胸痛", "胸疼": "胸痛", "心口痛": "胸痛", "心口疼": "胸痛",
    "胸口剧痛": "胸痛", "胸口闷痛": "胸痛", "胸口绞痛": "胸痛", "心绞痛": "胸痛",
    "肚子疼": "腹痛", "肚子痛": "腹痛", "头晕": "眩晕",
}

# 紧急红旗（任一出现 → emergency，立即危及生命）
EMERGENCY_REDFLAGS = [
    "昏迷", "意识丧失", "神志不清", "昏迷不醒",
    "抽搐", "惊厥", "癫痫发作",
    "大出血", "咯血", "呕血", "便血", "喷血", "血崩",
    "呼吸停止", "窒息", "无法呼吸",
    "过敏性休克", "喉头水肿", "休克",
    "心跳骤停", "心脏骤停", "心脏停跳",
    "服毒", "自杀", "自残", "自尽", "轻生", "想死", "不想活", "急性中毒",
    "口角歪斜", "半身不遂", "偏瘫",   # 卒中明确体征
]

# 紧急组合（全部同时出现 → emergency）
EMERGENCY_COMBOS = [
    {"胸痛", "呼吸困难"}, {"胸痛", "冷汗"}, {"胸痛", "放射"},
    {"发热", "颈强直"},
    {"剧烈头痛", "喷射性呕吐"},
    {"突发", "言语不清"}, {"突发", "肢体无力"},
]

# 高危单症状（任一出现 → high）
HIGH_RISK_SYMPTOMS = [
    "胸痛", "胸闷", "呼吸困难", "气促", "气短",
    "意识模糊", "嗜睡", "反应迟钝",
    "剧烈头痛", "持续呕吐", "频繁呕吐", "喷射性呕吐",
    "高热不退", "持续高热", "高烧不退",
    "晕厥", "晕倒", "黑朦",
    "剧烈腹痛", "腹痛难忍",
    "面部下垂", "言语含糊", "言语不清", "肢体麻木", "肢体无力",
    "血尿", "黑便",
    "严重脱水", "少尿", "无尿",
]

# 中危组合（提示需重视，但非立即危险 → medium）
MEDIUM_RISK_COMBOS = [
    {"发热", "皮疹"}, {"腹痛", "发热"}, {"头痛", "发热"}, {"头痛", "呕吐"},
]

# 中危修饰词（病程/程度，提示需关注 → medium）
MEDIUM_RISK_KEYWORDS = [
    "持续", "加重", "反复", "严重", "剧烈", "好几天", "数日", "多日",
    "不缓解", "越来越", "一直",
]


def _normalize(text: str) -> str:
    t = text or ""
    for k, v in _SYNONYM.items():
        t = t.replace(k, v)
    return t


def _grade_risk(symptoms: str):
    """确定性分级，返回 (risk_level, reasons)。优先级 emergency > high > medium > low。"""
    text = _normalize(symptoms)
    for kw in EMERGENCY_REDFLAGS:
        if kw in text:
            return "emergency", [f"紧急红旗症状：{kw}"]
    for combo in EMERGENCY_COMBOS:
        if all(c in text for c in combo):
            return "emergency", [f"紧急症状组合：{'＋'.join(sorted(combo))}"]
    hits = [kw for kw in HIGH_RISK_SYMPTOMS if kw in text]
    if hits:
        return "high", [f"高风险症状：{kw}" for kw in hits]
    for combo in MEDIUM_RISK_COMBOS:
        if all(c in text for c in combo):
            return "medium", [f"需关注的症状组合：{'＋'.join(sorted(combo))}"]
    for kw in MEDIUM_RISK_KEYWORDS:
        if kw in text:
            return "medium", [f"病程/程度提示需关注：{kw}"]
    return "low", []


def _recommendation(level: str) -> str:
    return {
        "emergency": "危急！请立即拨打 120 或前往最近急诊，切勿等待或自行处理",
        "high": "建议立即就医或拨打急救电话 120",
        "medium": "建议尽快就医，不要拖延，必要时前往医院",
    }.get(level, "建议密切观察症状变化，如果症状加重或持续不缓解，应及时就医")


async def assess_risk(symptoms: str) -> Dict[str, Any]:
    """评估症状风险等级。

    Args:
        symptoms: 症状描述（字符串）

    Returns:
        {
            "answer": "格式化的风险评估结果",
            "risk_level": "low/medium/high/emergency",
            "recommendation": "就医建议"
        }
    """
    logger.info(f"Assessing risk: symptoms={symptoms}")

    # 确定性红旗分级（emergency>high>medium>low），口语已归一化
    risk_level, reasons = _grade_risk(symptoms)
    recommendation = _recommendation(risk_level)

    # 从 RAG 知识库获取风险相关的医学建议
    import asyncio
    kb_advice = None
    try:
        # 重型同步检索放线程池，避免阻塞事件循环
        kb = await asyncio.to_thread(get_knowledge_base)
        # 根据风险等级查询相关医学知识
        risk_query = f"{symptoms} 紧急程度 风险评估 就医建议"
        results = await asyncio.to_thread(
            kb.search,
            query=risk_query,
            top_k=1,
            filter_type=None,
        )
        if results and results[0]["score"] > 0.5:
            kb_advice = results[0]["content"][:300]  # 限制长度
    except Exception as e:
        logger.warning(f"Failed to get KB advice: {e}")

    return {
        "answer": format_assessment(
            symptoms, risk_level, reasons, recommendation, kb_advice
        ),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "kb_advice": kb_advice
    }


def format_assessment(
    symptoms: str,
    level: str,
    reasons: list,
    recommendation: str,
    kb_advice: str = None,
) -> str:
    """格式化风险评估结果。"""
    level_map = {
        "low": "低危",
        "medium": "中危",
        "high": "高危",
        "emergency": "紧急"
    }

    output = [
        f"【症状风险评估】",
        f"\n症状描述：{symptoms}",
        f"\n风险等级：{level_map.get(level, level)}",
    ]

    if reasons:
        output.append("\n风险因素：")
        for reason in reasons:
            output.append(f"  • {reason}")

    output.append(f"\n就医建议：{recommendation}")

    # 添加来自知识库的医学建议
    if kb_advice:
        output.append("\n【医学知识库补充】")
        output.append(kb_advice)

    if level == "high" or level == "emergency":
        output.append("\n请立即就医或拨打 120！")

    output.append("\n数据来源：风险规则引擎 + 医学知识库（Milvus RAG）")

    return "\n".join(output)
