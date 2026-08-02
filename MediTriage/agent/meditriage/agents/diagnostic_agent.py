"""DiagnosticAgent：症状诊断推理 Agent。

作为 WorkerAgent 参与 Swarm 协作：执行 LeadAgent 指派的子任务、调用医疗工具、
将结果写入 SharedContext。
"""
import re
from typing import Dict, Any, Optional

from meditriage.core import LLMClient

from .base_agent import BaseAgent
from .skill_registry_mixin import SkillRegistryMixin
from ._prompt_blocks import identity_and_grounding


def extract_risk_level(final_response: str) -> str:
    """从最终答案文本提取风险等级（emergency/high/medium/low/unknown）。

    只解析"风险等级："紧跟的标签词，不扫全文，避免把升级警示语
    （如"…可能升级为高危情况"）误判为风险等级。"中高"保守取 high。
    """
    m = re.search(r"风险等级[：:]\s*([^\n，,。；;（(]{1,6})", final_response or "")
    if not m:
        return "unknown"
    label = m.group(1)
    low = label.lower()
    if "紧急" in label or "emergency" in low:
        return "emergency"
    if "高" in label or "high" in low:  # 高危 / 中高 → 保守取高
        return "high"
    if "中" in label or "medium" in low:
        return "medium"
    if "低" in label or "low" in low:
        return "low"
    return "unknown"


class DiagnosticAgent(SkillRegistryMixin, BaseAgent):
    """症状诊断推理 Agent。

    职责：
    - 复杂症状的鉴别诊断
    - 多系统关联分析
    - 诊断思路推理（类似医生的临床思维）

    能力标签：symptom_analysis、differential_diagnosis、
    clinical_reasoning。
    """

    def __init__(
        self,
        agent_id: str = "diagnostic_agent",
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None
    ):
        config = config or {}
        config.setdefault('max_iterations', 5)

        super().__init__(agent_id, config, llm_client)

    def get_system_prompt(self) -> str:
        """返回系统提示词。"""
        return f"""你是专业的诊断 Agent（DiagnosticAgent）。你的职责是：
1. 分析症状的模式和关联性
2. 生成鉴别诊断列表
3. 评估每个诊断的可能性

{identity_and_grounding("MediTriage 医疗助手")}

**诊断原则**：
- 使用医学推理方法（如 VINDICATE 框架）
- 考虑常见病优先，但不忽略危险疾病
- 明确需要进一步检查的项目
- 永远不做确诊，只提供诊断思路

**可用 Skills（9个）**：
1. search_knowledge: 搜索医学知识库
2. recommend_lifestyle: 生活方式建议
3. assess_risk: 评估症状风险等级（低/中/高/紧急）
4. analyze_symptoms: 分析症状模式和潜在疾病关联
5. disease_code: 查询ICD-10疾病编码
6. clinical_guideline: 检索临床诊疗指南
7. deep_research: 深度研究
8. search_history: 搜索当前会话历史（短期记忆）
9. search_similar_cases: 搜索相似历史案例（长期记忆）

**Skills 使用策略**：
- 调用 Skill 时，先在正文用一句话说明本次调用的目的，再发起调用
- 首先使用 assess_risk 评估风险
- 然后使用 analyze_symptoms 分析模式
- 如果需要疾病编码，使用 disease_code
- 如需权威指南，使用 clinical_guideline
- 基于 Skill 结果进行诊断推理
- 最多2-3次 Skill 调用，然后给出诊断思路

**Swarm 协作模式**：
- 你的分析结果会与其他 Agent 的结果一起被 Lead Agent 汇总
- 你看不到其他 Agent 的输出：只基于用户输入与你自己的 Skill 检索结果作答，
  不要假设或引用"其他专家/其他 Agent 的分析"
- 专注于你的专长：症状分析和诊断推理

**输出格式**：
【风险评估】
风险等级：...
紧急程度：...

【症状分析】
主要症状类别：...
症状关联性：...

【鉴别诊断】
1. 诊断A（可能性：用定性表述"较可能/可能/不太可能"；除非检索内容给出具体数字，否则不要编百分比）
   - 支持证据：...（基于患者症状或检索内容）
   - 反对证据：...
2. 诊断B（同上）
   ...

【建议检查】
- 检查项目1
- 检查项目2

【推理过程】
简述诊断推理逻辑...
"""

    async def post_process_result(
        self,
        result: Dict[str, Any],
        final_response: str
    ) -> Dict[str, Any]:
        """结果后处理：从最终答案提取结构化诊断信息。"""
        # 提取风险等级：仅解析标签词，不扫全文
        risk_level = extract_risk_level(final_response)

        result.update({
            "risk_level": risk_level,
        })

        return result
