"""约束验证器，运行时检查 Agent 行为是否违反约束。

约束清单见同目录 agent_constraints.yaml，每个键对应本文件一处检查；
检出的违规附 auto_fixable 标记，可修复项交 AutoFixer 处理。
"""
import re
from typing import Dict, Any
import yaml
from pathlib import Path
from loguru import logger

from ._keywords import (
    HIGH_RISK_KEYWORDS, DRUG_CONTEXT_PAT, mentions_doctor_visit,
)

# 药名近距剂量（模式4 专用）：不含裸 g/克——膳食语境（盐5克/钠2克）
# 与药类词近距共现时不应误报处方
_RX_DOSE_NEAR = re.compile(
    r"\d+(\.\d+)?\s*(mg/kg|mg|毫克|ml|毫升|片|粒|滴|IU)"
)

# 命中点前置窗口里出现这些否定词 → 是安全劝阻而非违规（"请勿自行用药"）
_NEGATION_PAT = re.compile(r"请勿|不要|不可|不能|不应|不建议|切勿|避免|无法|难以|勿|别")


def _hit_without_negation(pattern: re.Pattern, text: str, window: int = 6) -> bool:
    """正则命中且命中点前 window 字内无否定词，才算违规。

    命中点紧前一个字是否定字（"不建议手术"从"建议"起匹配，"不"被拆出）
    也视为否定。
    """
    for m in pattern.finditer(text):
        pre = text[max(0, m.start() - window):m.start()]
        if pre and pre[-1] in "不没别勿莫非":
            continue
        if _NEGATION_PAT.search(pre):
            continue
        return True
    return False


class ConstraintValidator:
    """约束验证器。"""

    def __init__(self):
        """初始化约束验证器（约束 yaml 锚定在本模块同目录，不依赖 CWD）。"""
        # 加载 Agent 约束
        agent_path = Path(__file__).parent / "agent_constraints.yaml"
        with open(agent_path, 'r', encoding='utf-8') as f:
            self.agent_constraints = yaml.safe_load(f)

        logger.info("ConstraintValidator initialized")

    def validate_tool_call(
        self, agent_id: str, tool_name: str
    ) -> Dict[str, Any]:
        """验证工具调用是否允许。

        Args:
            agent_id: Agent ID
            tool_name: 工具名称

        Returns:
            {
                "valid": bool,
                "reason": str (如果不允许)
            }
        """
        agent_constraints = self.agent_constraints['agents'].get(agent_id, {})
        allowed_tools = agent_constraints.get('allowed_tools', [])

        # 如果 allowed_tools 为空，表示没有限制
        if not allowed_tools:
            return {"valid": True}

        # 检查工具是否在允许列表中（告警由调用方 agent_loop 统一打，
        # 此处不另打 logger.warning，避免同一事件双条告警）
        if tool_name not in allowed_tools:
            reason = f"工具 {tool_name} 不在 {agent_id} 的推荐工具列表中"
            return {"valid": False, "reason": reason}

        return {"valid": True}

    def validate_output(self, agent_id: str, output: str) -> Dict[str, Any]:
        """验证输出是否符合约束。

        Args:
            agent_id: Agent ID
            output: Agent 的输出文本

        Returns:
            {
                "valid": bool,
                "violations": List[str],
                "auto_fixable": List[str]  # 可以自动修复的违规
            }
        """
        agent_constraints = self.agent_constraints['agents'].get(agent_id, {})
        output_constraints = agent_constraints.get('output_constraints', [])
        common_constraints = self.agent_constraints.get('common', {}).get(
            'output_constraints', [])

        # 合并约束
        all_constraints = output_constraints + common_constraints

        violations = []
        auto_fixable = []

        # 检查免责声明
        if 'must_include_disclaimer' in all_constraints:
            if ('免责声明' not in output
                    and 'disclaimer' not in output.lower()
                    and '仅供参考' not in output):
                violations.append("缺少免责声明")
                auto_fixable.append("add_disclaimer")

        # 检查长度限制
        max_length_constraint = next(
            (c for c in all_constraints
             if isinstance(c, dict) and 'max_response_length' in c),
            None
        )
        if max_length_constraint:
            max_length = max_length_constraint.get('max_response_length')
            if len(output) > max_length:
                violations.append(f"回答过长（{len(output)} > {max_length}字）")

        # 检查高危症状必须建议就医
        if 'must_recommend_doctor_visit_if_high_risk' in all_constraints:
            if any(kw in output for kw in HIGH_RISK_KEYWORDS):
                if not mentions_doctor_visit(output):
                    violations.append("高危症状未建议就医")
                    auto_fixable.append("add_emergency_warning")

        # 检查禁止行为（越界安全边界）
        forbidden_actions = agent_constraints.get('forbidden_actions', [])
        # 主语锚定 + 否定前置排除："高血压就是指…"/"不能确诊为…"不算诊断
        if 'diagnose_disease' in forbidden_actions:
            if _hit_without_negation(
                    re.compile(
                        r'(您|你|患者)(患有|就是|肯定是|得的是)'
                        r'|确诊为|可以确定(是|为)'
                    ),
                    output):
                violations.append("包含明确诊断（越界行为）")

        if 'prescribe_medication' in forbidden_actions:
            # 只检测明确的药物处方模式（避免误报）
            # 模式1: 具体药物剂量（如：硝苯地平20mg）
            if re.search(r'(药物|药品|药).{0,10}(\d+\s*(mg|g|毫克|克))', output):
                violations.append("包含具体药物处方（越界行为）")
            # 模式2: 用药频率和剂量（如：每日3次，每次10mg）
            elif re.search(
                    r'每(日|天|次).{0,5}\d+\s*次.{0,10}(\d+\s*(mg|g|毫克|克))',
                    output):
                violations.append("包含具体药物处方（越界行为）")
            # 模式3: 明确的处方建议（如：建议服用XX 20mg / 可以吃XX 500mg）
            elif re.search(
                    r'(建议|推荐|可以)(服用|使用|吃)'
                    r'.{0,15}\d+\s*(mg|g|毫克|克|ml|毫升|片|粒|滴)',
                    output):
                violations.append("包含具体药物处方（越界行为）")
            # 模式4: 真实处方写药名不写"药"字——药名与剂量近距共现
            #（膳食用量如"钾摄入3500mg"无药名共现，不会误伤）
            else:
                for m in DRUG_CONTEXT_PAT.finditer(output):
                    lo = max(0, m.start() - 25)
                    hi = min(len(output), m.end() + 25)
                    if _RX_DOSE_NEAR.search(output[lo:hi]):
                        violations.append("包含具体药物处方（越界行为）")
                        break

        # 不建议自行用药/治疗（绕过就医）；否定前置排除"请勿自行服用"
        if 'suggest_self_treatment' in forbidden_actions:
            if _hit_without_negation(
                    re.compile(
                        r'(自行|自己)(购买|服用|用药|吃药)|在家(自行)?用药'
                        r'|建议(自行|自己)(治疗|处理)'
                    ),
                    output):
                violations.append("建议自行用药/治疗（越界行为）")

        # 不保证治愈（过度承诺）；"不能保证治愈"是合规表述
        if 'guarantee_cure' in forbidden_actions:
            if _hit_without_negation(
                    re.compile(
                        r'保证(治愈|根治|痊愈|好)|彻底(治愈|根治)|包治'
                        r'|(一定|肯定|百分之百|100%)(能|可以|会)?'
                        r'(治愈|治好|痊愈|根治)'
                    ),
                    output):
                violations.append("保证治愈（过度承诺，越界行为）")

        # 不定夺侵入性治疗/手术方案（应引导就医评估）；连接词可省
        #（"建议尽快手术"），"不建议手术"经否定排除
        if 'recommend_treatment' in forbidden_actions:
            if _hit_without_negation(
                    re.compile(
                        r'(建议|推荐|你应该|你需要|必须)(立即|尽快)?'
                        r'(做|接受|进行)?(手术|化疗|放疗|透析|介入治疗)'
                    ),
                    output):
                violations.append("定夺侵入性治疗方案（越界行为）")

        # 越界类违规统一可修复：追加纠正性安全提示（不构成诊断/处方）
        if any("越界行为" in v for v in violations):
            auto_fixable.append("add_overreach_caveat")

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "auto_fixable": auto_fixable
        }
