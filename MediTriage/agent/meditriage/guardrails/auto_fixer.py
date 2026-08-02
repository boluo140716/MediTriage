"""根据约束违规自动修复 Agent 输出。

修复方式只有一种：在原文前后补固定提示文本（免责声明 / 高危就医警告 /
越界纠正），不改写原文；要补哪些由 validator 检出后经 auto_fixable 传入。
"""
from typing import Dict, Any, List
from loguru import logger

from ._keywords import HIGH_RISK_KEYWORDS, mentions_doctor_visit


class AutoFixer:
    """约束违规自动修复器。"""

    def fix_output(self, output: str, auto_fixable: List[str]) -> str:
        """自动修复输出。

        Args:
            output: 原始输出
            auto_fixable: 可修复的违规列表

        Returns:
            修复后的输出
        """
        fixed_output = output

        for fix_type in auto_fixable:
            if fix_type == "add_disclaimer":
                fixed_output = self.fix_missing_disclaimer(fixed_output)
            elif fix_type == "add_emergency_warning":
                fixed_output = self.fix_high_risk_warning(fixed_output)
            elif fix_type == "add_overreach_caveat":
                fixed_output = self.fix_overreach_caveat(fixed_output)

        if fixed_output != output:
            logger.info("输出已自动修复")

        return fixed_output

    def fix_missing_disclaimer(self, output: str) -> str:
        """自动添加免责声明。

        Args:
            output: 原始输出

        Returns:
            添加免责声明后的输出
        """
        if "免责" not in output and "仅供参考" not in output:
            disclaimer = "\n\n【免责声明】\n以上信息仅供参考，不能替代专业医生的诊断和治疗。如有疑虑，请及时就医。"
            logger.debug("+ 自动添加免责声明")
            return output + disclaimer
        return output

    def fix_overreach_caveat(self, output: str) -> str:
        """越界内容纠正：追加"不构成诊断/处方"的强提示（幂等）。

        覆盖明确诊断 / 具体处方 / 自行用药 / 保证治愈 / 定夺侵入性治疗
        这些检测到但无法安全改写原文的违规：保留原文，但显式纠正其效力。
        """
        if "不构成诊断结论" in output:
            return output
        caveat = (
            "\n\n【请注意】以上内容仅为健康信息参考，不构成诊断结论或治疗、"
            "用药方案。具体诊疗请由执业医师面诊评估决定，切勿据此自行用药或处理。"
        )
        logger.debug("+ 自动追加越界纠正提示")
        return output + caveat

    def fix_high_risk_warning(self, output: str) -> str:
        """自动添加高危症状警告。

        Args:
            output: 原始输出

        Returns:
            添加警告后的输出
        """
        # 检查是否包含高危症状且未建议就医
        if any(kw in output for kw in HIGH_RISK_KEYWORDS):
            if not mentions_doctor_visit(output):
                warning = "**重要提醒**：您描述的症状可能提示严重问题，建议立即就医或拨打急救电话120，不要延误。\n\n"
                logger.debug("+ 自动添加高危症状警告")
                return warning + output

        return output
