"""证据综合器。

整合多个来源的信息，生成结构化的研究报告。
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger

from meditriage.core import LLMClient


@dataclass
class ResearchReport:
    """研究报告数据结构。"""
    query: str  # 原始查询
    key_findings: List[str] = field(default_factory=list)  # 关键发现
    evidence_level: str = "C"  # 证据等级（A/B/C）
    sources: List[Dict[str, str]] = field(default_factory=list)  # 信息来源
    confidence: float = 0.0  # 置信度 (0-1)
    conflicts: List[str] = field(default_factory=list)  # 信息冲突
    summary: str = ""  # 综合总结
    recommendations: List[str] = field(default_factory=list)  # 建议


class EvidenceSynthesizer:
    """证据综合器。

    功能：
    - 整合多个来源的信息
    - 识别信息冲突和一致性
    - 生成结构化的研究报告
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        初始化综合器

        Args:
            llm_client: LLM 客户端
        """
        self.llm_client = llm_client or LLMClient()

    async def synthesize(
        self,
        query: str,
        kb_results: List[Dict[str, Any]] = None,
        pubmed_results: List[Dict[str, Any]] = None
    ) -> ResearchReport:
        """综合本地知识库 + PubMed 文献证据，生成研究报告。

        Args:
            query: 研究问题
            kb_results: 本地 Milvus 知识库检索结果
            pubmed_results: PubMed 文献检索结果

        Returns:
            研究报告
        """
        logger.info(f"Synthesizing evidence for: {query}")

        if kb_results is None:
            kb_results = []
        if pubmed_results is None:
            pubmed_results = []

        # 构建综合提示
        prompt = self._build_synthesis_prompt(query, kb_results, pubmed_results)

        try:
            # 调用 LLM 进行综合
            response = await self.llm_client.chat([
                {"role": "user", "content": prompt}
            ])

            # 解析响应生成报告
            report = self._parse_response(
                query, response, kb_results, pubmed_results
            )

            logger.info(
                f"Research report generated: "
                f"{len(report.key_findings)} findings")
            return report

        except Exception as e:
            logger.error(f"Evidence synthesis error: {e}")
            # 返回空报告
            return ResearchReport(
                query=query,
                summary=f"综合失败：{str(e)}"
            )

    def _build_synthesis_prompt(
        self,
        query: str,
        kb_results: List[Dict[str, Any]],
        pubmed_results: List[Dict[str, Any]] = None
    ) -> str:
        """构建综合提示。"""
        pubmed_results = pubmed_results or []
        prompt = f"""你是医学证据综合专家。请整合以下来源的信息，回答用户问题。

【用户问题】
{query}

"""

        # 添加本地知识库检索结果（Milvus 返回的字典）
        if kb_results:
            prompt += "【本地知识库检索结果】\n"
            for i, doc in enumerate(kb_results[:5], 1):
                metadata = doc.get('metadata', {})
                prompt += f"{i}. {metadata.get('title', '医学知识')}\n"
                prompt += f"   内容: {doc.get('content', '')[:300]}...\n"
                prompt += f"   相似度: {doc.get('score', 0):.2f}\n\n"

        # 添加 PubMed 文献（权威、可引：标题/期刊/年份/PMID 均真实）
        if pubmed_results:
            prompt += "【PubMed 文献】（权威来源，可按标题/期刊/年份/PMID 引用）\n"
            for i, p in enumerate(pubmed_results[:5], 1):
                prompt += (
                    f"{i}. {p.get('title', '')}"
                    f"（{p.get('journal', '')}，{p.get('year', '')}；"
                    f"PMID:{p.get('pmid', '')}）\n"
                )
                abstract = (p.get('abstract') or '')[:300]
                if abstract:
                    prompt += f"   摘要: {abstract}...\n"
                prompt += "\n"

        prompt += """
**接地约束（必须遵守，违反即视为错误）**：
- 只能使用上面检索结果中**真实出现**的来源、年份、机构、指南名；上面没有的一律不得杜撰——不要写"2023年研究报告""2024年临床指南"等检索结果里不存在的具名出处或年份。
- 检索信息不足时，如实说明"现有检索来源有限"，可据通用医学常识谨慎作答，但不得包装成具名权威来源，也不得用"多项文献指出/有研究显示/与专家共识高度一致/具有较强临床指导意义"等无具体出处的泛化权威背书。

请生成综合研究报告，包含以下部分：

【关键发现】
- 列出 3-5 条最重要的发现
- 每条发现应简洁明确

【证据等级】（这是对**本次综合可信度**的标注，不是某外部权威报告的官方分级，表述时不要说成"某年报告/指南的证据等级"）
- A级：高质量随机对照试验或系统评价
- B级：队列研究或病例对照研究
- C级：专家共识或观察性研究
- 仅依据上方检索来源判断；来源未给明确分级时，按来源质量保守取 C，并视为"系统综合判断"

【信息来源】
- 列出主要参考来源（网站或文档标题）

【置信度】
- 0.0-1.0 之间的数值
- 基于信息来源的权威性和一致性

【信息冲突】
- 如果不同来源存在矛盾，明确指出
- 如果没有冲突，写"无明显冲突"

【综合总结】
- 200-300字的综合性回答
- 客观、专业、易懂

【建议】
- 给出 2-3 条实用建议
- 如需就医，明确指出

**输出格式**：
按照上述结构输出，使用【】标记各个部分。
"""

        return prompt

    def _parse_response(
        self,
        query: str,
        response: str,
        kb_results: List[Dict[str, Any]],
        pubmed_results: List[Dict[str, Any]] = None
    ) -> ResearchReport:
        """解析 LLM 响应。"""
        import re

        report = ResearchReport(query=query)

        # 提取关键发现
        findings_match = re.search(r'【关键发现】(.*?)【', response, re.DOTALL)
        if findings_match:
            findings_text = findings_match.group(1).strip()
            report.key_findings = [
                line.strip('- ').strip()
                for line in findings_text.split('\n')
                if line.strip() and line.strip().startswith('-')
            ]

        # 提取证据等级
        evidence_match = re.search(r'【证据等级】(.*?)【', response, re.DOTALL)
        if evidence_match:
            evidence_text = evidence_match.group(1).strip()
            if 'A级' in evidence_text or 'A 级' in evidence_text:
                report.evidence_level = "A"
            elif 'B级' in evidence_text or 'B 级' in evidence_text:
                report.evidence_level = "B"
            else:
                report.evidence_level = "C"

        # 提取置信度
        confidence_match = re.search(r'【置信度】(.*?)【', response, re.DOTALL)
        if confidence_match:
            confidence_text = confidence_match.group(1).strip()
            # 尝试提取数字
            numbers = re.findall(r'0\.\d+|\d+\.\d+', confidence_text)
            if numbers:
                try:
                    report.confidence = float(numbers[0])
                except:
                    report.confidence = 0.5
        else:
            report.confidence = 0.5

        # 提取信息冲突
        conflicts_match = re.search(r'【信息冲突】(.*?)【', response, re.DOTALL)
        if conflicts_match:
            conflicts_text = conflicts_match.group(1).strip()
            if "无" not in conflicts_text and "没有" not in conflicts_text:
                report.conflicts = [
                    line.strip('- ').strip()
                    for line in conflicts_text.split('\n')
                    if line.strip() and line.strip().startswith('-')
                ]

        # 提取综合总结
        summary_match = re.search(r'【综合总结】(.*?)【', response, re.DOTALL)
        if summary_match:
            report.summary = summary_match.group(1).strip()
        else:
            # 降级：使用整个响应
            report.summary = response[:500]

        # 提取建议
        recommendations_match = re.search(
            r'【建议】(.*?)(?:【|$)', response, re.DOTALL)
        if recommendations_match:
            recommendations_text = recommendations_match.group(1).strip()
            report.recommendations = [
                line.strip('- ').strip()
                for line in recommendations_text.split('\n')
                if line.strip() and line.strip().startswith('-')
            ]

        # 收集来源
        for doc in kb_results[:5]:
            metadata = doc.get('metadata', {})
            report.sources.append({
                "type": "knowledge_base",
                "title": metadata.get("title", "医学知识"),
                "id": doc.get('id', 'unknown')
            })
        for p in (pubmed_results or [])[:5]:
            report.sources.append({
                "type": "pubmed",
                "title": p.get("title", ""),
                "url": p.get("url", ""),
            })

        return report
