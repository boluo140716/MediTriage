"""检索结果 → 人类可读引用。

把 Milvus chunk 的 metadata（source/doc_id/section/year）组装成用户能看懂的
来源标注，供 Skill 的格式化输出与前端"证据来源"面板共用。索引里没有的
字段一律不编造：标题映射缺失时退回机构 + doc_id。
"""
import re
from typing import Any, Dict, List

# doc_id → 可读标题（与 MediTriage/data/rag_corpus 入库语料一一对应）
DOC_TITLES: Dict[str, str] = {
    "ada_diabetes_2025": "ADA《糖尿病诊疗标准》2025",
    "aha_acc_chol_2018": "AHA/ACC《血脂管理指南》2018",
    "aha_acc_chol_highlights_2019": "AHA/ACC 血脂指南要点解读 2019",
    "aha_acc_hcm_2024": "AHA/ACC《肥厚型心肌病指南》2024",
    "aha_acc_vhd_2020": "ACC/AHA《心脏瓣膜病管理指南》2020",
    "aha_acs_2025": "ACC/AHA《急性冠脉综合征管理指南》2025",
    "esc_acs_2023": "ESC《急性冠脉综合征管理指南》2023",
    "esc_afib_2024_essential": "ESC《房颤管理指南（精要版）》2024",
    "esc_hf_2023_focused": "ESC《心力衰竭指南 2023 重点更新》",
    "esc_htn_2024": "ESC《血压升高与高血压管理指南》2024",
    "esc_pe_2019": "ESC《急性肺栓塞诊断与管理指南》2019",
    "gina_asthma_2025_summary": "GINA《哮喘管理与预防策略（摘要）》2025",
    "gold_copd_2025": "GOLD《慢阻肺诊断、管理与预防全球策略》2025",
    "kdigo_ckd_2024": "KDIGO《慢性肾脏病评估与管理指南》2024",
    "who_hypertension_treatment_2021": "WHO《成人高血压药物治疗指南》2021",
    "01_lifestyle_hypertension": "本地知识库·高血压生活方式",
    "02_lifestyle_diabetes": "本地知识库·糖尿病生活方式",
    "03_lifestyle_cold": "本地知识库·感冒生活护理",
    "04_lifestyle_general_health": "本地知识库·一般健康生活方式",
    "05_symptoms_emergency": "本地知识库·急症症状识别",
    "10_icd10_cardiovascular": "本地知识库·ICD-10 心血管疾病编码",
    "11_icd10_endocrine": "本地知识库·ICD-10 内分泌疾病编码",
    "12_icd10_infectious": "本地知识库·ICD-10 感染性疾病编码",
    "20_guideline_hypertension": "本地知识库·高血压诊疗要点",
    "21_guideline_diabetes": "本地知识库·糖尿病诊疗要点",
}

_MD_NOISE = re.compile(r"[*_#`>【】\[\]]+")
_WS = re.compile(r"\s+")


def _clean(text: str, limit: int) -> str:
    """去 markdown 噪声、压空白、截断。"""
    t = _WS.sub(" ", _MD_NOISE.sub(" ", text or "")).strip()
    return t[:limit] + ("…" if len(t) > limit else "")


_SENT_SPLIT = re.compile(r"(?<=[。！？.!?])\s+|\n+")


def _hit_window(text: str, query: str, limit: int) -> str:
    """命中句截取：按 query 词重叠选最相关句，居中截取上下文。

    reranker 打高分的关键句常在 chunk 中后段，恒取头部的摘录可能完全不含命中要点。
    query 为空或无重叠时退回头部截断。轻量纯字符串、不触 GPU。
    """
    t = _WS.sub(" ", _MD_NOISE.sub(" ", text or "")).strip()
    if not query or len(t) <= limit:
        return t[:limit] + ("…" if len(t) > limit else "")
    qterms = {w.lower() for w in re.findall(r"[A-Za-z0-9]{2,}|[一-鿿]{2,}", query)}
    if not qterms:
        return t[:limit] + "…"
    sents = [s for s in _SENT_SPLIT.split(t) if s.strip()]
    best_i, best_hit = 0, 0
    for i, s in enumerate(sents):
        sl = s.lower()
        hit = sum(1 for q in qterms if q in sl)
        if hit > best_hit:
            best_hit, best_i = hit, i
    if best_hit == 0:
        return t[:limit] + "…"
    # 以最相关句为中心，向两侧拼相邻句直到接近 limit
    window, lo, hi, total = [sents[best_i]], best_i - 1, best_i + 1, len(sents[best_i])
    while total < limit and (lo >= 0 or hi < len(sents)):
        if hi < len(sents) and total + len(sents[hi]) <= limit:
            window.append(sents[hi]); total += len(sents[hi]); hi += 1
        elif lo >= 0 and total + len(sents[lo]) <= limit:
            window.insert(0, sents[lo]); total += len(sents[lo]); lo -= 1
        else:
            break
    out = " ".join(window)
    prefix = "…" if lo >= 0 and window and window[0] != sents[0] else ""
    suffix = "…" if hi < len(sents) else ""
    return (prefix + out[:limit] + suffix).strip()


def source_label(metadata: Dict[str, Any]) -> str:
    """单条 chunk 的来源标注，如「ESC《血压升高与高血压管理指南》2024」。"""
    doc_id = str(metadata.get("doc_id", "") or "")
    if doc_id in DOC_TITLES:
        return DOC_TITLES[doc_id]
    if doc_id.startswith("medlineplus:"):
        topic = doc_id.split(":", 1)[1].replace("_", " ").strip()
        return f"MedlinePlus 健康主题：{topic}"
    # 兜底：机构 + 年份 + doc_id，绝不编造标题
    org = str(metadata.get("source", "") or "")
    year = str(metadata.get("year", "") or "")
    parts = [p for p in (org, year, doc_id) if p]
    return " ".join(parts) if parts else "医学知识库"


def build_citations(
    results: List[Dict[str, Any]], excerpt_chars: int = 300, query: str = ""
) -> List[Dict[str, Any]]:
    """检索结果列表 → 前端可渲染的引用列表。

    输入元素须含 content/metadata（kb.search 的返回形态），输出：
    [{"label": 来源标注, "section": 章节, "excerpt": 干净摘录, "score": 分数}]
    给 query 时摘录按命中句截取（居中展示真正相关的片段），否则取头部。
    """
    citations = []
    for doc in results or []:
        meta = doc.get("metadata") or {}
        citations.append({
            "label": source_label(meta),
            "section": _clean(str(meta.get("section", "") or ""), 60),
            "excerpt": _hit_window(doc.get("content", ""), query, excerpt_chars),
            "score": round(float(doc.get("score", 0) or 0), 3),
        })
    return citations
