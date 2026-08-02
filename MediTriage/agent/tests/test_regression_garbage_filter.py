"""回归：RAG 垃圾块过滤（纯单元，不依赖 Milvus/模型）。

守护两层防线：
① _is_garbage_chunk 识别期刊 front-matter（刊头/编委名单/ISSN），入库即拦；
② _rerank_and_format 检索出口兜底——旧库存量垃圾块不返回给调用方。
"""
from meditriage.knowledge.langchain_rag import LangChainRAG, _is_garbage_chunk


_FRONT_MATTER_SAMPLES = [
    "【**DEPUTY EDITORS**】\nCheryl A.M. Anderson, PhD, MPH, MS\n"
    "John B. Buse, MD, PhD\nElizabeth Selvin, PhD, MPH 以及更多编辑成员名单"
    "（此处补足长度避免触发短文本规则）",
    "【EDITORIAL BOARD】 David Aguilar, MD / Anastasia-Stefania "
    "Alexopoulos, MBBS, MHSc Mohammed 等几十位编辑委员会成员的完整名单列表",
    "【THE JOURNAL OF CLINICAL AND APPLIED RESEARCH AND EDUCATION】"
    "The American Diabetes Association publishes this journal for members.",
]

_REAL_CONTENT = (
    "高血压患者的降压目标：一般成人建议血压控制在 140/90 mmHg 以下，"
    "合并糖尿病或慢性肾脏病者建议更严格的 130/80 mmHg 以下，"
    "并结合生活方式干预与规律随访。"
)


def test_front_matter_detected_as_garbage():
    for s in _FRONT_MATTER_SAMPLES:
        assert _is_garbage_chunk(s), s[:50]


def test_real_clinical_content_not_garbage():
    assert not _is_garbage_chunk(_REAL_CONTENT)


def _bare_rag() -> LangChainRAG:
    """不触发 __init__（不连 Milvus / 不加载模型）的裸实例。"""
    rag = object.__new__(LangChainRAG)
    rag.use_reranker = False
    rag._reranker = None
    rag.expand_context = False  # 纯单元：不触 Milvus 取兄弟块
    return rag


def test_search_exit_filters_stored_garbage():
    rag = _bare_rag()
    parsed = [
        {"content": _FRONT_MATTER_SAMPLES[0],
         "metadata": {"doc_id": "ada_diabetes_2025"}, "vdist": 0.1},
        {"content": _REAL_CONTENT,
         "metadata": {"doc_id": "esc_htn_2024"}, "vdist": 0.3},
    ]
    out = rag._rerank_and_format("高血压 降压目标", parsed, top_k=2)
    assert len(out) == 1
    assert out[0]["id"] == "esc_htn_2024"


def test_search_exit_keeps_results_if_all_filtered():
    """极端：候选全是垃圾时不至于空手而归（保持原样返回，宁可有据可查）。"""
    rag = _bare_rag()
    parsed = [
        {"content": _FRONT_MATTER_SAMPLES[1],
         "metadata": {"doc_id": "ada_diabetes_2025"}, "vdist": 0.2},
    ]
    out = rag._rerank_and_format("编辑委员会", parsed, top_k=1)
    assert len(out) == 1
