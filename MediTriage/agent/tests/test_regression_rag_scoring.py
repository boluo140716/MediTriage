"""回归：RAG 分数语义与稀疏路类型过滤（纯单元，不依赖 Milvus/模型）。

守护三件事：
① reranker 不可用的兜底路径分数 = 余弦相似度本身（曾被 1-vdist 反转，
   最优结果反被下游阈值拒掉）；
② BM25 候选取回尊重 filter_type（稀疏路全库打分，不过滤会把科普块
   包装成临床指南返回）；
③ search_knowledge 低相关全军覆没时坦诚弃权，不喂弱命中给 LLM。
"""
import asyncio
import sys
from importlib import import_module
from pathlib import Path

from meditriage.knowledge.langchain_rag import LangChainRAG


_CONTENT_A = (
    "高血压患者的降压目标：一般成人建议控制在140/90mmHg以下，"
    "合并糖尿病或慢性肾脏病者建议130/80mmHg以下，并配合生活方式干预。"
)
_CONTENT_B = (
    "哮喘的阶梯治疗：按症状控制水平逐级调整吸入性糖皮质激素剂量，"
    "并定期评估患者的吸入技术与用药依从性；急性发作时按需使用速效"
    "缓解药物，控制不佳时升级治疗并排查诱因与共病。"
)


def _bare_rag() -> LangChainRAG:
    rag = object.__new__(LangChainRAG)
    rag.use_reranker = False
    rag._reranker = None
    rag.expand_context = False  # 纯单元：不触 Milvus 取兄弟块
    return rag


def test_fallback_score_is_cosine_similarity_not_inverted():
    rag = _bare_rag()
    parsed = [
        {"content": _CONTENT_A, "metadata": {"doc_id": "a"}, "vdist": 0.9},
        {"content": _CONTENT_B, "metadata": {"doc_id": "b"}, "vdist": 0.2},
    ]
    out = rag._rerank_and_format("降压目标", parsed, top_k=2)
    assert out[0]["score"] == 0.9   # 相似度本身，最相关者分数最高
    assert out[1]["score"] == 0.2


def test_bm25_candidate_respects_filter_type():
    rag = object.__new__(LangChainRAG)
    rag._bm25_content = {1: _CONTENT_A, 2: _CONTENT_B}
    rag._bm25_meta = {
        1: {"type": "clinical_guideline", "doc_id": "g1"},
        2: {"type": "health_topic", "doc_id": "h1"},
    }
    # 类型不符的候选被拒
    assert rag._bm25_candidate(2, "clinical_guideline") is None
    c = rag._bm25_candidate(1, "clinical_guideline")
    assert c is not None
    assert c["vdist"] == 0.0  # 无 dense 证据的中性占位（相似度口径，0 为最低）
    # 不带过滤时两者都可取回
    assert rag._bm25_candidate(2) is not None


def _load_search_skill():
    skill_dir = (Path(__file__).resolve().parents[1] / "meditriage" / "skills"
                 / "search-knowledge" / "script")
    sys.path.insert(0, str(skill_dir))
    try:
        return import_module("search")
    finally:
        sys.path.remove(str(skill_dir))


class _FakeKB:
    def __init__(self, results):
        self._results = results

    def search(self, query, top_k, filter_type=None):
        return self._results


def test_search_knowledge_all_weak_hits_abstain(monkeypatch, tmp_path):
    from meditriage.knowledge import badcase
    monkeypatch.setattr(badcase, "LOG_DIR", tmp_path)  # 不污染真实 badcase 日志
    search = _load_search_skill()
    weak = [{"id": "x", "content": "弱相关内容", "metadata": {}, "score": 0.05}]
    monkeypatch.setattr(search, "get_knowledge_base", lambda: _FakeKB(weak))
    out = asyncio.run(search.search_knowledge("罕见病基因治疗"))
    assert out.get("not_found") is True
    assert "知识库未收录" in out["answer"]
    assert "严禁编造" in out["answer"]


def test_search_knowledge_strong_hits_kept(monkeypatch):
    search = _load_search_skill()
    strong = [{
        "id": "esc_htn_2024",
        "content": _CONTENT_A,
        "metadata": {"doc_id": "esc_htn_2024", "source": "ESC",
                     "year": 2024, "section": "BP targets"},
        "score": 0.97,
    }]
    monkeypatch.setattr(search, "get_knowledge_base", lambda: _FakeKB(strong))
    out = asyncio.run(search.search_knowledge("高血压降压目标"))
    assert out["total_found"] == 1
    assert "ESC《血压升高与高血压管理指南》2024" in out["answer"]
    assert out["citations"][0]["label"].startswith("ESC")
