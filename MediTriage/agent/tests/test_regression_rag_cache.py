"""回归：RAG 检索结果 LRU 缓存。

守护：
① 相同 (query, top_k, filter, rewrite) 第二次命中缓存，不重复检索；
② 缓存返回值是浅拷贝，调用方修改不污染缓存；
③ LRU 超容量淘汰最旧条目；
④ cache key 区分 filter_type 与 rewrite 开关，避免串缓存。
"""
import threading
from collections import OrderedDict

from meditriage.knowledge.langchain_rag import LangChainRAG


def _make_rag(cache_size=3):
    """绕过 __init__（会连 Milvus/在线 embed），只装配缓存字段。"""
    rag = LangChainRAG.__new__(LangChainRAG)
    rag._search_cache = OrderedDict()
    rag._search_cache_size = cache_size
    rag._search_cache_lock = threading.Lock()
    return rag


def test_cache_put_get_hit():
    rag = _make_rag()
    key = ("腰疼", 3, "", "default")
    rag._cache_put(key, [{"id": 1, "content": "c", "metadata": {}, "score": 0.9}])
    hit = rag._cache_get(key)
    assert hit is not None and hit[0]["id"] == 1


def test_cache_returns_copy_not_reference():
    rag = _make_rag()
    key = ("腰疼", 3, "", "default")
    val = [{"id": 1, "content": "c", "metadata": {}, "score": 0.9}]
    rag._cache_put(key, val)
    got = rag._cache_get(key)
    got[0]["id"] = 999  # 修改返回值
    assert rag._search_cache[key][0]["id"] == 1  # 缓存未被污染


def test_cache_lru_evicts_oldest():
    rag = _make_rag(cache_size=2)
    for i in range(3):
        rag._cache_put(("q%d" % i, 3, "", "default"), [{"id": i}])
    assert rag._cache_get(("q0", 3, "", "default")) is None  # 最旧被淘汰
    assert rag._cache_get(("q1", 3, "", "default")) is not None
    assert rag._cache_get(("q2", 3, "", "default")) is not None


def test_cache_key_distinguishes_filter_and_rewrite():
    rag = _make_rag()
    k1 = rag._cache_key("腰疼", 3, None, None)
    k2 = rag._cache_key("腰疼", 3, "guideline", None)
    k3 = rag._cache_key("腰疼", 3, None, True)
    k4 = rag._cache_key("腰疼", 3, None, False)
    assert len({k1, k2, k3, k4}) == 4


def test_search_hits_cache(monkeypatch):
    rag = _make_rag()
    calls = {"n": 0}

    def fake_impl(query, top_k=3, filter_type=None, rewrite=None):
        calls["n"] += 1
        return [{"id": 1, "content": "c", "metadata": {}, "score": 0.9}]

    monkeypatch.setattr(rag, "_search_impl", fake_impl)
    r1 = rag.search("腰疼", top_k=3)
    r2 = rag.search("腰疼", top_k=3)
    assert calls["n"] == 1       # 第二次命中缓存
    assert r1 == r2              # 结果一致


def test_search_cache_misses_on_different_params(monkeypatch):
    rag = _make_rag()
    calls = {"n": 0}

    def fake_impl(query, top_k=3, filter_type=None, rewrite=None):
        calls["n"] += 1
        return [{"id": calls["n"]}]

    monkeypatch.setattr(rag, "_search_impl", fake_impl)
    rag.search("腰疼", top_k=3)
    rag.search("腰疼", top_k=5)          # top_k 不同
    rag.search("腰疼", top_k=3, filter_type="guideline")  # filter 不同
    assert calls["n"] == 3
