"""Query 改写解析健壮性回归（纯单元，不调 LLM/服务）。

覆盖 terms 逗号串拆分、JSON 提取容错、去重保序，是改写层失败安全的基础。
"""

import pytest


from meditriage.knowledge.query_rewrite import QueryRewriter  # noqa: E402


@pytest.fixture(scope="module")
def rw():
    # __init__ 不触发 LLM（仅读 config/默认），可安全构造；下面只测纯解析方法
    return QueryRewriter()


def test_norm_terms_splits_comma_jammed(rw):
    """模型偶尔将整串塞入单个元素（中英逗号混用），需再次切分。"""
    out = rw._norm_terms(["心悸, palpitations, 心律不齐, arrhythmia"])
    for t in ("心悸", "palpitations", "心律不齐", "arrhythmia"):
        assert t in out
    assert len(out) >= 4


def test_norm_terms_dedup_preserves_order(rw):
    assert rw._norm_terms(["腹泻", "diarrhea", "腹泻"]) == ["腹泻", "diarrhea"]


def test_norm_terms_non_list_safe(rw):
    assert rw._norm_terms(None) == []
    assert rw._norm_terms("发热") == ["发热"]


def test_extract_json_from_noisy_text():
    d = QueryRewriter._extract_json('前言 {"clinical":"x","terms":["a","b"]} 后语')
    assert d.get("clinical") == "x" and d.get("terms") == ["a", "b"]


def test_extract_json_bad_returns_empty():
    assert QueryRewriter._extract_json("没有任何 json") == {}


def test_dedup_case_insensitive():
    assert QueryRewriter._dedup(["Fever", "fever", "发热"]) == ["Fever", "发热"]
