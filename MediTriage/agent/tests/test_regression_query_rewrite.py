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


def test_simple_query_short_circuit():
    """简短查询（<=12 字符且空格少）跳过 LLM 改写。"""
    assert QueryRewriter._is_simple_query("腰疼怎么办")
    assert QueryRewriter._is_simple_query("眼睛干涩")
    assert QueryRewriter._is_simple_query("")
    # 长/复杂查询保留改写（多路召回增益值得一次 LLM 调用）
    assert not QueryRewriter._is_simple_query(
        "眼睛干涩酸胀不适持续一周 常见原因 干眼症 视疲劳")
    assert not QueryRewriter._is_simple_query(
        "low back pain with radiating leg pain")


def test_rewrite_skips_llm_for_simple_query(monkeypatch):
    """短查询 rewrite() 不触发 LLM，直接返回原 query。"""
    rw = QueryRewriter()
    called = {"n": 0}

    def boom(query):
        called["n"] += 1
        raise AssertionError("短查询不应调用 LLM 改写")

    monkeypatch.setattr(rw, "_call_llm", boom)
    assert rw.rewrite("腰疼怎么办") == ["腰疼怎么办"]
    assert called["n"] == 0


def test_default_timeout_reduced_to_4s():
    """rewrite 超时默认 4s（失败快速失败，不再白等 8s）。"""
    assert QueryRewriter().timeout == 4.0


def test_circuit_breaker_skips_llm_after_failures(monkeypatch):
    """连续失败达到阈值（默认 2 次）-> 熔断，后续不再调 LLM。"""
    rw = QueryRewriter(circuit_breaker_threshold=2)
    calls = {"n": 0}

    def boom(query):
        calls["n"] += 1
        raise RuntimeError("api down")

    monkeypatch.setattr(rw, "_call_llm", boom)
    q1 = "这是一个比较长的查询需要改写一下"
    q2 = "这是另一个比较长的查询需要改写一下"
    q3 = "这是第三个比较长的查询需要改写一下"
    assert rw.rewrite(q1) == [q1]   # 失败 1
    assert rw.rewrite(q2) == [q2]   # 失败 2
    assert calls["n"] == 2
    assert rw.rewrite(q3) == [q3]   # 熔断：不调 LLM
    assert calls["n"] == 2          # 未增加


def test_circuit_breaker_resets_on_success(monkeypatch):
    """未达阈值前成功一次即重置熔断计数；之后重新累计。"""
    rw = QueryRewriter(circuit_breaker_threshold=2)
    calls = {"n": 0}

    def flaky(query):
        calls["n"] += 1
        if calls["n"] == 2:
            return {"clinical": "临床规范表述", "english": "clinical term", "terms": []}
        raise RuntimeError("api down")

    monkeypatch.setattr(rw, "_call_llm", flaky)
    qs = [f"第{i}个比较长的查询需要改写一下" for i in range(5)]
    rw.rewrite(qs[0])          # 失败 1（failures=1）
    r = rw.rewrite(qs[1])      # 成功 -> 重置（calls=2, failures=0）
    assert calls["n"] == 2
    assert "临床规范表述" in r   # 改写成功带出变体
    rw.rewrite(qs[2])          # 失败 1
    rw.rewrite(qs[3])          # 失败 2（failures=2）
    assert calls["n"] == 4
    rw.rewrite(qs[4])          # 熔断：不调 LLM
    assert calls["n"] == 4
