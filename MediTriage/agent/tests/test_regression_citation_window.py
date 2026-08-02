"""回归：citations 命中句截取（纯单元）。

守护：给 query 时摘录定位最相关句居中截取，而非恒取 chunk 头部；query 为空
或无词重叠时退回头部截断（向后兼容）。
"""
from meditriage.knowledge.citations import build_citations, _hit_window


_LONG = (
    "本段开头是一些无关的导言文字，介绍背景与方法学，篇幅较长用于占位。"
    "正文中段才出现关键结论：二甲双胍是2型糖尿病的一线降糖药物，"
    "推荐糖化血红蛋白控制目标为低于7%。"
    "随后是大量参考说明与免责声明，继续占位拉长这一整块内容到足够长度。"
) * 2


def test_hit_window_centers_on_relevant_sentence():
    out = _hit_window(_LONG, "二甲双胍 一线 糖化血红蛋白 目标", limit=120)
    assert "二甲双胍" in out and "一线" in out
    assert len(out) <= 130


def test_hit_window_falls_back_to_head_without_query():
    out = _hit_window(_LONG, "", limit=80)
    assert out.startswith("本段开头")


def test_build_citations_uses_query_for_excerpt():
    results = [{"content": _LONG, "metadata": {"doc_id": "ada_diabetes_2025"}}]
    cites = build_citations(results, query="二甲双胍 糖化血红蛋白目标")
    assert "二甲双胍" in cites[0]["excerpt"]
    # 头部截断则不含命中要点
    head = build_citations(results)  # 无 query
    assert head[0]["excerpt"].startswith("本段开头")
