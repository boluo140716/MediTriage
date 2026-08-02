"""回归：recommend_lifestyle 确定性处方表（离线，不依赖 vLLM/Milvus）。

守护：覆盖病名/别名路由到正确条目；未收录的病返回明确提示而非无关泛化文案
（旧版会把“多喝水”等通用建议套上“【X生活方式建议】”返回，此处防其回归）。
"""
import asyncio
import importlib.util
from pathlib import Path

import pytest

_PY = (
    Path(__file__).resolve().parent.parent
    / "meditriage/skills/recommend-lifestyle/script/lifestyle.py"
)
_spec = importlib.util.spec_from_file_location("recommend_lifestyle_skill", _PY)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_find = _mod._find
recommend_lifestyle = _mod.recommend_lifestyle


@pytest.mark.parametrize("query,expected_key", [
    ("高血压", "高血压"),
    ("血压高", "高血压"),
    ("2型糖尿病", "糖尿病"),
    ("糖尿病", "糖尿病"),
    ("冠心病", "血脂异常与动脉粥样硬化性心血管病"),
    ("高血脂", "血脂异常与动脉粥样硬化性心血管病"),
    ("慢阻肺", "慢性阻塞性肺病"),
    ("CKD", "慢性肾病"),
    ("感冒", "感冒"),
])
def test_lookup_routes_to_entry(query, expected_key):
    e = _find(query)
    assert e is not None, f"{query!r} 应命中"
    assert e["key"] == expected_key


@pytest.mark.parametrize("query", ["痛风", "失眠", "牙疼", "一种不存在的病", ""])
def test_uncovered_is_honest_miss(query):
    """未收录的病必须明确未收录，绝不返回无关泛化文案。"""
    assert _find(query) is None
    r = asyncio.run(recommend_lifestyle(query))
    assert r["source"] == "未收录"
    assert r["categories"] == []
    assert "未收录" in r["answer"]


def test_hit_contract_has_sections_and_source():
    r = asyncio.run(recommend_lifestyle("高血压"))
    assert r["source"].startswith("本地整理")
    assert "饮食" in r["categories"] and "运动" in r["categories"]
    assert "【高血压·生活方式建议】" in r["answer"]
