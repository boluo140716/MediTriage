"""回归：disease_code ICD-10 确定性查表（离线，不依赖 vLLM/Milvus）。

守护两点：
1. 标准/口语病名命中正确编码（口语归一、去尾、子串）。
2. 无关/不存在的查询返回“未收录”，绝不返回最近邻错码——旧版 RAG filter 会对
   覆盖外的病返回最近的 chunk（如“哮喘”返脑膜炎编码），此处防其回归。
"""
import asyncio
import importlib.util
from pathlib import Path

import pytest

# skills/disease-code/script/code.py（与 stdlib `code` 同名，用 importlib 唯一命名加载）
_CODE_PY = (
    Path(__file__).resolve().parent.parent
    / "meditriage/skills/disease-code/script/code.py"
)
_spec = importlib.util.spec_from_file_location("disease_code_skill", _CODE_PY)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_match = _mod._match
disease_code = _mod.disease_code


@pytest.mark.parametrize("query,expected_top", [
    ("哮喘", "J45"),
    ("痛风", "M10"),
    ("缺铁性贫血", "D50"),
    ("2型糖尿病", "E11"),   # 口语归一 → 非胰岛素依赖型糖尿病
    ("1型糖尿病", "E10"),
    ("乙肝", "B16"),        # 口语归一 → 乙型肝炎
    ("慢阻肺", "J44"),      # 口语归一 → 慢性阻塞性肺
    ("高血压病", "I10"),    # 去尾“病” → 高血压
])
def test_lookup_hits_expected_code(query, expected_top):
    matches = _match(query)
    assert matches, f"{query!r} 应命中"
    assert matches[0]["code"] == expected_top


@pytest.mark.parametrize("query,expected_member", [
    ("高血压", "I10"),
    ("冠心病", "I25"),   # 口语归一 → 缺血性心脏病（I24/I25）
    ("糖尿病", "E11"),
    ("肺结核", "A16"),   # 口语归一 → 呼吸道结核（A15/A16；压测实测曾未命中）
    ("肺痨", "A16"),
])
def test_lookup_contains_expected_code(query, expected_member):
    codes = [m["code"] for m in _match(query)]
    assert expected_member in codes, f"{query!r} 应含 {expected_member}，实得 {codes}"


@pytest.mark.parametrize("query", ["外星综合征", "打喷嚏", "一种不存在的病", ""])
def test_no_false_nearest_neighbor(query):
    """无关/不存在的查询必须“未收录”，绝不返回最近邻错码。"""
    assert _match(query) == []


def test_miss_contract():
    r = asyncio.run(disease_code("外星综合征"))
    assert r["icd10_code"] == ""
    assert r["matches"] == []
    assert "未" in r["answer"]


def test_hit_contract():
    r = asyncio.run(disease_code("痛风"))
    assert r["icd10_code"] == "M10"
    assert any(m["code"] == "M10" for m in r["matches"])
