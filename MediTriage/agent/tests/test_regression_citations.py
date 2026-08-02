"""回归：RAG 引用可读化（纯单元，不依赖 Milvus/vLLM）。

守护三件事：
① source_label 用索引真实 metadata 字段组装可读来源（不再取不存在的
   disease/organization 字段产出 "N/A"）；
② build_citations 产出前端可渲染的 label/section/excerpt/score；
③ skill 格式化输出携带来源标注，不再裸贴 chunk。
"""
from meditriage.knowledge.citations import build_citations, source_label


_GUIDELINE_META = {
    "source": "ESC", "topic": "hypertension", "year": 2024, "lang": "en",
    "type": "clinical_guideline", "doc_id": "esc_htn_2024",
    "section": "2024 ESC Guidelines", "part": 0,
}


def test_source_label_known_doc():
    assert source_label(_GUIDELINE_META) == "ESC《血压升高与高血压管理指南》2024"


def test_source_label_medlineplus():
    label = source_label({"doc_id": "medlineplus:abdominal_pain",
                          "source": "medlineplus"})
    assert "MedlinePlus" in label and "abdominal pain" in label


def test_source_label_unknown_falls_back_without_fabrication():
    label = source_label({"doc_id": "some_new_doc", "source": "ORG",
                          "year": 2030})
    assert label == "ORG 2030 some_new_doc"  # 机构+年份+doc_id，不编造标题


def test_build_citations_shape_and_cleaning():
    results = [{
        "content": "**Blood pressure**  targets\n\nshould be <130/80 mmHg " * 10,
        "metadata": _GUIDELINE_META,
        "score": 0.8765,
    }]
    cites = build_citations(results, excerpt_chars=80)
    assert len(cites) == 1
    c = cites[0]
    assert c["label"].startswith("ESC")
    assert "**" not in c["excerpt"] and "\n" not in c["excerpt"]
    assert len(c["excerpt"]) <= 81  # 80 + 省略号
    assert c["score"] == round(0.8765, 3)  # 保留 3 位（浮点表示下为 0.876）


def test_search_format_results_carries_source():
    from importlib import import_module
    import sys
    from pathlib import Path
    skill_dir = (Path(__file__).resolve().parents[1] / "meditriage" / "skills"
                 / "search-knowledge" / "script")
    sys.path.insert(0, str(skill_dir))
    try:
        search = import_module("search")
        text = search.format_results([{
            "source": "ESC《血压升高与高血压管理指南》2024",
            "section": "BP targets",
            "content": "血压目标 <130/80 mmHg",
            "score": 0.9,
        }])
    finally:
        sys.path.remove(str(skill_dir))
    assert "来源：ESC《血压升高与高血压管理指南》2024" in text
    assert "BP targets" in text


def test_guideline_format_uses_real_fields():
    from importlib import import_module
    import sys
    from pathlib import Path
    skill_dir = (Path(__file__).resolve().parents[1] / "meditriage" / "skills"
                 / "clinical-guideline" / "script")
    sys.path.insert(0, str(skill_dir))
    try:
        g = import_module("guideline")
        text = g.format_guideline("内容正文", _GUIDELINE_META)
    finally:
        sys.path.remove(str(skill_dir))
    assert "ESC《血压升高与高血压管理指南》2024" in text
    assert "发布机构：ESC" in text
    assert "N/A相关临床指南" not in text
