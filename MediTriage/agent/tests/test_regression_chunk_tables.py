"""回归：表格线性化（纯单元，不依赖服务）。

守护：clean_markdown 在压平管道符之前先线性化数据表，保住"药名↔剂量↔靶值"
同行绑定（若先被 _RE_PIPES 压平，表格会散成无结构的词串，丢失行内对应关系）；目录(点导引)
与编委等样板表整块丢弃。
"""
import sys
from importlib import import_module
from pathlib import Path

# clean_chunk.py 在 MediTriage/data/rag_corpus（不在 meditriage 包内），动态加载
_RC = next(
    p / "data" / "rag_corpus"
    for p in Path(__file__).resolve().parents
    if (p / "data" / "rag_corpus" / "clean_chunk.py").is_file()
)
sys.path.insert(0, str(_RC))
clean_markdown = import_module("clean_chunk").clean_markdown


def test_drug_dose_table_keeps_row_binding():
    md = ("| Nitroglycerin* | SL (tablets, spray) | 0.3 or 0.4 mg every 5 min |\n"
          "| Morphine | IV | 2-4 mg; may repeat every 5-15 min |")
    out = clean_markdown(md, lang="en")
    # 同一药的名称/途径/剂量留在同一行（不被打散）
    line = [ln for ln in out.splitlines() if "Nitroglycerin" in ln][0]
    assert "SL" in line and "0.3 or 0.4 mg" in line
    line2 = [ln for ln in out.splitlines() if "Morphine" in ln][0]
    assert "2-4 mg" in line2


def test_header_table_maps_columns():
    md = ("| Drug | Route | Dose |\n|------|-------|------|\n"
          "| Aspirin | Oral | 162-325 mg loading |")
    out = clean_markdown(md, lang="en")
    assert "Drug: Aspirin" in out and "Dose: 162-325 mg loading" in out


def test_toc_table_dropped():
    md = ("| Top 10 Take-Home Messages . . . . . . 1 |\n"
          "| Preamble . . . . . . . . 3 |\n| 1. Introduction . . . . 5 |")
    assert clean_markdown(md, lang="en").strip() == ""


def test_editor_boilerplate_table_dropped():
    md = ("| DEPUTY EDITORS Cheryl A.M. Anderson, PhD |\n"
          "| AD HOC EDITORS |\n| Mark A. Atkinson, PhD |")
    assert clean_markdown(md, lang="en").strip() == ""
