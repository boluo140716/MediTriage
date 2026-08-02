"""RAG 解析器赛马 — 第 1 步：用指定解析器抽取测试 PDF，产出到 _bakeoff/<parser>/<stem>.md。

同样几份文件喂给不同解析器，第 2 步 rag_parse_score.py 再统一打分
（启发式 + LLM 连贯性）。

用法（用解析器 venv 的 python）：
  ~/rag_parsers_venv/bin/python MediTriage/diag/rag_parse_bakeoff.py --parser docling
  ~/rag_parsers_venv/bin/python MediTriage/diag/rag_parse_bakeoff.py --parser marker
  python3 MediTriage/diag/rag_parse_bakeoff.py --parser pdfplumber   # 基线(容器/宿主皆可)
"""
import argparse
import time
from pathlib import Path

# 路径引导：向上定位 MediTriage 根（含 agent/meditriage/paths.py），从任意目录可运行。
import sys as _sys
from pathlib import Path as _Path
_ASK = next(
    p for p in _Path(__file__).resolve().parents
    if (p / 'agent' / 'meditriage' / 'paths.py').is_file()
)
for _p in (str(_ASK / 'agent'),):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import meditriage.paths as _paths
GUIDE = _paths.DATA_DIR / "rag_corpus/guidelines"
OUT = _paths.DATA_DIR / "rag_corpus/_bakeoff"

# 赛马样本：最糟的双栏英文 PDF（房颤、糖尿病）+ 一个高分对照（COPD）。
TEST_FILES = [
    "esc_afib_2024_essential.pdf",
    "ada_diabetes_2025.pdf",
    "gold_copd_2025.pdf",
]


def extract_pdfplumber(pdf: Path) -> str:
    import pdfplumber
    with pdfplumber.open(str(pdf)) as pf:
        return "\n".join((pg.extract_text() or "") for pg in pf.pages)


def extract_docling(pdf: Path) -> str:
    import os
    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption,
    )
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        AcceleratorOptions,
        AcceleratorDevice,
    )
    # 默认 CPU，绕开 cuda/cpu 张量混放 bug。
    dev = os.environ.get("DOCLING_DEVICE", "cpu").lower()
    acc = AcceleratorDevice.CUDA if dev == "cuda" else AcceleratorDevice.CPU
    popts = PdfPipelineOptions()
    popts.accelerator_options = AcceleratorOptions(device=acc)
    # 公平协议：这些 PDF 有文字层，关 OCR，只用文字层 + 版面模型，更快更干净。
    popts.do_ocr = False
    conv = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=popts)
        }
    )
    res = conv.convert(str(pdf))
    return res.document.export_to_markdown()


def extract_marker(pdf: Path) -> str:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    # 公平协议：关 OCR，只用文字层 + 版面模型（与 docling 对齐）。
    try:
        from marker.config.parser import ConfigParser
        cp = ConfigParser({"disable_ocr": True})
        converter = PdfConverter(
            config=cp.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=cp.get_processors(),
            renderer=cp.get_renderer(),
        )
    except Exception:
        # 退化为默认配置，仍优先文字层。
        converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(str(pdf))
    txt, _, _ = text_from_rendered(rendered)
    return txt


def extract_mineru(pdf: Path) -> str:
    # MinerU(magic-pdf) 新版 API；失败则提示用 CLI。
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    data = PymuDocDataset(pdf.read_bytes())
    infer = data.apply(doc_analyze, ocr=False)
    pipe = infer.pipe_txt_mode(None)
    return pipe.get_markdown(None)


EXTRACTORS = {
    "pdfplumber": extract_pdfplumber,
    "docling": extract_docling,
    "marker": extract_marker,
    "mineru": extract_mineru,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parser", required=True, choices=list(EXTRACTORS))
    a = ap.parse_args()
    fn = EXTRACTORS[a.parser]
    outdir = OUT / a.parser
    outdir.mkdir(parents=True, exist_ok=True)
    for name in TEST_FILES:
        pdf = GUIDE / name
        if not pdf.exists():
            print(f"  [missing] {name}")
            continue
        t0 = time.time()
        try:
            text = fn(pdf)
        except Exception as e:
            print(f"  [FAIL] {a.parser} {name}: {type(e).__name__}: {e}")
            continue
        out = outdir / (pdf.stem + ".md")
        out.write_text(text, encoding="utf-8")
        print(
            f"  [ok] {a.parser} {name}: {len(text)} chars in "
            f"{time.time()-t0:.1f}s -> {out}"
        )


if __name__ == "__main__":
    main()
