"""生产解析：把 guidelines/*.pdf 解析为干净 Markdown 到 _parsed/<parser>/。

docling 关 OCR、走 CPU（这些 PDF 有文字层）；marker 用于 docling 解析
失败的篇目（GPU）。

用法（需装有 docling / marker 的 Python 环境）：
  CUDA_VISIBLE_DEVICES=-1 python data/rag_corpus/parse_corpus.py --parser docling
  CUDA_VISIBLE_DEVICES=0  python data/rag_corpus/parse_corpus.py --parser marker --only ada_diabetes_2025,esc_htn_2024
"""
import argparse, time
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
ROOT = _paths.DATA_DIR / "rag_corpus"
GUIDE = ROOT / "guidelines"


def extract_docling(pdf):
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
    popts = PdfPipelineOptions()
    popts.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CPU
    )
    popts.do_ocr = False
    conv = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=popts)
        }
    )
    return conv.convert(str(pdf)).document.export_to_markdown()


def extract_marker(pdf):
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    try:
        from marker.config.parser import ConfigParser
        cp = ConfigParser({"disable_ocr": True})
        conv = PdfConverter(
            config=cp.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=cp.get_processors(),
            renderer=cp.get_renderer(),
        )
    except Exception:
        conv = PdfConverter(artifact_dict=create_model_dict())
    txt, _, _ = text_from_rendered(conv(str(pdf)))
    return txt


EXTRACTORS = {"docling": extract_docling, "marker": extract_marker}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parser", required=True, choices=list(EXTRACTORS))
    ap.add_argument(
        "--only", default="", help="逗号分隔的 stem 白名单（不填=全部）"
    )
    ap.add_argument("--force", action="store_true", help="覆盖已存在产出")
    a = ap.parse_args()
    fn = EXTRACTORS[a.parser]
    outdir = ROOT / "_parsed" / a.parser
    outdir.mkdir(parents=True, exist_ok=True)
    only = set(s.strip() for s in a.only.split(",") if s.strip())
    pdfs = sorted(GUIDE.glob("*.pdf"))
    for pdf in pdfs:
        if only and pdf.stem not in only:
            continue
        out = outdir / (pdf.stem + ".md")
        if out.exists() and not a.force:
            print(f"  [skip exists] {pdf.stem}")
            continue
        t0 = time.time()
        try:
            text = fn(pdf)
        except Exception as e:
            print(f"  [FAIL] {a.parser} {pdf.stem}: {type(e).__name__}: {e}")
            continue
        out.write_text(text, encoding="utf-8")
        print(
            f"  [ok] {a.parser} {pdf.stem}: {len(text)} chars in "
            f"{time.time()-t0:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
