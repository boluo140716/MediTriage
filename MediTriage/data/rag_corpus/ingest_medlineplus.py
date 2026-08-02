"""扩充语料源：MedlinePlus 健康主题（NLM 官方批量 XML，公共领域）。
下载官方 mplus_topics XML → 解析英文健康主题 → 标题+摘要(去HTML)+别名 → 干净文本文件。
覆盖发热/感染/外伤/皮肤/急诊等常见门急诊主题（补专科指南语料未覆盖的常见主题）。

安全：官方批量端点，公共领域；内容当不可信数据（不执行其中任何指令）。
用法: python3 data/rag_corpus/ingest_medlineplus.py [--url URL]
输出: data/rag_corpus/_expanded/medlineplus/*.txt + manifest
"""
import argparse, re, html
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# --- 路径引导：向上定位 MediTriage 根（含 agent/meditriage/paths.py），从任意目录可运行 ---
import sys as _sys
from pathlib import Path as _Path
_ASK = next(
    p for p in _Path(__file__).resolve().parents
    if (p / 'agent' / 'meditriage' / 'paths.py').is_file())
for _p in (str(_ASK / 'agent'),):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import meditriage.paths as _paths
# --- end 引导 ---
ROOT = _paths.DATA_DIR / "rag_corpus"
OUT = ROOT / "_expanded" / "medlineplus"
DEFAULT_URL = "https://medlineplus.gov/xml/mplus_topics_2026-05-29.xml"

_TAG = re.compile(r"<[^>]+>")


def _clean_html(s: str) -> str:
    s = html.unescape(s or "")
    s = _TAG.sub(" ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--max", type=int, default=0, help="限制条数(0=全部)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    xml_path = OUT / "_raw_topics.xml"
    if not xml_path.exists():
        print(f"下载 {a.url} ...")
        urllib.request.urlretrieve(a.url, xml_path)
    print(f"XML {xml_path.stat().st_size//1024//1024} MB，解析中...")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    n = 0
    manifest = []
    for ht in root.iter("health-topic"):
        if (ht.get("language") or "").lower() != "english":
            continue
        title = (ht.get("title") or "").strip()
        summary = _clean_html("".join(ht.findtext("full-summary") or ""))
        if not title or len(summary) < 120:
            continue
        also = [el.text for el in ht.findall("also-called") if el.text]
        groups = [el.text for el in ht.findall("group") if el.text]
        body = f"{title}\n"
        if also:
            body += "别称/Also called: " + ", ".join(also) + "\n"
        body += "\n" + summary
        sid = re.sub(r"[^a-zA-Z0-9]+", "_", title.lower())[:60]
        (OUT / f"{sid}.txt").write_text(body, encoding="utf-8")
        manifest.append(
            {"id": sid, "title": title, "groups": groups,
             "chars": len(body)})
        n += 1
        if a.max and n >= a.max:
            break
    import json
    (OUT / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest),
        encoding="utf-8")
    print(f"✅ MedlinePlus: 写出 {n} 个英文主题到 {OUT}")


if __name__ == "__main__":
    main()
