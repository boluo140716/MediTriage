"""RAG 清洗 + 结构化分块（基于 docling Markdown 输出）。

docling 正文解析准确，但会忠实抽出大量非正文内容——目录、关键词行、刊头、
广告、编辑名单、版权下载声明、修订记录、参考文献列表，这些是检索证据噪声的
主要来源。本模块分三步处理：A) 段落级去样板；B) 行内清洗；C) 按 Markdown
结构分块（以标题/表格为单元，附标题路径前缀）。

用法：
  clean_markdown(md, lang="en") -> 干净正文 markdown
  chunk_markdown(md, source_meta) -> [{"content","metadata"}]  结构化块
  CLI: python3 clean_chunk.py --clean-bakeoff
       清洗 _bakeoff/docling/*.md -> _bakeoff/docling_cleaned/
"""
import re
from pathlib import Path


def zh_type_from_stem(stem: str) -> str:
    """按文件名前缀编号映射中文本地文档的 type（对齐 skill 的 filter_type）。

    01–09 映射 lifestyle，10–19 映射 disease_classification(icd10)，
    20–29 映射 clinical_guideline，其余映射 local。
    """
    num = stem.split("_")[0]
    if not num.isdigit():
        return "local"
    n = int(num)
    return (
        "lifestyle" if n < 10
        else "disease_classification" if n < 20
        else "clinical_guideline" if n < 30
        else "local"
    )


# ---------- A. 段落级去样板（整行/整段丢弃）----------
# 每条命中即丢弃该行，按证据来源归类。
_LINE_JUNK = [
    # 目录点导引行  "... Preamble ........ 12"
    re.compile(r"\.{5,}\s*\d*\s*$"),
    # 关键词行
    re.compile(r"^\s*Keywords\b", re.I),
    # 含邮箱（刊头/广告联系方式）
    re.compile(r"\S+@\S+\.\w+"),
    re.compile(r"\b(ADVERTISING|PEER REVIEW|PRODUCTION MANAGER|Account Manager|EDITOR|eHealthcare|DIGITAL PRODUCTION)\b"),
    re.compile(r"Downloaded\s+from|permissions?@|All rights reserved|©|doi\.org|academic\.oup\.com|https?://", re.I),
    # 修订记录
    re.compile(r"\b(has|have) been (updated|added|revised|included|removed|changed|expanded|moved|reorganized)\b", re.I),
    # 修订记录（单数动词）
    re.compile(r"\b(was|were) (revised|updated|added|removed|renamed|reworded|reorganized|expanded)\b", re.I),
    # 修订摘要条目
    re.compile(r"^\s*Recommendation\s+\d+\.\d+\b.*\b(was|were|has|have|revised|updated|added|new)\b", re.I),
    re.compile(r"\bis revised annually\b|\bSummary of Revisions\b", re.I),
    # 图表清单条目
    re.compile(r"^\s*(Figure|Table)\s+\d+\.\d+\s+['‘\"].*['’\"]", re.I),
    # 期号/日期页眉
    re.compile(r"^\s*(Volume|Supplement|January|February|March|April|May|June|July|August|September|October|November|December)\s+\d", re.I),
    # 单独的增刊页码 "S27"
    re.compile(r"^\s*S\d{1,3}\s*$"),
    # docling 图片占位
    re.compile(r"^\s*<!--\s*image\s*-->\s*$"),
    # 纯分隔/空行噪声
    re.compile(r"^\s*[·•|\-–—\s]*$"),
]
# 段落块级：从某标题起整段丢弃，直到下一个标题。标题可带编号
# （如 "18. References"），故在整行标题文本里搜关键词，而非锚定 # 之后。
_SECTION_DROP = re.compile(
    r"^#{1,6}\s.*\b(References|Bibliography|Acknowledg\w*|Summary of Revisions|Author Information|"
    r"Conflict\w* of Interest|Fundi\w*|Disclosure\w*|Abbreviations|Appendix|Supplementary"
    r"|Writing Committee|Table of Contents|Key ?Words|Correspondence|Peer Review Committee)\b", re.I)

# ---------- B. 行内清洗 ----------
# 断词：ben-\nefits -> benefits
_RE_HYPHEN = re.compile(r"([A-Za-z])-\n([a-z])")
# 句末引用上标 word.554,557–559
_RE_REFSUP = re.compile(r"(?<=[a-z])\.\d{1,3}(?:[,–-]\d{1,3})*(?=\s|$)")
# 表格管道符+空单元格 "| | | |"
_RE_PIPES = re.compile(r"(\s*\|\s*)+")
# 表格分隔线/横线 "------"
_RE_RULES = re.compile(r"[-_]{4,}")
_RE_MULTISPACE = re.compile(r"[ \t]{2,}")

# ---------- B0. 表格线性化（在压平管道符之前，保住行内绑定）----------
_TBL_ROW = re.compile(r"^\s*\|.*\|?\s*$")            # markdown 表格行
_TBL_SEP = re.compile(r"^\s*\|?[\s:|-]*-{3,}[\s:|-]*\|?\s*$")  # 分隔行 |---|---|
_DOTLEAD = re.compile(r"\.\s*\.\s*\.")              # 目录点导引
# 表格里的样板（编委/期刊刊头）整块丢弃
_TBL_BOILER = re.compile(
    r"\b(DEPUTY EDITORS|EDITOR[\s-]?IN[\s-]?CHIEF|ASSOCIATE EDITORS|AD HOC"
    r"|EDITORIAL BOARD|ISSN|Writing Committee)\b", re.I)


def _split_cells(row: str):
    """拆一行表格的单元格：去首尾管道符、按 | 切、strip、丢空。"""
    return [c.strip() for c in row.strip().strip("|").split("|") if c.strip()]


def _looks_like_header(cells):
    """首行像表头：单元格短、无句末标点（区别于数据/正文行）。"""
    return bool(cells) and all(
        len(c) <= 28 and not re.search(r"[。.!?；;]\s*$", c) for c in cells)


def _linearize_tables(md: str) -> str:
    """把 Markdown 表格块按行线性化为 "列头: 单元格" 句，保住药↔剂量↔靶值同行绑定。

    在 _strip_inline 压平管道符之前做。目录(点导引)/编委等样板表整块丢弃；有表头则
    映射为 "h1: c1 · h2: c2"，无表头则单元格以 " — " 连接；单列表退化为纯文本。
    解析异常的行回退原样（交给后续 _RE_PIPES 压平），绝不抛出。
    """
    lines = md.splitlines()
    out, i, n = [], 0, len(lines)
    while i < n:
        if not _TBL_ROW.match(lines[i]) or "|" not in lines[i]:
            out.append(lines[i]); i += 1
            continue
        j = i
        while j < n and _TBL_ROW.match(lines[j]) and "|" in lines[j]:
            j += 1
        block = lines[i:j]
        i = j
        # 目录/样板表：整块丢弃
        if any(_DOTLEAD.search(r) for r in block[:3]) or \
           any(_TBL_BOILER.search(r) for r in block[:4]):
            continue
        rows = [_split_cells(r) for r in block if not _TBL_SEP.match(r)]
        rows = [r for r in rows if r]
        if not rows:
            continue
        header = rows[0] if _looks_like_header(rows[0]) and len(rows) > 1 else None
        data = rows[1:] if header else rows
        for cells in data:
            if header and len(cells) == len(header):
                out.append("; ".join(f"{h}: {c}" for h, c in zip(header, cells)))
            elif len(cells) == 1:
                out.append(cells[0])                 # 单列：纯文本
            else:
                out.append(" — ".join(cells))        # 无表头：单元格连接
    return "\n".join(out)


def _strip_inline(text: str) -> str:
    text = _RE_HYPHEN.sub(r"\1\2", text)  # 先合断词
    text = _RE_REFSUP.sub(".", text)  # 句末引用号 -> 仅留句号
    # 压平 markdown 表格管道符/空单元格，保留正文可读
    text = _RE_PIPES.sub(" ", text)
    # 压平表格分隔线/横线（避免一堵 "----" 墙）
    text = _RE_RULES.sub(" ", text)
    text = _RE_MULTISPACE.sub(" ", text)
    return text


def clean_markdown(md: str, lang: str = "en") -> str:
    """去样板 + 行内清洗，返回干净正文 markdown。中文 txt 本就纯正文，仅做轻量行内。"""
    import html
    md = html.unescape(md)  # 恢复 &lt;/&gt;/&amp;/≤/≥（"&lt; 60 mL/min" 破坏阈值匹配）
    md = _linearize_tables(md)  # 先线性化表格（保住行内绑定），再压平残余管道符
    md = _strip_inline(md)
    out, drop_section = [], False
    for line in md.splitlines():
        is_header = bool(re.match(r"^#{1,6}\s", line))
        if is_header:
            drop_section = bool(_SECTION_DROP.match(line))  # 进入/退出可丢段
            if drop_section:
                continue
        if drop_section:
            continue
        if lang == "en" and any(p.search(line) for p in _LINE_JUNK):
            continue
        out.append(line)
    # 合并多余空行
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return cleaned


# ---------- C. 结构化分块 ----------
_HEADER = re.compile(r"^(#{1,6})\s+(.*)$")
# 标题编号前缀（容忍 markdown 粗体星号与 "S13" 增刊页码前缀）："7.4.3" -> 层级 3
_NUM_PREFIX = re.compile(r"^\**\s*(?:S\d+\s+)?(\d+(?:\.\d+)+|\d+)[.\)\-]?\s")
# 样板首标题黑名单：docling 常把这些非正文大标题排在文首，若不过滤会被钉成
# 所有块的固定祖先（"Revision Process > ..." 殃及全本）。命中则不作祖先。
_CRUMB_BOILER = re.compile(
    r"^(revision process|clinical practice guidelines?|circulation"
    r"|contents|table of contents|writing committee|guidelines?)\b", re.I)


def _num_level(title: str):
    """从标题编号前缀推断层级（7.4.3 -> 3，'1.' -> 1）；无编号返回 None。"""
    m = _NUM_PREFIX.match(title)
    return (m.group(1).count(".") + 1) if m else None


def _split_blocks(md: str):
    """按 Markdown 标题切成 (header_path, body) 段；表格作为整体保留在 body 内。

    层级以标题编号前缀为准（docling 把全部标题压成同级 H2，原 markdown # 数失真）；
    无编号标题视为顶层小节、不继承上文祖先，避免"样板首标题 > 当前节"的污染。
    """
    sections, cur_body, stack = [], [], []
    for line in md.splitlines():
        m = _HEADER.match(line)
        # 超长"标题"实为被误判的表格/正文，降级为正文
        if m and len(m.group(2).strip()) > 150:
            cur_body.append(m.group(2).strip())
            m = None
        if m:
            if cur_body:
                sections.append((" > ".join(stack), "\n".join(cur_body).strip()))
                cur_body = []
            title = m.group(2).strip()
            nl = _num_level(title)
            if nl:
                stack = stack[:nl - 1] + [title]       # 按编号建真层级
            else:
                stack = [title]                         # 无编号=顶层，不带祖先
        else:
            cur_body.append(line)
    if cur_body:
        sections.append((" > ".join(stack), "\n".join(cur_body).strip()))
    return [(h, b) for h, b in sections if b]


_ALNUM_C = re.compile(r"[A-Za-z0-9一-鿿]")


def _short_crumb(header_path: str) -> str:
    """精简标题路径前缀：丢弃超长大标题与样板祖先，只留最具体的 1-2 级小节，截断封顶。"""
    segs = [s.strip() for s in header_path.split(" > ") if s.strip()]
    segs = [s for s in segs if len(s) <= 70 and not _CRUMB_BOILER.match(s)]
    if not segs:
        return ""
    crumb = " > ".join(segs[-2:])
    return crumb[:120]  # 总长封顶，防止异常长标题混入


def _chunk_is_junk(text: str, min_len: int) -> bool:
    """块级垃圾判定：过短 / 字母数字占比过低（残留表格符号、双栏碎片、纯编号）。"""
    t = text.strip()
    if len(t) < min_len:
        return True
    alnum = len(_ALNUM_C.findall(t))
    if alnum / max(len(t), 1) < 0.55:
        return True
    # 实词太少（多为编号/符号碎片）
    words = re.findall(r"[A-Za-z一-鿿]{3,}", t)
    return len(words) < 8


def _sentences(body: str, hard: int = 1400):
    sents = [s for s in re.split(r"(?<=[。！？.!?])\s+|\n{2,}", body) if s.strip()]
    # 无句号的超长段（如压平后的表格）硬切，避免单"句"撑出巨块
    out = []
    for s in sents:
        if len(s) <= hard:
            out.append(s)
        else:
            out.extend(s[i:i + hard] for i in range(0, len(s), hard))
    return out


def _tail_overlap(buf, overlap_chars):
    """取 buf 末尾若干整句作重叠，总长 ≤ overlap_chars（避免把超长'句'整段带走）。"""
    ov, n = [], 0
    for s in reversed(buf):
        if n + len(s) > overlap_chars:
            break
        ov.insert(0, s)
        n += len(s)
    return ov


def chunk_markdown(md: str, source_meta: dict, chunk_size: int = 1100,
                   overlap_chars: int = 200, min_len: int = 80):
    """结构化分块：按章节切，过长再按句切（≤200字尾句重叠），精简标题路径前缀，过滤垃圾块。"""
    chunks = []
    for header_path, body in _split_blocks(md):
        body = body.strip()
        if len(body) <= chunk_size:
            pieces = [body]
        else:
            pieces, buf, buflen = [], [], 0
            # 单句硬切到 ≤ chunk_size
            for sent in _sentences(body, hard=chunk_size):
                if buflen + len(sent) > chunk_size and buf:
                    pieces.append(" ".join(buf).strip())
                    buf = _tail_overlap(buf, overlap_chars) + [sent]
                    buflen = sum(len(x) for x in buf)
                else:
                    buf.append(sent)
                    buflen += len(sent)
            if buf:
                pieces.append(" ".join(buf).strip())
        crumb_txt = _short_crumb(header_path)
        for i, p in enumerate(pieces):
            if _chunk_is_junk(p, min_len):  # 丢弃残留垃圾块
                continue
            crumb = f"【{crumb_txt}】\n" if crumb_txt else ""
            chunks.append({"content": crumb + p.strip(),
                           "metadata": {**source_meta, "section": crumb_txt, "part": i}})
    return chunks


def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--clean-bakeoff", action="store_true",
        help="清洗 _bakeoff/docling/*.md -> docling_cleaned/")
    a = ap.parse_args()
    if a.clean_bakeoff:
        src = Path(__file__).parent / "_bakeoff" / "docling"
        dst = Path(__file__).parent / "_bakeoff" / "docling_cleaned"
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob("*.md")):
            raw = f.read_text(encoding="utf-8")
            cleaned = clean_markdown(raw, lang="en")
            (dst / f.name).write_text(cleaned, encoding="utf-8")
            print(f"  {f.name}: {len(raw)} -> {len(cleaned)} chars ({100*(1-len(cleaned)/max(len(raw),1)):.0f}% 去除)")


if __name__ == "__main__":
    _cli()
