"""疾病编码查询 Skill：本地 ICD-10 中文码表确定性查表（自包含，无需 Milvus/LLM）。

数据：MediTriage/data/icd10/disease.csv（1586 条三位类目码，code<TAB>中文名）
      + disease_catalog.csv（章节/区间目录）。来源与许可见同目录 SOURCE.md。
匹配：口语归一 → 精确 → 子串（病名包含查询词，按码表顺序）；命中多条并列返回；
      不做模糊近似，没把握就明确返回“未收录”，绝不返回不相关的最近邻。
"""
import csv
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from loguru import logger

from meditriage.paths import DATA_DIR

_ICD_DIR = DATA_DIR / "icd10"

# 常见口语 → 标准病名（每条均已在码表中验证可命中；仅做名称归一，不直接指定编码）
_ALIAS = {
    "冠心病": "缺血性心脏病",
    "心梗": "心肌梗死",
    "心肌梗塞": "心肌梗死",
    "脑梗": "脑梗死",
    "脑梗塞": "脑梗死",
    "乙肝": "乙型肝炎",
    "慢阻肺": "慢性阻塞性肺",
    "1型糖尿病": "胰岛素依赖型糖尿病",
    "2型糖尿病": "非胰岛素依赖型糖尿病",
    # 码表标准名是"呼吸道结核"（A15/A16），口语"肺结核"原样查不到
    "肺结核": "呼吸道结核",
    "肺痨": "呼吸道结核",
}


@lru_cache(maxsize=1)
def _load_codes() -> List[Tuple[str, str]]:
    """加载 [(code, name), ...]；文件缺失则返回空表（跳过本地查表）。"""
    rows: List[Tuple[str, str]] = []
    path = _ICD_DIR / "disease.csv"
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)  # 跳过表头 code/disease
            for r in reader:
                if len(r) >= 2 and r[0].strip():
                    rows.append((r[0].strip(), r[1].strip()))
    except FileNotFoundError:
        logger.warning(f"ICD-10 码表缺失：{path}（disease_code 将无法本地查表）")
    return rows


@lru_cache(maxsize=1)
def _load_catalog() -> List[Tuple[str, str, int, str]]:
    """加载章节/区间目录 [(lo, hi, level, name), ...]。"""
    rows: List[Tuple[str, str, int, str]] = []
    path = _ICD_DIR / "disease_catalog.csv"
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for r in reader:
                if len(r) >= 4 and r[2].strip().isdigit():
                    rows.append(
                        (r[0].strip(), r[1].strip(), int(r[2]), r[3].strip())
                    )
    except FileNotFoundError:
        pass
    return rows


def _code_key(code: str):
    """'I10' -> ('I', 10)，用于区间比较；非法返回 None。"""
    code = code.strip().upper()
    if len(code) >= 3 and code[0].isalpha() and code[1:3].isdigit():
        return (code[0], int(code[1:3]))
    return None


def _chapter_of(code: str) -> str:
    """返回 code 所属的（章 / 区间）描述，按层级拼接。"""
    k = _code_key(code)
    if not k:
        return ""
    hits = []
    for lo, hi, level, name in _load_catalog():
        klo, khi = _code_key(lo), _code_key(hi)
        if klo and khi and klo <= k <= khi:
            hits.append((level, name))
    hits.sort()  # level 1（章）在前
    return " / ".join(name for _, name in hits)


def _match(query: str, top_k: int = 5) -> List[Dict[str, str]]:
    """口语归一 → 精确 → 子串（病名包含查询词）。

    只在“病名包含查询词”时命中；不做模糊近似，宁可返回“未收录”也不返回
    不相关的最近邻。命中多条按码表顺序（低位码多为未特指型）。
    """
    raw = query.strip()
    q = _ALIAS.get(raw, raw)
    codes = _load_codes()
    if not q or not codes:
        return []

    # 候选：归一词；若以“病/症”结尾再补一个去尾变体（高血压病 → 高血压）
    cands = [q]
    if len(q) > 2 and q[-1] in "病症":
        cands.append(q[:-1])

    ranked: List[Tuple[str, str]] = []
    for cand in cands:
        exact = [(c, n) for c, n in codes if n == cand]
        contains = [(c, n) for c, n in codes if n != cand and cand in n]
        ranked = exact + contains  # 码表按编码排序，低位码在前
        if ranked:
            break

    out: List[Dict[str, str]] = []
    seen = set()
    for c, n in ranked[:top_k]:
        if c in seen:
            continue
        seen.add(c)
        out.append({"code": c, "name": n, "chapter": _chapter_of(c)})
    return out


async def disease_code(disease_name: str) -> Dict[str, Any]:
    """查询疾病 ICD-10 编码（本地中文码表，确定性查表）。

    Args:
        disease_name: 疾病名称（标准或常见口语均可）。

    Returns:
        {answer, icd10_code, category, matches, source}
        无匹配时 icd10_code 为空、matches 为空，answer 明确告知未收录。
    """
    logger.info(f"ICD-10 lookup: {disease_name}")
    matches = _match(disease_name)

    if not matches:
        return {
            "answer": (
                f"未在 ICD-10 码表中找到与“{disease_name}”匹配的条目。"
                "请尝试更标准的疾病名称，或用 search_knowledge 检索相关说明。"
            ),
            "icd10_code": "",
            "category": "",
            "matches": [],
            "source": "ICD-10 本地码表",
        }

    top = matches[0]
    return {
        "answer": _format(disease_name, matches),
        "icd10_code": top["code"],
        "category": top["chapter"],
        "matches": matches,
        "source": "ICD-10 本地码表（中文三位类目）",
    }


def _format(query: str, matches: List[Dict[str, str]]) -> str:
    """格式化匹配结果。"""
    lines = [f"【ICD-10 编码】“{query}”匹配到 {len(matches)} 条："]
    for m in matches:
        chapter = f"（{m['chapter']}）" if m["chapter"] else ""
        lines.append(f"- {m['code']}  {m['name']}{chapter}")
    lines.append(
        "\n注：以上为 ICD-10 三位类目码；四位及以上细分编码以临床版/医保版为准。"
    )
    return "\n".join(lines)
