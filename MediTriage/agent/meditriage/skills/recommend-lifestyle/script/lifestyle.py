"""生活方式建议 Skill：本地确定性处方表查表（自包含，无需 Milvus/LLM）。

数据：MediTriage/data/lifestyle/lifestyle_table.json —— 常见慢病/状态的结构化
生活方式建议，内容整理自中国防治指南 /《居民膳食指南》与 GOLD/KDIGO/ACC-AHA/ADA
等权威指南（来源见同目录 SOURCE.md）。
匹配：病名归一（标准名 + 常见别名）→ 精确 → 子串；命中即给结构化建议并标注来源；
未收录则明确告知，绝不返回与查询无关的泛化文案。
"""
import json
from functools import lru_cache
from typing import Any, Dict, List, Optional

from loguru import logger

from meditriage.paths import DATA_DIR

_TABLE_PATH = DATA_DIR / "lifestyle" / "lifestyle_table.json"


@lru_cache(maxsize=1)
def _load_entries() -> List[Dict[str, Any]]:
    """加载处方表条目；文件缺失/损坏则返回空表。"""
    try:
        with open(_TABLE_PATH, encoding="utf-8") as f:
            return json.load(f).get("entries", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"生活方式处方表加载失败：{_TABLE_PATH}（{e}）")
        return []


def _find(query: str) -> Optional[Dict[str, Any]]:
    """病名归一 → 精确（key/别名）→ 子串；无把握返回 None（未收录）。"""
    q = query.strip()
    if not q:
        return None
    entries = _load_entries()

    for e in entries:  # 1) 精确
        if q == e["key"] or q in e.get("aliases", []):
            return e

    if len(q) < 2:  # 子串匹配要求查询≥2字，避免单字误命中
        return None
    for e in entries:  # 2) 子串（病名含查询词，或反之）
        for n in [e["key"]] + e.get("aliases", []):
            if len(n) >= 2 and (n in q or q in n):
                return e
    return None


async def recommend_lifestyle(diagnosis: str) -> Dict[str, Any]:
    """提供生活方式建议（本地确定性处方表）。

    Args:
        diagnosis: 疾病名称（标准名或常见别名）。

    Returns:
        {answer, diagnosis, categories, source}；未收录时 categories 为空、
        source 为 '未收录'，answer 明确告知，不返回无关泛化文案。
    """
    logger.info(f"Lifestyle lookup: {diagnosis}")
    entry = _find(diagnosis)
    if not entry:
        return {
            "answer": (
                f"未收录“{diagnosis}”的专项生活方式建议。建议咨询医生或营养师；"
                "如需一般健康建议可查询“通用健康”。"
            ),
            "diagnosis": diagnosis,
            "categories": [],
            "source": "未收录",
        }

    advice = entry["advice"]
    return {
        "answer": _format(entry),
        "diagnosis": diagnosis,
        "categories": list(advice.keys()),
        "source": entry["source"],
    }


def _format(entry: Dict[str, Any]) -> str:
    """格式化为分节建议文本（显示命中的标准病名，不冒用查询词）。"""
    lines = [f"【{entry['key']}·生活方式建议】"]
    for section, items in entry["advice"].items():
        lines.append(f"\n〔{section}〕")
        lines.extend(f"- {it}" for it in items)
    lines.append(f"\n来源：{entry['source']}")
    lines.append("【免责声明】以上建议仅供参考，具体请遵医嘱或咨询营养师。")
    return "\n".join(lines)
