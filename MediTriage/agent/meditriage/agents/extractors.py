"""Agent 最终答案的结构化抽取（对应各 worker 的输出格式约定）。"""
import re
from typing import List

_SUGGESTION_HEADER = "【核心建议】"


def extract_suggestions(text: str, limit: int = 5) -> List[str]:
    """抽取【核心建议】段落里的编号建议条目。

    定位【核心建议】到下一个【…】标记之间的文本，取其中的编号列表项；
    无该段落或无编号项时返回空列表，返回至多 limit 条（已去空白）。
    """
    if not text or _SUGGESTION_HEADER not in text:
        return []
    start = text.find(_SUGGESTION_HEADER)
    end = text.find("【", start + len(_SUGGESTION_HEADER))
    if end == -1:
        end = len(text)
    block = text[start:end]
    items = re.findall(r"\d+\.\s*([^\n]+)", block)
    return [s.strip() for s in items if s.strip()][:limit]
