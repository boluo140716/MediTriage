"""RAG 未命中事件的被动沉淀。

把线上请求触发的低相关 / 空检索（miss 事件）以结构化 JSON 行落盘到
LOG_DIR/badcase/rag_misses.jsonl，供周期复盘与语料扩展选题：

    {"ts", "route", "reason", "query", "n_hits", "top": [{doc_id, score}]}

reason 取值：
    low_relevance  检索有结果但全部低于阈值（兜底弃权）
    empty          检索结果为空
    unavailable    Milvus 不可用（基础设施故障，非检索质量）
    borderline     命中但 top1 分数处于灰区——勉强喂给了 LLM，相关性存疑

注意：query 可能含用户健康信息，本文件只随 log/ 留在本机（已 .gitignore），
不得外发；落盘失败只记 debug 日志，绝不影响检索主流程。
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from meditriage.paths import LOG_DIR

_BADCASE_FILE = "rag_misses.jsonl"


def log_rag_miss(
    query: str,
    results: Optional[List[Dict[str, Any]]] = None,
    route: str = "search",
    reason: str = "low_relevance",
    base_dir: Optional[Path] = None,
) -> None:
    """记录一次检索未命中 / 弱命中（append 一行 JSON，永不抛错）。

    Args:
        query: 用户查询（截断到 200 字保存）。
        results: 检索返回（可空）；只保留 top3 的来源与分数，不存正文。
        route: 触发来源（search / guideline）。
        reason: 见模块 docstring。
        base_dir: 覆盖输出目录（测试注入用），默认 LOG_DIR/badcase。
    """
    try:
        top = []
        for d in (results or [])[:3]:
            meta = d.get("metadata", {}) if isinstance(d, dict) else {}
            top.append({
                "doc_id": str(meta.get("doc_id") or meta.get("source") or ""),
                "score": round(float(d.get("score", 0.0) or 0.0), 4),
            })
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "route": route,
            "reason": reason,
            "query": (query or "")[:200],
            "n_hits": len(results or []),
            "top": top,
        }
        out_dir = Path(base_dir) if base_dir else LOG_DIR / "badcase"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / _BADCASE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:  # 遥测通道绝不反噬主流程
        logger.debug(f"badcase 落盘失败（忽略）: {e}")
