"""agent_memory 维护：清理测试污染 / 剪枝陈旧 / 统计。

用法（容器内）:
  docker exec medix-fix python3 /workspace/MediTriage/diag/memory_maintenance.py --stats
  docker exec medix-fix python3 /workspace/MediTriage/diag/memory_maintenance.py --clean-test
  docker exec medix-fix python3 /workspace/MediTriage/diag/memory_maintenance.py --prune-stale 90

删除前会把被删行备份到 /workspace/log/agent_memory_deleted_backup.jsonl。
"""
import sys
import json
import argparse
from datetime import datetime, timedelta

# --- 路径引导：向上定位 MediTriage 根（含 agent/meditriage/paths.py），从任意目录可运行 ---
import sys as _sys
from pathlib import Path as _Path
_ASK = next(
    p for p in _Path(__file__).resolve().parents
    if (p / "agent" / "meditriage" / "paths.py").is_file()
)
for _p in (str(_ASK / "agent"), str(_Path(__file__).resolve().parent)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import meditriage.paths as _paths
# --- end 引导 ---
from pymilvus import MilvusClient

URI = "http://medical-milvus:19530"
COLL = "agent_memory"
# 压测/对抗/测试会话前缀（这些 session 的记忆视为污染）
JUNK_PREFIXES = (
    "mem-", "cap-", "stress", "adv-", "adv2-", "test-", "verify-",
    "regress-", "ablate-", "handson-",
)


def _all_rows(c):
    c.flush(COLL)
    n = c.get_collection_stats(COLL).get("row_count", 0)
    if not n:
        return []
    # 主键字段名默认 id（create_collection auto_id=True）；id>=0 匹配全部
    return c.query(
        COLL, filter="id >= 0", output_fields=["id", "metadata"], limit=n
    )


def _meta(r):
    try:
        return json.loads(r.get("metadata", "{}"))
    except Exception:
        return {}


def _is_junk(md):
    sid = str(md.get("session_id", ""))
    uid = str(md.get("user_id", ""))
    return sid.startswith(JUNK_PREFIXES) or uid.startswith("test:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--clean-test", action="store_true")
    ap.add_argument("--prune-stale", type=int, default=0, metavar="DAYS")
    a = ap.parse_args()

    c = MilvusClient(uri=URI)
    rows = _all_rows(c)
    print(f"agent_memory 共 {len(rows)} 行")

    if a.stats:
        from collections import Counter
        users = Counter(_meta(r).get("user_id", "?") for r in rows)
        print("按 user_id:", dict(users))
        junk = [r for r in rows if _is_junk(_meta(r))]
        print(f"疑似污染(test 命名空间或压测前缀): {len(junk)} 行")

    del_ids = []
    if a.clean_test:
        del_ids = [r["id"] for r in rows if _is_junk(_meta(r))]
    if a.prune_stale > 0:
        cutoff = datetime.now() - timedelta(days=a.prune_stale)
        for r in rows:
            ts = _meta(r).get("timestamp", "")
            try:
                if (ts and datetime.fromisoformat(ts) < cutoff
                        and r["id"] not in del_ids):
                    del_ids.append(r["id"])
            except Exception:
                pass

    if del_ids:
        bak = [r for r in rows if r["id"] in set(del_ids)]
        with open(
            str(_paths.LOG_DIR / "agent_memory_deleted_backup.jsonl"),
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                "\n".join(json.dumps(b, ensure_ascii=False) for b in bak)
                + "\n"
            )
        c.delete(COLL, ids=del_ids)
        c.flush(COLL)
        print(
            f"已删除 {len(del_ids)} 行（备份至 "
            f"log/agent_memory_deleted_backup.jsonl）"
        )
    elif a.clean_test or a.prune_stale:
        print("无匹配可删除")


if __name__ == "__main__":
    main()
