"""转诊单存储：SQLite（stdlib sqlite3，WAL + 单例锁，线程安全）。

表结构见 _SCHEMA；状态机：
    ai_processing(建单) -> escalated(已转人工) -> doctor_replied(医生已回复)
非法流转抛 ValueError（Web 层转 409）。幂等建单：同一会话存在未回复单时
复用已有单，不重复创建（防追问/重复请求刷单）。

DB 路径默认 data/escalations.db（paths.DATA_DIR），env MEDITRIAGE_ESCALATION_DB 可覆盖。
"""
import hashlib
import json
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from meditriage.paths import DATA_DIR

# ---- 状态机常量 ----
STATUS_AI_PROCESSING = "ai_processing"
STATUS_ESCALATED = "escalated"
STATUS_DOCTOR_REPLIED = "doctor_replied"
OPEN_STATUSES = (STATUS_AI_PROCESSING, STATUS_ESCALATED)

# 合法流转表（key=当前状态, value=允许的下一状态集合）
TRANSITIONS: Dict[str, set] = {
    STATUS_AI_PROCESSING: {STATUS_ESCALATED},
    STATUS_ESCALATED: {STATUS_DOCTOR_REPLIED},
    STATUS_DOCTOR_REPLIED: set(),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS escalations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    escalation_id TEXT UNIQUE NOT NULL,
    session_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL DEFAULT '', 
    user_id TEXT NOT NULL DEFAULT 'default',
    question TEXT NOT NULL,
    answer_preview TEXT,
    summary_json TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    confidence REAL NOT NULL,
    reasons_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    doctor_reply TEXT,
    replied_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_esc_status ON escalations(status);
CREATE INDEX IF NOT EXISTS idx_esc_session ON escalations(session_id);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    """sqlite3.Row -> dict，并把 summary_json / reasons_json 还原为对象。"""
    if row is None:
        return None
    d = dict(row)
    for key in ("summary_json", "reasons_json"):
        raw = d.pop(key, None)
        try:
            d[key.replace("_json", "")] = json.loads(raw) if raw else None
        except (TypeError, ValueError):
            d[key.replace("_json", "")] = None
    return d


class EscalationStore:
    """SQLite 转诊单存储。进程内单例使用；每个实例独立连接 + 锁。"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(
            db_path or os.environ.get(
                "MEDITRIAGE_ESCALATION_DB", str(DATA_DIR / "escalations.db")
            )
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(f"PRAGMA journal_mode=WAL;{_SCHEMA}")
            self._ensure_column("fingerprint")
            self._conn.commit()

    def _ensure_column(self, column: str):
        """老库平滑迁移：表已存在但缺列时 ALTER TABLE 补列。"""
        cols = [
            r[1] for r in self._conn.execute("PRAGMA table_info(escalations)")
        ]
        if column not in cols:
            self._conn.execute(
                f"ALTER TABLE escalations ADD COLUMN {column} TEXT NOT NULL "
                "DEFAULT ''"
            )

    @staticmethod
    def _fingerprint(question: str) -> str:
        """同一提问（忽略空白差异）映射为稳定指纹，用于幂等建单。"""
        norm = re.sub(r"\s+", "", question or "")
        return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]

    def close(self):
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    # ---- 内部 ----
    def _execute(self, sql: str, params=()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    @staticmethod
    def _new_escalation_id() -> str:
        return f"ESC-{uuid.uuid4().hex[:8].upper()}"

    def _get(self, esc_id_or_int) -> Optional[Dict[str, Any]]:
        if isinstance(esc_id_or_int, int) or str(esc_id_or_int).isdigit():
            cur = self._execute(
                "SELECT * FROM escalations WHERE id=?", (int(esc_id_or_int),)
            )
        else:
            cur = self._execute(
                "SELECT * FROM escalations WHERE escalation_id=?",
                (str(esc_id_or_int),),
            )
        return _row_to_dict(cur.fetchone())

    def _find_open(self, session_id: str,
                   fingerprint: str = "") -> Optional[Dict[str, Any]]:
        ph = ",".join("?" * len(OPEN_STATUSES))
        cur = self._execute(
            "SELECT * FROM escalations WHERE session_id=? AND fingerprint=? "
            f"AND status IN ({ph}) ORDER BY id DESC LIMIT 1",
            (session_id, fingerprint, *OPEN_STATUSES),
        )
        return _row_to_dict(cur.fetchone())

    def _transition(self, esc_id_or_int, to_status: str) -> Dict[str, Any]:
        """状态流转；非法流转抛 ValueError（Web 层转 409）。"""
        row = self._get(esc_id_or_int)
        if row is None:
            raise KeyError(f"转诊单不存在: {esc_id_or_int}")
        cur = row["status"]
        if to_status not in TRANSITIONS.get(cur, set()):
            raise ValueError(f"非法状态流转: {cur} -> {to_status}")
        self._execute(
            "UPDATE escalations SET status=?, updated_at=? WHERE id=?",
            (to_status, _now(), row["id"]),
        )
        return self._get(row["id"])

    # ---- 对外 ----
    def create(
        self,
        session_id: str,
        question: str,
        summary: Dict[str, Any],
        risk_level: str,
        confidence: float,
        reasons: List[str],
        user_id: str = "default",
        answer_preview: str = "",
        fingerprint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """建单（AI处理中 -> 已转人工）。

        幂等键 = session_id + question 指纹：同一提问重复触发复用已有单；
        同会话内不同危机（换症状/换问法）会开新单，避免返回过期信息。
        """
        sid = session_id or "anon"
        fp = fingerprint or self._fingerprint(question)
        existing = self._find_open(sid, fp)
        if existing:
            return existing

        now = _now()
        esc_id = self._new_escalation_id()
        with self._lock:
            self._conn.execute(
                "INSERT INTO escalations (escalation_id, session_id, user_id, "
                "question, answer_preview, summary_json, risk_level, confidence, "
                "reasons_json, status, created_at, updated_at, fingerprint) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    esc_id, sid, user_id or "default", question or "",
                    (answer_preview or "")[:500],
                    json.dumps(summary, ensure_ascii=False),
                    risk_level, float(confidence),
                    json.dumps(list(reasons or []), ensure_ascii=False),
                    STATUS_AI_PROCESSING, now, now, fp,
                ),
            )
            self._conn.commit()
        # 建单即流转：AI处理中 -> 已转人工
        return self._transition(esc_id, STATUS_ESCALATED)

    def get(self, esc_id_or_int) -> Optional[Dict[str, Any]]:
        return self._get(esc_id_or_int)

    def list(self, status: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        if status:
            rows = self._execute(
                "SELECT * FROM escalations WHERE status=? ORDER BY id DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._execute(
                "SELECT * FROM escalations ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def delete(self, esc_id_or_int) -> bool:
        """删除单条转诊单（幂等：不存在返回 False）。

        与 _get 一致：支持 int id 或 "ESC-XXXX" 字符串。"""
        row = self._get(esc_id_or_int)
        if row is None:
            return False
        self._execute("DELETE FROM escalations WHERE id=?", (row["id"],))
        return True

    def delete_many(self, ids) -> int:
        """批量删除转诊单（幂等：不存在的 id 跳过），返回实际删除数。"""
        n = 0
        for x in (ids or []):
            if self.delete(x):
                n += 1
        return n

    def reply(self, esc_id_or_int, reply_text: str) -> Dict[str, Any]:
        """医生回复：escalated -> doctor_replied。非法流转/空回复抛 ValueError。

        状态校验与写入统一走 _transition（TRANSITIONS 为单一事实源），
        这里保留友好的错误提示，并先落医生回复内容。
        """
        reply_text = (reply_text or "").strip()
        if not reply_text:
            raise ValueError("回复内容不能为空")
        row = self._get(esc_id_or_int)
        if row is None:
            raise KeyError(f"转诊单不存在: {esc_id_or_int}")
        if row["status"] != STATUS_ESCALATED:
            raise ValueError(
                f"当前状态 {row['status']}，仅「已转人工」可回复"
            )
        self._execute(
            "UPDATE escalations SET doctor_reply=?, replied_at=?, updated_at=? "
            "WHERE id=?",
            (reply_text, _now(), _now(), row["id"]),
        )
        return self._transition(row["id"], STATUS_DOCTOR_REPLIED)
