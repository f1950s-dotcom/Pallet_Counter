"""計数イベントの永続化（SQLite）。FR-09 / FR-12 / FR-11。

時刻/拠点/方向/枚数/状態/信頼度/静止画パス/確認状態を保存する。
依存ライブラリを増やさないため標準ライブラリの sqlite3 を使用。
"""
from __future__ import annotations

import csv
import io
import os
import sqlite3
from contextlib import closing
from typing import List, Optional

from ..types import CountEvent, Direction, StateClass


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT NOT NULL,
    site          TEXT NOT NULL,
    direction     TEXT NOT NULL,
    count         INTEGER NOT NULL,
    state         TEXT NOT NULL,
    confidence    REAL NOT NULL,
    track_id      INTEGER NOT NULL,
    frame_index   INTEGER NOT NULL,
    snapshot_path TEXT,
    auto_confirmed INTEGER NOT NULL DEFAULT 0,
    reviewed      INTEGER NOT NULL DEFAULT 0
);
"""


class EventStore:
    def __init__(self, db_path: str = "data/db/events.sqlite3"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        with closing(self._conn()) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- 書き込み ---
    def add(self, e: CountEvent) -> int:
        with closing(self._conn()) as conn:
            cur = conn.execute(
                """INSERT INTO events
                   (timestamp, site, direction, count, state, confidence,
                    track_id, frame_index, snapshot_path, auto_confirmed, reviewed)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    e.timestamp,
                    e.site,
                    e.direction.value,
                    e.count,
                    e.state.value,
                    e.confidence,
                    e.track_id,
                    e.frame_index,
                    e.snapshot_path,
                    int(e.auto_confirmed),
                    int(e.reviewed),
                ),
            )
            conn.commit()
            return cur.lastrowid

    def review(
        self,
        event_id: int,
        count: Optional[int] = None,
        direction: Optional[str] = None,
    ) -> None:
        """人による確認/補正（FR-11）。枚数・方向を上書きし reviewed=1。"""
        sets = ["reviewed = 1"]
        params: list = []
        if count is not None:
            sets.append("count = ?")
            params.append(int(count))
        if direction is not None:
            sets.append("direction = ?")
            params.append(direction)
        params.append(event_id)
        with closing(self._conn()) as conn:
            conn.execute(f"UPDATE events SET {', '.join(sets)} WHERE id = ?", params)
            conn.commit()

    # --- 読み出し ---
    def _row_to_event(self, r: sqlite3.Row) -> dict:
        d = dict(r)
        d["auto_confirmed"] = bool(d["auto_confirmed"])
        d["reviewed"] = bool(d["reviewed"])
        return d

    def list(self, site: Optional[str] = None, limit: int = 500) -> List[dict]:
        q = "SELECT * FROM events"
        params: list = []
        if site:
            q += " WHERE site = ?"
            params.append(site)
        q += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with closing(self._conn()) as conn:
            return [self._row_to_event(r) for r in conn.execute(q, params)]

    def pending_review(self, site: Optional[str] = None) -> List[dict]:
        """低信頼度かつ未確認の確認キュー（FR-11）。"""
        q = "SELECT * FROM events WHERE auto_confirmed = 0 AND reviewed = 0"
        params: list = []
        if site:
            q += " AND site = ?"
            params.append(site)
        q += " ORDER BY id DESC"
        with closing(self._conn()) as conn:
            return [self._row_to_event(r) for r in conn.execute(q, params)]

    def export_csv(self, site: Optional[str] = None) -> str:
        """確認済み計数を中間データ(CSV)として出力（FR-12 / 8章）。"""
        rows = self.list(site=site, limit=10 ** 9)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            ["id", "timestamp", "site", "direction", "count",
             "state", "confidence", "reviewed", "auto_confirmed"]
        )
        for r in reversed(rows):  # 古い順
            writer.writerow(
                [r["id"], r["timestamp"], r["site"], r["direction"], r["count"],
                 r["state"], r["confidence"], int(r["reviewed"]),
                 int(r["auto_confirmed"])]
            )
        return buf.getvalue()
