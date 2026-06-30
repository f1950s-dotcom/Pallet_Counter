"""理論在庫の集計（要件 UC-03 / FR-10 / 3.4）。

理論在庫 = 期首在庫（導入時の実地棚卸で確定）+ Σ入庫 − Σ出庫。
イベントストアから受払いを集計する。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..events.store import EventStore
from ..types import Direction


@dataclass
class InventorySummary:
    site: str
    opening: int       # 期首在庫
    inbound: int       # Σ入庫
    outbound: int      # Σ出庫
    theoretical: int   # 理論在庫
    events: int        # 計数イベント数
    pending_review: int  # 確認待ち件数


class InventoryService:
    def __init__(self, store: EventStore, opening_stock: int = 0):
        self.store = store
        self.opening_stock = opening_stock

    def summary(self, site: Optional[str] = None) -> InventorySummary:
        rows = self.store.list(site=site, limit=10 ** 9)
        inbound = sum(r["count"] for r in rows if r["direction"] == Direction.INBOUND.value)
        outbound = sum(r["count"] for r in rows if r["direction"] == Direction.OUTBOUND.value)
        theoretical = self.opening_stock + inbound - outbound
        pending = len(self.store.pending_review(site=site))
        return InventorySummary(
            site=site or "ALL",
            opening=self.opening_stock,
            inbound=inbound,
            outbound=outbound,
            theoretical=theoretical,
            events=len(rows),
            pending_review=pending,
        )
