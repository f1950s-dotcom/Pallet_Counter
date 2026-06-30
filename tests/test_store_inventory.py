"""イベントストアと理論在庫のテスト（FR-09 / FR-10 / FR-11 / UC-03）。"""
import os
import tempfile

from palletcounter.events.store import EventStore
from palletcounter.inventory import InventoryService
from palletcounter.types import CountEvent, Direction, StateClass


def _event(direction, count, conf=0.95, auto=True):
    return CountEvent(
        timestamp="2026-06-30T10:00:00",
        site="DOCK-01",
        direction=direction,
        count=count,
        state=StateClass.EMPTY_STACK,
        confidence=conf,
        track_id=1,
        frame_index=1,
        auto_confirmed=auto,
    )


def _store():
    tmp = tempfile.mkdtemp()
    return EventStore(os.path.join(tmp, "events.sqlite3"))


def test_inventory_balance():
    store = _store()
    store.add(_event(Direction.INBOUND, 5))
    store.add(_event(Direction.INBOUND, 3))
    store.add(_event(Direction.OUTBOUND, 2))
    inv = InventoryService(store, opening_stock=100)
    s = inv.summary(site="DOCK-01")
    assert s.inbound == 8
    assert s.outbound == 2
    assert s.theoretical == 106  # 100 + 8 - 2


def test_pending_queue_and_review():
    store = _store()
    store.add(_event(Direction.INBOUND, 5, conf=0.6, auto=False))  # 要確認
    store.add(_event(Direction.INBOUND, 2, conf=0.95, auto=True))  # 自動確定
    pending = store.pending_review(site="DOCK-01")
    assert len(pending) == 1
    eid = pending[0]["id"]

    # 人が枚数を6へ補正
    store.review(eid, count=6)
    assert store.pending_review(site="DOCK-01") == []
    inv = InventoryService(store, opening_stock=0)
    assert inv.summary(site="DOCK-01").inbound == 8  # 6 + 2


def test_csv_export_header_and_rows():
    store = _store()
    store.add(_event(Direction.INBOUND, 5))
    csv = store.export_csv(site="DOCK-01")
    lines = csv.strip().splitlines()
    assert lines[0].startswith("id,timestamp,site,direction,count")
    assert len(lines) == 2
