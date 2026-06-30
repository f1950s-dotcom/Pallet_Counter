"""ライン通過・方向判定・二重計数抑止のテスト（FR-05 / NFR-05）。"""
from palletcounter.counting.line_crossing import LineCrossingDetector
from palletcounter.types import Direction, ObjectClass, Track


def _track(tid, cx):
    return Track(track_id=tid, bbox=(cx - 10, 0, cx + 10, 20), cls=ObjectClass.FORKLIFT)


def test_left_to_right_is_inbound():
    det = LineCrossingDetector(line_x=100, inbound_is_left_to_right=True)
    assert det.update([_track(1, 40)], 0) == []          # 左側、初回は未交差
    res = det.update([_track(1, 160)], 1)                 # 右へ通過
    assert len(res) == 1
    assert res[0].direction == Direction.INBOUND


def test_right_to_left_is_outbound():
    det = LineCrossingDetector(line_x=100, inbound_is_left_to_right=True)
    det.update([_track(1, 160)], 0)
    res = det.update([_track(1, 40)], 1)
    assert len(res) == 1
    assert res[0].direction == Direction.OUTBOUND


def test_inbound_direction_can_be_reversed():
    det = LineCrossingDetector(line_x=100, inbound_is_left_to_right=False)
    det.update([_track(1, 40)], 0)
    res = det.update([_track(1, 160)], 1)
    assert res[0].direction == Direction.OUTBOUND


def test_no_double_count_within_cooldown():
    """滞留・往復しても、クールダウン中は二重計数しない。"""
    det = LineCrossingDetector(line_x=100, cooldown_frames=30)
    det.update([_track(1, 40)], 0)
    r1 = det.update([_track(1, 160)], 1)   # 1回目: 計数
    r2 = det.update([_track(1, 40)], 2)    # すぐ戻る: 抑止
    r3 = det.update([_track(1, 160)], 3)   # また越える: 抑止
    assert len(r1) == 1
    assert r2 == [] and r3 == []


def test_recount_after_cooldown():
    det = LineCrossingDetector(line_x=100, cooldown_frames=5)
    det.update([_track(1, 40)], 0)
    assert len(det.update([_track(1, 160)], 1)) == 1
    # クールダウン経過後の再通過は計数する
    assert det.update([_track(1, 40)], 10) != []
