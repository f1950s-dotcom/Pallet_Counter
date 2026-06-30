"""状態分類・枚数算出のテスト（FR-06 / FR-07 / FR-08 / 3.3 / 3.5）。"""
from palletcounter.classification.state import StateClassifier
from palletcounter.config import CountingConfig
from palletcounter.types import Detection, ObjectClass, StateClass


def clf():
    return StateClassifier(CountingConfig())


# フォークリフトは x=400 付近、積荷は前方 x=300 付近に置く
FORK = (380, 350, 460, 462)


def _pallet(cx, base_y, w=120, h=26):
    return Detection(ObjectClass.PPALLET, (cx - w // 2, base_y - h, cx + w // 2, base_y))


def test_empty_fork_counts_zero():
    r = clf().classify(FORK, [])
    assert r.state == StateClass.EMPTY_FORK
    assert r.count == 0


def test_empty_stack_counts_layers():
    # 縦に3段
    dets = [_pallet(300, 450 - i * 38) for i in range(3)]
    r = clf().classify(FORK, dets)
    assert r.state == StateClass.EMPTY_STACK
    assert r.count == 3


def test_loaded_double_deep_counts_two_units():
    # 横並び2枚 + 積荷 -> モードB ユニット=2
    dets = [_pallet(250, 450), _pallet(390, 450),
            Detection(ObjectClass.GOODS, (190, 320, 450, 420))]
    r = clf().classify(FORK, dets)
    assert r.state == StateClass.LOADED
    assert r.count == 2


def test_loaded_single_unit():
    dets = [_pallet(300, 450), Detection(ObjectClass.GOODS, (240, 320, 360, 420))]
    r = clf().classify(FORK, dets)
    assert r.state == StateClass.LOADED
    assert r.count == 1


def test_nonpallet_not_counted():
    dets = [Detection(ObjectClass.NONPALLET, (250, 390, 360, 450))]
    r = clf().classify(FORK, dets)
    assert r.state == StateClass.EMPTY_FORK
    assert r.count == 0


def test_tall_stack_lowers_confidence_for_review():
    dets = [_pallet(300, 452 - i * 38) for i in range(8)]
    r = clf().classify(FORK, dets)
    assert r.count == 8
    assert r.confidence < 0.85  # 高段は要確認へ


def test_stack_capped_at_max_layers():
    cfg = CountingConfig(max_stack_layers=10)
    dets = [_pallet(300, 452 - i * 38) for i in range(13)]
    r = StateClassifier(cfg).classify(FORK, dets)
    assert r.count == 10
