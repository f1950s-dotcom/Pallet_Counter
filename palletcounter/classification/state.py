"""状態分類と枚数算出（要件 3.3 / 3.5 / FR-06 / FR-07 / FR-08）。

本システムの中核。フォークリフトに紐づく検出からPパレの状態を分類し、
状態別ルールで枚数を算出する。

  - 空フォーク（積荷なし）           -> 0
  - 空パレット段積み（モードA）       -> 段数（縦方向の層数）
  - 荷物積載（モードB）              -> ユニット数（横方向の基底ユニット数。
                                        ダブルディープ=2 が標準）

Pパレ以外(nonpallet)は計数しない（FR-08 / 3.5）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from ..config import CountingConfig
from ..types import (
    BBox,
    Detection,
    ObjectClass,
    StateClass,
    bbox_area,
    bbox_center,
)


@dataclass
class ClassificationResult:
    state: StateClass
    count: int
    confidence: float
    pallet_boxes: List[BBox] = field(default_factory=list)
    note: str = ""


def _load_region(box: BBox) -> BBox:
    """フォークリフト矩形から積荷領域を作る。

    側面視点では積荷は車体の前方（横方向）に載り、空パレットの段積みは
    車体より高く（上方向に）積み上がる。横に広く・上に大きく取り、
    最大10段相当の段積みまで取りこぼさない領域とする。
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (
        int(x1 - w * 2.4),   # 左前方
        int(y1 - h * 4.0),   # 上方向（高段積み対応）
        int(x2 + w * 2.4),   # 右前方
        int(y2 + h * 0.3),   # 接地側
    )


def _contains_center(region: BBox, box: BBox) -> bool:
    cx, cy = bbox_center(box)
    rx1, ry1, rx2, ry2 = region
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def _cluster_1d(values: List[float], gap: float) -> int:
    """1次元の座標値を gap 以上の間隙で分割し、クラスタ数を返す。

    モードA: y中心を分割 -> 段数。 モードB: x中心を分割 -> ユニット数。
    """
    if not values:
        return 0
    vs = sorted(values)
    clusters = 1
    for prev, cur in zip(vs, vs[1:]):
        if cur - prev > gap:
            clusters += 1
    return clusters


class StateClassifier:
    def __init__(self, config: CountingConfig):
        self.config = config

    def classify(
        self, forklift_box: BBox, detections: List[Detection]
    ) -> ClassificationResult:
        # フォークリフト周辺（積荷領域）に紐づく検出を抽出。
        region = _load_region(forklift_box)
        pallets = [
            d
            for d in detections
            if d.cls == ObjectClass.PPALLET and _contains_center(region, d.bbox)
        ]
        goods = [
            d
            for d in detections
            if d.cls == ObjectClass.GOODS and _contains_center(region, d.bbox)
        ]
        # nonpallet は近接していても計数しない（FR-08）が、曖昧さの指標にはする
        nonpallets = [
            d
            for d in detections
            if d.cls == ObjectClass.NONPALLET and _contains_center(region, d.bbox)
        ]

        if not pallets:
            return ClassificationResult(
                state=StateClass.EMPTY_FORK,
                count=0,
                confidence=0.99,
                note="積荷なし",
            )

        boxes = [p.bbox for p in pallets]
        base_conf = sum(p.confidence for p in pallets) / len(pallets)
        # パレット代表サイズ（クラスタ間隙のしきい値に使用）
        avg_h = sum((b[3] - b[1]) for b in boxes) / len(boxes)
        avg_w = sum((b[2] - b[0]) for b in boxes) / len(boxes)

        if goods:
            # モードB：荷物積載 -> 横方向の基底ユニット数
            x_centers = [bbox_center(b)[0] for b in boxes]
            units = _cluster_1d(x_centers, gap=avg_w * 0.5)
            units = max(1, units)
            state = StateClass.LOADED
            count = units
            note = f"積載/モードB ユニット={units}"
            confidence = base_conf
        else:
            # モードA：空パレット段積み -> 縦方向の層数
            y_centers = [bbox_center(b)[1] for b in boxes]
            layers = _cluster_1d(y_centers, gap=avg_h * 0.5)
            layers = max(1, min(layers, self.config.max_stack_layers))
            state = StateClass.EMPTY_STACK
            count = layers
            note = f"段積み/モードA 段数={layers}"
            confidence = base_conf
            # 高段は継ぎ目判別が難しく信頼度を下げる（要件リスク#3）
            if layers >= 7:
                confidence *= 0.7
                note += " (高段:要確認傾向)"

        # nonpallet 近接は曖昧 -> 信頼度を下げ人確認へ回す
        if nonpallets:
            confidence *= 0.6
            note += " (非Pパレ近接)"

        return ClassificationResult(
            state=state,
            count=count,
            confidence=round(confidence, 3),
            pallet_boxes=boxes,
            note=note,
        )
