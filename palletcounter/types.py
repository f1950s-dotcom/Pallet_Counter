"""パイプライン全体で共有するデータ構造。

各モジュール間はこれらの軽量な値オブジェクトで疎結合に連携する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def bbox_center(b: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(b: BBox) -> float:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a: BBox, b: BBox) -> float:
    """2つの矩形の IoU。トラッキングの対応付けに使用。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


class ObjectClass(str, Enum):
    """検出器が返す物体クラス。"""

    FORKLIFT = "forklift"      # フォークリフト車体
    PPALLET = "ppallet"        # Pパレ（計数対象）
    GOODS = "goods"            # 積荷（積載判定に使用）
    NONPALLET = "nonpallet"    # Pパレ以外のパレット/物体（計数対象外）


@dataclass
class Detection:
    """1フレーム内の検出結果。"""

    cls: ObjectClass
    bbox: BBox
    confidence: float = 1.0


@dataclass
class Track:
    """フレーム間で同一物体に付与されるID付き軌跡（フォークリフト用）。"""

    track_id: int
    bbox: BBox
    cls: ObjectClass
    # 軌跡（中心座標の履歴）。方向判定に使用。
    history: list = field(default_factory=list)
    missed: int = 0  # 連続未検出フレーム数

    @property
    def center(self) -> Tuple[float, float]:
        return bbox_center(self.bbox)


class Direction(str, Enum):
    """通過方向。"""

    INBOUND = "inbound"    # 入庫（受け / +）
    OUTBOUND = "outbound"  # 出庫（払い / -）
    UNKNOWN = "unknown"


class StateClass(str, Enum):
    """積荷の状態分類（要件 3.3）。"""

    EMPTY_FORK = "empty_fork"      # 空フォーク -> 0
    EMPTY_STACK = "empty_stack"    # 空パレット段積み（モードA）-> 段数
    LOADED = "loaded"              # 荷物積載（モードB）-> ユニット数


# 受払いの符号。在庫計算 = 期首 + Σ(sign * 枚数)
DIRECTION_SIGN = {
    Direction.INBOUND: +1,
    Direction.OUTBOUND: -1,
    Direction.UNKNOWN: 0,
}


@dataclass
class CountEvent:
    """確定した計数イベント（要件 FR-09）。"""

    timestamp: str          # ISO8601
    site: str               # 拠点/ドックID
    direction: Direction
    count: int              # 枚数
    state: StateClass
    confidence: float
    track_id: int
    frame_index: int
    snapshot_path: Optional[str] = None
    auto_confirmed: bool = False   # 高信頼度で自動確定したか
    reviewed: bool = False         # 人が確認/補正済みか

    @property
    def signed_count(self) -> int:
        return DIRECTION_SIGN[self.direction] * self.count
