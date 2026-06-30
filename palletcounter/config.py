"""1ドック単位ユニットの設定（要件 NFR-10: 横展開性）。

倉庫ごと・ドックごとに本設定を差し替えることで横展開する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class LineConfig:
    """仮想計数ライン(ROI)の設定。

    画面を縦に横切る垂直ラインを基本とする（側面設置カメラ前提・要件6章）。
    x_ratio: フレーム幅に対するラインのx位置(0.0-1.0)。
    inbound_is_left_to_right: 左->右の通過を入庫とみなすか。
    """

    x_ratio: float = 0.5
    inbound_is_left_to_right: bool = True


@dataclass
class CountingConfig:
    """状態分類・枚数算出のしきい値。"""

    # 高信頼度の自動確定しきい値（要件 NFR-01: 自動確定率を最大化）
    auto_confirm_threshold: float = 0.85
    # 積荷(goods)がパレット上に存在すると判定する重なり比率
    goods_overlap_ratio: float = 0.15
    # 段積み(モードA)の最大段数（要件: 最大10段程度）
    max_stack_layers: int = 10
    # トラッキングで同一とみなす最小IoU
    track_iou_threshold: float = 0.2
    # 連続未検出でトラックを破棄するまでのフレーム数
    track_max_missed: int = 15
    # 同一トラックの再計数を抑止するクールダウン（NFR-05 二重計数抑止）
    recount_cooldown_frames: int = 30


@dataclass
class DockConfig:
    """1ドック分の全設定。"""

    site_id: str = "DOCK-01"
    line: LineConfig = field(default_factory=LineConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)
    # 検出器の種類: "mock"（合成映像用）/ "yolo"（学習済みモデル）
    detector: str = "mock"
    # スナップショット保存先
    snapshot_dir: str = "data/snapshots"
    # イベントDBパス
    db_path: str = "data/db/events.sqlite3"

    def frame_size(self, w: int, h: int) -> Tuple[int, int]:
        return (w, h)
