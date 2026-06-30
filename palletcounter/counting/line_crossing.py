"""仮想計数ライン通過の検知と方向判定（要件 FR-05 / 3.2 / 3.4）。

垂直ラインを車体中心が所定方向に「完全通過」した瞬間を1イベントとする。
滞留・後退・往復による二重計数を抑止する（NFR-05）ため、トラックごとに
最後に計数したフレームを記録しクールダウンを設ける。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..types import Direction, Track


@dataclass
class CrossingResult:
    track: Track
    direction: Direction


class LineCrossingDetector:
    def __init__(
        self,
        line_x: int,
        inbound_is_left_to_right: bool = True,
        cooldown_frames: int = 30,
    ):
        self.line_x = line_x
        self.inbound_is_left_to_right = inbound_is_left_to_right
        self.cooldown_frames = cooldown_frames
        # track_id -> 直近のラインに対する符号 (-1: 左, +1: 右)
        self._last_side: Dict[int, int] = {}
        # track_id -> 最後に計数したフレーム
        self._last_count_frame: Dict[int, int] = {}

    def _side(self, x: float) -> int:
        return 1 if x >= self.line_x else -1

    def update(self, tracks: List[Track], frame_index: int) -> List[CrossingResult]:
        results: List[CrossingResult] = []
        active_ids = set()
        for track in tracks:
            active_ids.add(track.track_id)
            cx, _ = track.center
            side = self._side(cx)
            prev = self._last_side.get(track.track_id)
            self._last_side[track.track_id] = side

            if prev is None or prev == side:
                continue  # 未交差 or 同じ側に留まる

            # 符号が反転 = ライン通過
            last = self._last_count_frame.get(track.track_id, -10 ** 9)
            if frame_index - last < self.cooldown_frames:
                continue  # クールダウン中（往復/滞留の二重計数を抑止）

            crossed_l_to_r = prev < side  # -1 -> +1
            inbound = crossed_l_to_r == self.inbound_is_left_to_right
            direction = Direction.INBOUND if inbound else Direction.OUTBOUND
            self._last_count_frame[track.track_id] = frame_index
            results.append(CrossingResult(track=track, direction=direction))

        # 消滅したトラックの状態を掃除（メモリリーク防止）
        for tid in list(self._last_side.keys()):
            if tid not in active_ids:
                self._last_side.pop(tid, None)
        return results
