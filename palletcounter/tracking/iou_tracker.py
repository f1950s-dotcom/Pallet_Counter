"""軽量IoUトラッカー（フォークリフト車体の追跡用）。

重い依存（ByteTrack等）を避け、IoUベースの貪欲な対応付けでIDと軌跡を維持する。
1ドックを同時通過する車両数は少なく、これで通過/方向判定には十分。
本番では同インターフェースのままByteTrack等へ差し替え可能。
"""
from __future__ import annotations

from typing import List

from ..types import Detection, ObjectClass, Track, bbox_center, iou


class IouTracker:
    def __init__(self, iou_threshold: float = 0.2, max_missed: int = 15):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, detections: List[Detection]) -> List[Track]:
        """フォークリフト検出のみを追跡対象とし、現在の有効トラックを返す。"""
        forklifts = [d for d in detections if d.cls == ObjectClass.FORKLIFT]

        unmatched = set(range(len(forklifts)))
        # 既存トラックを最良IoUの検出に貪欲対応付け
        for track in self.tracks:
            best_j, best_iou = -1, self.iou_threshold
            for j in unmatched:
                score = iou(track.bbox, forklifts[j].bbox)
                if score >= best_iou:
                    best_j, best_iou = j, score
            if best_j >= 0:
                det = forklifts[best_j]
                track.bbox = det.bbox
                track.history.append(bbox_center(det.bbox))
                track.missed = 0
                unmatched.discard(best_j)
            else:
                track.missed += 1

        # 未対応の検出を新規トラック化
        for j in unmatched:
            det = forklifts[j]
            t = Track(
                track_id=self._next_id,
                bbox=det.bbox,
                cls=ObjectClass.FORKLIFT,
                history=[bbox_center(det.bbox)],
            )
            self._next_id += 1
            self.tracks.append(t)

        # 期限切れトラックを破棄
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return [t for t in self.tracks if t.missed == 0]
