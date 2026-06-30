"""計数パイプライン（要件 3.6 のフローを実装）。

  動体/存在トリガー -> 物体検出 -> トラッキング -> ライン通過/方向判定
   -> 状態分類 -> 枚数算出 -> 計数イベント確定 -> 在庫反映/出力

各フレームを process_frame() に渡すと、確定した計数イベントを返す。
オフライン動画にもリアルタイムストリームにも同一ロジックで適用できる。
"""
from __future__ import annotations

import datetime as _dt
import os
from typing import List, Optional

import cv2
import numpy as np

from .classification.state import StateClassifier
from .config import DockConfig
from .counting.line_crossing import LineCrossingDetector
from .detection import build_detector
from .events.store import EventStore
from .tracking.iou_tracker import IouTracker
from .types import CountEvent, Detection, ObjectClass


class CountingPipeline:
    def __init__(
        self,
        config: DockConfig,
        store: Optional[EventStore] = None,
        save_snapshots: bool = True,
        detector_kwargs: Optional[dict] = None,
    ):
        self.config = config
        self.detector = build_detector(config.detector, **(detector_kwargs or {}))
        self.tracker = IouTracker(
            iou_threshold=config.counting.track_iou_threshold,
            max_missed=config.counting.track_max_missed,
        )
        self.classifier = StateClassifier(config.counting)
        self.store = store or EventStore(config.db_path)
        self.save_snapshots = save_snapshots
        self.frame_index = -1
        self._line_x: Optional[int] = None
        self._crossing: Optional[LineCrossingDetector] = None
        if save_snapshots:
            os.makedirs(config.snapshot_dir, exist_ok=True)

    def _ensure_line(self, frame: np.ndarray) -> None:
        if self._crossing is not None:
            return
        h, w = frame.shape[:2]
        self._line_x = int(w * self.config.line.x_ratio)
        self._crossing = LineCrossingDetector(
            line_x=self._line_x,
            inbound_is_left_to_right=self.config.line.inbound_is_left_to_right,
            cooldown_frames=self.config.counting.recount_cooldown_frames,
        )

    @staticmethod
    def _has_motion(detections: List[Detection]) -> bool:
        """動体/存在トリガー（FR-02）。フォークリフトが居る時のみ後段を起動。

        実機では軽量な動体検知でモデル起動を絞るが、本PoCでは検出結果に
        フォークリフトが含まれるかで近似する。
        """
        return any(d.cls == ObjectClass.FORKLIFT for d in detections)

    def _save_snapshot(self, frame: np.ndarray) -> Optional[str]:
        if not self.save_snapshots:
            return None
        name = f"{self.config.site_id}_{self.frame_index:06d}.jpg"
        path = os.path.join(self.config.snapshot_dir, name)
        cv2.imwrite(path, frame)
        return path

    def process_frame(self, frame: np.ndarray) -> List[CountEvent]:
        self.frame_index += 1
        self._ensure_line(frame)

        detections = self.detector.detect(frame)
        if not self._has_motion(detections):
            # 通過なし：重い後段はスキップ（省リソース）。トラッカーは更新。
            self.tracker.update(detections)
            return []

        tracks = self.tracker.update(detections)
        crossings = self._crossing.update(tracks, self.frame_index)

        events: List[CountEvent] = []
        for cr in crossings:
            result = self.classifier.classify(cr.track.bbox, detections)
            auto = result.confidence >= self.config.counting.auto_confirm_threshold
            snapshot = self._save_snapshot(frame)
            event = CountEvent(
                timestamp=_dt.datetime.now().isoformat(timespec="seconds"),
                site=self.config.site_id,
                direction=cr.direction,
                count=result.count,
                state=result.state,
                confidence=result.confidence,
                track_id=cr.track.track_id,
                frame_index=self.frame_index,
                snapshot_path=snapshot,
                auto_confirmed=auto,
                reviewed=False,
            )
            self.store.add(event)
            events.append(event)
        return events

    def run_video(self, path: str, progress: bool = False) -> List[CountEvent]:
        """動画ファイル全体を処理し、確定イベントの一覧を返す。"""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"動画を開けません: {path}")
        all_events: List[CountEvent] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                evts = self.process_frame(frame)
                all_events.extend(evts)
                if progress and evts:
                    for e in evts:
                        print(
                            f"[frame {e.frame_index}] {e.direction.value} "
                            f"{e.state.value} count={e.count} "
                            f"conf={e.confidence} "
                            f"{'AUTO' if e.auto_confirmed else 'REVIEW'}"
                        )
        finally:
            cap.release()
        return all_events
