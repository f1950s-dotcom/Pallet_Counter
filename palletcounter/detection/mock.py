"""モック検出器：色域抽出による検出。

合成映像（tools/generate_sample_video.py）で描画された既知色の領域を
HSVしきい値＋輪郭抽出で検出する。学習済みモデルや実カメラ映像が無い段階で
パイプライン全体（検出->追跡->計数->保存->表示）を実機相当に通すための実装。
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from ..types import Detection, ObjectClass
from .base import Detector
from .palette import HSV_RANGES


class MockDetector(Detector):
    def __init__(self, min_area: int = 150):
        self.min_area = min_area

    def detect(self, frame: np.ndarray) -> List[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections: List[Detection] = []
        for cls, (lo, hi) in HSV_RANGES.items():
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            # ノイズ除去
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in contours:
                area = cv2.contourArea(c)
                if area < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                detections.append(
                    Detection(cls=cls, bbox=(x, y, x + w, y + h), confidence=0.95)
                )
        return detections
