"""YOLO検出器（学習済みモデルによる仮実装）。

要件 7章の通り、本番ではPパレ/フォークリフト/積荷を自社データで学習した
モデルが必要。COCO等の汎用学習済みモデルにはPパレクラスが無いため、本クラスは
「学習済みモデルの組み込み口（インテグレーションポイント）」を提供する位置づけ。

- weights にPパレ学習済みモデル(.pt)を渡せば本番相当で動作する。
- class_map で「モデルのクラス名 -> ObjectClass」を対応付ける。
- ultralytics 未導入時はインポート時に明示的なエラーを出す。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..types import Detection, ObjectClass
from .base import Detector

# 既定のクラス対応（自社学習モデルのラベル名を想定）
DEFAULT_CLASS_MAP: Dict[str, ObjectClass] = {
    "forklift": ObjectClass.FORKLIFT,
    "ppallet": ObjectClass.PPALLET,
    "p_pallet": ObjectClass.PPALLET,
    "pallet": ObjectClass.PPALLET,
    "goods": ObjectClass.GOODS,
    "load": ObjectClass.GOODS,
    "nonpallet": ObjectClass.NONPALLET,
}


class YoloDetector(Detector):
    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.25,
        class_map: Optional[Dict[str, ObjectClass]] = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "YoloDetector を使うには 'pip install ultralytics' が必要です。"
                "学習データが無い段階では detector='mock' を使用してください。"
            ) from e
        self.model = YOLO(weights)
        self.conf = conf
        self.class_map = class_map or DEFAULT_CLASS_MAP

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        detections: List[Detection] = []
        for r in results:
            names = r.names
            for box in r.boxes:
                name = names[int(box.cls)]
                mapped = self.class_map.get(name)
                if mapped is None:
                    continue  # 対応表に無いクラスは計数対象外
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                detections.append(
                    Detection(
                        cls=mapped,
                        bbox=(x1, y1, x2, y2),
                        confidence=float(box.conf),
                    )
                )
        return detections
