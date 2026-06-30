"""オープン語彙検出器（YOLO-World）。

学習データが無い段階で「本物の映像」を試すための検出器。
テキストで指定したクラス（例: "forklift", "wooden pallet"）をゼロショットで
検出する。学習版モデルには精度で劣るが、収集・学習の前に実映像で
パイプラインを試すのに有用。

  pip install ultralytics
  検出器名 "openvocab"（detector='openvocab'）で利用。

注意: 本番精度は要件 7 章の自社学習モデルが前提。本検出器は試行用。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..types import Detection, ObjectClass
from .base import Detector

# 各 ObjectClass に対応させるテキストプロンプト（複数可）。
# 同義語を並べると拾いやすい。
DEFAULT_PROMPTS: Dict[ObjectClass, List[str]] = {
    ObjectClass.FORKLIFT: ["forklift", "fork lift truck", "lift truck"],
    ObjectClass.PPALLET: ["pallet", "wooden pallet", "plastic pallet"],
    ObjectClass.GOODS: ["cardboard box", "stacked boxes", "cargo"],
}


class OpenVocabDetector(Detector):
    def __init__(
        self,
        weights: str = "yolov8s-world.pt",
        conf: float = 0.10,
        prompts: Optional[Dict[ObjectClass, List[str]]] = None,
    ):
        try:
            from ultralytics import YOLOWorld
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "OpenVocabDetector には 'pip install ultralytics' が必要です。"
            ) from e
        self.prompts = prompts or DEFAULT_PROMPTS
        # プロンプト文字列 -> ObjectClass の逆引き
        self._text_to_cls: Dict[str, ObjectClass] = {}
        ordered_texts: List[str] = []
        for cls, texts in self.prompts.items():
            for t in texts:
                self._text_to_cls[t] = cls
                ordered_texts.append(t)
        self.model = YOLOWorld(weights)
        self.model.set_classes(ordered_texts)
        self._names = ordered_texts
        self.conf = conf

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        detections: List[Detection] = []
        for r in results:
            names = r.names
            for box in r.boxes:
                text = names[int(box.cls)]
                cls = self._text_to_cls.get(text)
                if cls is None:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                detections.append(
                    Detection(cls=cls, bbox=(x1, y1, x2, y2),
                              confidence=float(box.conf))
                )
        return detections
