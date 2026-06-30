"""検出器の共通インターフェース。

検出器は差し替え可能（mock / yolo / 将来の自社学習モデル）。
パイプラインはこのインターフェースにのみ依存する（疎結合）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..types import Detection


class Detector(ABC):
    """1フレームを受け取り検出結果のリストを返す。"""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        ...
