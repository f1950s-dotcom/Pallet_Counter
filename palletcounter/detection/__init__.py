"""検出器ファクトリ。設定の detector 名から実装を生成する。"""
from __future__ import annotations

from .base import Detector
from .mock import MockDetector


def build_detector(name: str, **kwargs) -> Detector:
    name = (name or "mock").lower()
    if name == "mock":
        return MockDetector(**kwargs)
    if name == "yolo":
        from .yolo import YoloDetector

        return YoloDetector(**kwargs)
    if name == "openvocab":
        from .openvocab import OpenVocabDetector

        return OpenVocabDetector(**kwargs)
    raise ValueError(
        f"未知の検出器: {name!r}（'mock' / 'yolo' / 'openvocab'）"
    )


__all__ = ["Detector", "MockDetector", "build_detector"]
