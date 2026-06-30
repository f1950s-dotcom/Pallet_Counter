"""合成映像とモック検出器で共有する色定義。

合成映像は各オブジェクトを既知の色で描画し、モック検出器は
その色域を抽出して検出する。これにより「画素から検出 -> 計数」という
エンドツーエンドの流れを依存ライブラリ最小で実機相当に再現する。

色相(H)が重ならないようにクラスごとに分離している。
色は BGR（OpenCVの既定）で定義する。
"""
from __future__ import annotations

from ..types import ObjectClass

# 各クラスの代表色 (BGR)。合成映像の描画に使用。
COLORS_BGR = {
    ObjectClass.FORKLIFT: (0, 140, 255),    # オレンジ  H~16
    ObjectClass.GOODS: (40, 200, 230),       # 黄        H~30
    ObjectClass.NONPALLET: (60, 170, 60),    # 緑        H~60（Pパレ以外＝計数対象外）
    ObjectClass.PPALLET: (200, 110, 30),     # 青        H~108（Pパレ）
}

# HSV色域 (OpenCV: H 0-179, S 0-255, V 0-255) — モック検出器の抽出範囲。
HSV_RANGES = {
    ObjectClass.FORKLIFT: ((8, 120, 120), (22, 255, 255)),
    ObjectClass.GOODS: ((24, 120, 120), (40, 255, 255)),
    ObjectClass.NONPALLET: ((45, 80, 80), (80, 255, 255)),
    ObjectClass.PPALLET: ((95, 120, 80), (125, 255, 255)),
}
