"""合成サンプル映像の生成。

実カメラ映像・学習データが無い段階で、パイプライン全体を
エンドツーエンドに検証するための合成シーンを生成する。

側面設置カメラ（要件6章）を模し、フォークリフトが画面を横切りながら
仮想計数ラインを通過するシーンを複数連結する。各シーンは積荷の状態を
変え、モック検出器が色域から検出できるよう既知色で描画する。

シナリオ（既定）:
  1. 入庫・空フォーク（0枚）
  2. 入庫・空パレット段積み3段（モードA, 3枚）
  3. 入庫・積載ダブルディープ（モードB, 2枚）
  4. 出庫・空パレット段積み8段（モードA, 8枚→高段で要確認）
  5. 出庫・積載 単一（モードB, 1枚）
  6. 入庫・非Pパレ通過（0枚, 計数対象外）
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

# パッケージ未インストールでも単体実行できるよう調整
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from palletcounter.detection.palette import COLORS_BGR  # noqa: E402
from palletcounter.types import ObjectClass  # noqa: E402


W, H = 960, 540
FPS = 20
GROUND_Y = 460
PALLET_W = 120
PALLET_H = 26


@dataclass
class Scene:
    name: str
    direction: str           # "inbound"(左->右) / "outbound"(右->左)
    state: str               # "empty" / "stack" / "loaded" / "nonpallet"
    layers: int = 0          # モードA 段数
    units: int = 1           # モードB ユニット数（ダブルディープ=2）
    frames: int = 60


DEFAULT_SCENES: List[Scene] = [
    Scene("入庫・空フォーク", "inbound", "empty"),
    Scene("入庫・段積み3段", "inbound", "stack", layers=3),
    Scene("入庫・ダブルディープ", "inbound", "loaded", units=2),
    Scene("出庫・段積み8段", "outbound", "stack", layers=8),
    Scene("出庫・積載単一", "outbound", "loaded", units=1),
    Scene("入庫・非Pパレ", "inbound", "nonpallet"),
]


BODY_W, BODY_H = 80, 110
LOAD_GAP = 30  # 車体と積荷の間隔（オレンジ車体が分割されないよう分離）


def _draw_forklift(img: np.ndarray, cx: int, direction: str) -> Tuple[int, int]:
    """フォークリフト車体を描画し、前方の積荷基準座標(load_cx, base_y)を返す。

    側面視点。フォークは進行方向側に伸び、積荷は車体と重ならない位置に載る。
    """
    color = COLORS_BGR[ObjectClass.FORKLIFT]
    bx1 = cx - BODY_W // 2
    by1 = GROUND_Y - BODY_H
    cv2.rectangle(img, (bx1, by1), (bx1 + BODY_W, GROUND_Y), color, -1)
    # 進行方向（front）側に積荷を配置
    front = 1 if direction == "inbound" else -1
    load_cx = cx + front * (BODY_W // 2 + LOAD_GAP + PALLET_W // 2)
    load_base_y = GROUND_Y - 8
    return load_cx, load_base_y


def _draw_pallet(img: np.ndarray, cx: int, base_y: int) -> None:
    color = COLORS_BGR[ObjectClass.PPALLET]
    x1 = cx - PALLET_W // 2
    y1 = base_y - PALLET_H
    cv2.rectangle(img, (x1, y1), (x1 + PALLET_W, base_y), color, -1)
    # 継ぎ目（段の境界）を暗線で描き、層を分離可能にする
    cv2.rectangle(img, (x1, y1), (x1 + PALLET_W, base_y), (0, 0, 0), 1)


def _draw_load(img: np.ndarray, scene: Scene, cx: int, base_y: int) -> None:
    if scene.state == "empty":
        return
    if scene.state == "nonpallet":
        color = COLORS_BGR[ObjectClass.NONPALLET]
        x1 = cx - PALLET_W // 2
        cv2.rectangle(img, (x1, base_y - 60), (x1 + PALLET_W, base_y), color, -1)
        return
    if scene.state == "stack":
        # 縦に layers 段。段ごとに隙間(継ぎ目)を空けて分離。
        y = base_y
        for _ in range(scene.layers):
            _draw_pallet(img, cx, y)
            y -= PALLET_H + 12  # 段間の継ぎ目を明確に分離
        return
    if scene.state == "loaded":
        # 基底に units 枚を横並び（ダブルディープ=側面から見て2枚分）＋荷物
        spacing = PALLET_W + 16
        start = cx - (scene.units - 1) * spacing // 2
        xs = [start + i * spacing for i in range(scene.units)]
        for x in xs:
            _draw_pallet(img, x, base_y)
        # 積荷（黄）をパレット上に
        goods = COLORS_BGR[ObjectClass.GOODS]
        gx1 = min(xs) - PALLET_W // 2
        gx2 = max(xs) + PALLET_W // 2
        cv2.rectangle(img, (gx1, base_y - PALLET_H - 70),
                      (gx2, base_y - PALLET_H), goods, -1)


def _render_scene(writer, scene: Scene) -> None:
    for f in range(scene.frames):
        img = np.full((H, W, 3), 235, np.uint8)  # 明るい背景
        # 床ライン
        cv2.line(img, (0, GROUND_Y), (W, GROUND_Y), (180, 180, 180), 2)
        t = f / max(1, scene.frames - 1)
        if scene.direction == "inbound":
            cx = int(60 + t * (W - 120))
        else:
            cx = int((W - 60) - t * (W - 120))
        load_cx, base_y = _draw_forklift(img, cx, scene.direction)
        _draw_load(img, scene, load_cx, base_y)
        writer.write(img)


def generate(path: str, scenes: List[Scene] = None) -> str:
    scenes = scenes or DEFAULT_SCENES
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter を開けません（コーデック未対応の可能性）")
    try:
        for scene in scenes:
            _render_scene(writer, scene)
    finally:
        writer.release()
    return path


def expected_counts(scenes: List[Scene] = None) -> List[dict]:
    """各シーンの正解枚数（テスト・評価の基準）。"""
    scenes = scenes or DEFAULT_SCENES
    out = []
    for s in scenes:
        if s.state == "stack":
            count = s.layers
        elif s.state == "loaded":
            count = s.units
        else:  # empty / nonpallet
            count = 0
        out.append({"name": s.name, "direction": s.direction, "count": count})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="合成サンプル映像を生成")
    ap.add_argument("-o", "--output", default="data/videos/sample.mp4")
    args = ap.parse_args()
    out = generate(args.output)
    print(f"生成: {out}")
    print("期待計数:")
    for e in expected_counts():
        print(f"  {e['name']:20s} {e['direction']:8s} count={e['count']}")
