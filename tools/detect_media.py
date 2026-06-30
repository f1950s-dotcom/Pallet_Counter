"""自分の画像・動画を取り込んで検出/計数を試すツール。

  # 画像1枚（方向判定なし。写っているPパレ枚数/状態を推定）
  python tools/detect_media.py path/to/photo.jpg --detector openvocab

  # 動画（通過→計数まで。注釈動画も書き出し）
  python tools/detect_media.py path/to/clip.mp4 --detector openvocab -o out.mp4

検出器:
  mock      合成サンプル映像専用（本物の映像には使えない）
  yolo      学習済み .pt を --weights で指定（自社学習モデル＝本番想定）
  openvocab YOLO-World。テキスト指定でゼロショット。学習不要で本物を試せる
            （要 pip install ultralytics。本番精度は学習版が前提）

本物の映像は detector=openvocab か、学習済みweightsの yolo を使うこと。
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from palletcounter.classification.state import StateClassifier  # noqa: E402
from palletcounter.config import DockConfig  # noqa: E402
from palletcounter.detection import build_detector  # noqa: E402
from palletcounter.detection.palette import COLORS_BGR  # noqa: E402
from palletcounter.events.store import EventStore  # noqa: E402
from palletcounter.pipeline import CountingPipeline  # noqa: E402
from palletcounter.types import ObjectClass  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _draw(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        color = COLORS_BGR.get(d.cls, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{d.cls.value} {d.confidence:.2f}",
                    (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def run_image(path: str, detector, out: str) -> None:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    detections = detector.detect(img)

    # フォークリフトがあればその積荷領域で、無ければ画像全体で枚数推定
    forklifts = [d for d in detections if d.cls == ObjectClass.FORKLIFT]
    h, w = img.shape[:2]
    base_box = forklifts[0].bbox if forklifts else (0, 0, w, h)
    result = StateClassifier(DockConfig().counting).classify(base_box, detections)

    _draw(img, detections)
    banner = f"{result.state.value}  count={result.count}  conf={result.confidence}"
    cv2.rectangle(img, (0, 0), (w, 30), (20, 20, 20), -1)
    cv2.putText(img, banner, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(out, img)

    from collections import Counter
    print("検出:", dict(Counter(d.cls.value for d in detections)))
    print(f"状態: {result.state.value}  推定枚数: {result.count}  "
          f"信頼度: {result.confidence}  ({result.note})")
    print(f"注釈画像: {out}")


def run_video(path: str, detector_name: str, detector_kwargs: dict,
              out: str, db: str) -> None:
    config = DockConfig(detector=detector_name, db_path=db)
    if os.path.exists(db):
        os.remove(db)
    store = EventStore(db)
    pipeline = CountingPipeline(config, store=store, save_snapshots=False,
                               detector_kwargs=detector_kwargs)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    events_all = []
    banner, ttl = "", 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = pipeline.detector.detect(frame)
        events = pipeline.process_frame(frame)
        _draw(frame, detections)
        if pipeline._line_x is not None:
            cv2.line(frame, (pipeline._line_x, 0), (pipeline._line_x, h), (0, 0, 255), 2)
        for e in events:
            events_all.append(e)
            banner = (f"{e.direction.value} {e.state.value} +{e.count} "
                      f"conf={e.confidence} {'AUTO' if e.auto_confirmed else 'REVIEW'}")
            ttl = int(fps * 1.5)
        if ttl > 0:
            cv2.rectangle(frame, (0, 0), (w, 28), (20, 20, 20), -1)
            cv2.putText(frame, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            ttl -= 1
        writer.write(frame)
    cap.release()
    writer.release()

    print(f"確定イベント数: {len(events_all)}")
    for e in events_all:
        print(f"  {e.direction.value} {e.state.value} count={e.count} "
              f"conf={e.confidence} {'AUTO' if e.auto_confirmed else 'REVIEW'}")
    print(f"注釈動画: {out}")


def main():
    ap = argparse.ArgumentParser(description="自分の画像/動画で検出・計数を試す")
    ap.add_argument("media", help="画像 or 動画ファイル")
    ap.add_argument("--detector", default="openvocab",
                    choices=["mock", "yolo", "openvocab"])
    ap.add_argument("--weights", default=None, help="yolo/openvocab の重みパス")
    ap.add_argument("--conf", type=float, default=None, help="検出しきい値")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--db", default="data/db/trial.sqlite3")
    args = ap.parse_args()

    detector_kwargs = {}
    if args.weights:
        detector_kwargs["weights"] = args.weights
    if args.conf is not None:
        detector_kwargs["conf"] = args.conf

    ext = os.path.splitext(args.media)[1].lower()
    is_image = ext in IMAGE_EXT
    out = args.output or (
        os.path.splitext(args.media)[0] + ("_detected.png" if is_image else "_annotated.mp4")
    )

    if is_image:
        detector = build_detector(args.detector, **detector_kwargs)
        run_image(args.media, detector, out)
    else:
        run_video(args.media, args.detector, detector_kwargs, out, args.db)


if __name__ == "__main__":
    main()
