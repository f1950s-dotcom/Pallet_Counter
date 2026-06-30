"""検出・計数の可視化（検証用）。

入力動画を処理し、検出枠・計数ライン・確定イベントを重畳した動画を書き出す。
PoCのデモやアノテーション基準の確認に使う。

  python tools/annotate_video.py data/videos/sample.mp4 -o data/videos/annotated.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from palletcounter.config import DockConfig  # noqa: E402
from palletcounter.detection.palette import COLORS_BGR  # noqa: E402
from palletcounter.events.store import EventStore  # noqa: E402
from palletcounter.pipeline import CountingPipeline  # noqa: E402


def annotate(src: str, dst: str, db: str) -> str:
    config = DockConfig(db_path=db)
    store = EventStore(db)
    pipeline = CountingPipeline(config, store=store, save_snapshots=False)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise FileNotFoundError(src)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    total = 0
    banner = ""
    banner_ttl = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = pipeline.detector.detect(frame)
        events = pipeline.process_frame(frame)

        # 検出枠
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            color = COLORS_BGR.get(d.cls, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, d.cls.value, (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        # 計数ライン
        if pipeline._line_x is not None:
            cv2.line(frame, (pipeline._line_x, 0), (pipeline._line_x, h),
                     (0, 0, 255), 2)
        # 確定イベントのバナー
        for e in events:
            total += e.count if e.direction.value == "inbound" else -e.count
            banner = (f"{e.direction.value} {e.state.value} +{e.count} "
                      f"conf={e.confidence} "
                      f"{'AUTO' if e.auto_confirmed else 'REVIEW'}")
            banner_ttl = int(fps * 1.5)
        if banner_ttl > 0:
            cv2.rectangle(frame, (0, 0), (w, 28), (20, 20, 20), -1)
            cv2.putText(frame, banner, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            banner_ttl -= 1
        cv2.putText(frame, f"net={total}", (w - 120, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(frame)

    cap.release()
    writer.release()
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--output", default="data/videos/annotated.mp4")
    ap.add_argument("--db", default="data/db/annotate.sqlite3")
    args = ap.parse_args()
    if os.path.exists(args.db):
        os.remove(args.db)
    out = annotate(args.video, args.output, args.db)
    print(f"出力: {out}")
