"""CLIエントリポイント。

  python -m palletcounter run    <video>   # 動画を処理して計数
  python -m palletcounter summary          # 理論在庫サマリ
  python -m palletcounter export           # 計数結果をCSV出力
"""
from __future__ import annotations

import argparse
import sys

from .config import DockConfig
from .events.store import EventStore
from .inventory import InventoryService
from .pipeline import CountingPipeline


def _cmd_run(args) -> int:
    config = DockConfig(site_id=args.site, detector=args.detector, db_path=args.db)
    pipeline = CountingPipeline(config, save_snapshots=not args.no_snapshots)
    events = pipeline.run_video(args.video, progress=True)
    print(f"\n確定イベント数: {len(events)}")
    inv = InventoryService(pipeline.store, opening_stock=args.opening)
    s = inv.summary(site=args.site)
    print(
        f"在庫サマリ[{s.site}] 期首={s.opening} 入庫={s.inbound} "
        f"出庫={s.outbound} 理論在庫={s.theoretical} 確認待ち={s.pending_review}"
    )
    return 0


def _cmd_summary(args) -> int:
    store = EventStore(args.db)
    inv = InventoryService(store, opening_stock=args.opening)
    s = inv.summary(site=args.site)
    print(
        f"在庫サマリ[{s.site}] 期首={s.opening} 入庫={s.inbound} "
        f"出庫={s.outbound} 理論在庫={s.theoretical} "
        f"イベント={s.events} 確認待ち={s.pending_review}"
    )
    return 0


def _cmd_export(args) -> int:
    store = EventStore(args.db)
    csv_text = store.export_csv(site=args.site)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(csv_text)
        print(f"出力: {args.output}")
    else:
        sys.stdout.write(csv_text)
    return 0


def _cmd_dashboard(args) -> int:
    from .api import serve

    serve(
        db_path=args.db,
        host=args.host,
        port=args.port,
        opening_stock=args.opening,
        site=None if args.site == "ALL" else args.site,
    )
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="palletcounter")
    ap.add_argument("--db", default="data/db/events.sqlite3")
    ap.add_argument("--site", default="DOCK-01")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="動画を処理して計数")
    r.add_argument("video")
    r.add_argument("--detector", default="mock", choices=["mock", "yolo"])
    r.add_argument("--opening", type=int, default=0, help="期首在庫")
    r.add_argument("--no-snapshots", action="store_true")
    r.set_defaults(func=_cmd_run)

    s = sub.add_parser("summary", help="理論在庫サマリ")
    s.add_argument("--opening", type=int, default=0)
    s.set_defaults(func=_cmd_summary)

    e = sub.add_parser("export", help="計数結果をCSV出力")
    e.add_argument("-o", "--output")
    e.set_defaults(func=_cmd_export)

    d = sub.add_parser("dashboard", help="ダッシュボード/確認UIを起動")
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--port", type=int, default=8000)
    d.add_argument("--opening", type=int, default=0)
    d.set_defaults(func=_cmd_dashboard)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
