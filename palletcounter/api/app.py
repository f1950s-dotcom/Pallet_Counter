"""ダッシュボード兼確認/補正API（FR-11 / FR-12 / FR-13）。

追加ランタイム依存を避けるため標準ライブラリ http.server で実装する。

  GET  /                  ダッシュボードHTML
  GET  /api/summary       理論在庫サマリ(JSON)
  GET  /api/events        計数イベント一覧(JSON)
  GET  /api/pending       確認待ちキュー(JSON)
  POST /api/review        確認/補正  {id, count?, direction?}
  GET  /api/export.csv    確認済み計数のCSV出力（中間データ・8章）
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from ..events.store import EventStore
from ..inventory import InventoryService

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def make_handler(store: EventStore, opening_stock: int, site: str | None):
    inv = InventoryService(store, opening_stock=opening_stock)

    class Handler(BaseHTTPRequestHandler):
        # ログを静かに
        def log_message(self, *a):  # noqa: A003
            pass

        def _send(self, code: int, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/" or path == "/index.html":
                with open(os.path.join(_STATIC_DIR, "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif path == "/api/summary":
                s = inv.summary(site=site)
                self._json(s.__dict__)
            elif path == "/api/events":
                self._json(store.list(site=site))
            elif path == "/api/pending":
                self._json(store.pending_review(site=site))
            elif path == "/api/export.csv":
                body = store.export_csv(site=site).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header(
                    "Content-Disposition", "attachment; filename=pallet_events.csv"
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            path = urlparse(self.path).path
            if path != "/api/review":
                self._json({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                event_id = int(payload["id"])
            except (ValueError, KeyError, TypeError):
                self._json({"error": "id が必要です"}, 400)
                return
            count = payload.get("count")
            direction = payload.get("direction")
            store.review(
                event_id,
                count=int(count) if count is not None else None,
                direction=direction,
            )
            self._json({"ok": True, "id": event_id})

    return Handler


def serve(
    db_path: str = "data/db/events.sqlite3",
    host: str = "127.0.0.1",
    port: int = 8000,
    opening_stock: int = 0,
    site: str | None = None,
):
    store = EventStore(db_path)
    handler = make_handler(store, opening_stock, site)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"ダッシュボード: http://{host}:{port}/  (Ctrl-C で停止)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/events.sqlite3")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--opening", type=int, default=0)
    ap.add_argument("--site", default=None)
    a = ap.parse_args()
    serve(a.db, a.host, a.port, a.opening, a.site)
