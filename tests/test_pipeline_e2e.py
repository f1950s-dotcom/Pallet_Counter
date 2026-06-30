"""エンドツーエンド統合テスト。

合成映像を生成し、パイプラインを通した計数結果が各シーンの正解
（generate_sample_video.expected_counts）と一致することを検証する。
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from palletcounter.config import DockConfig
from palletcounter.events.store import EventStore
from palletcounter.pipeline import CountingPipeline
from tools.generate_sample_video import DEFAULT_SCENES, expected_counts, generate


def test_pipeline_matches_expected_counts():
    tmp = tempfile.mkdtemp()
    video = os.path.join(tmp, "sample.mp4")
    generate(video)

    config = DockConfig(
        site_id="DOCK-TEST",
        db_path=os.path.join(tmp, "events.sqlite3"),
    )
    store = EventStore(config.db_path)
    pipeline = CountingPipeline(config, store=store, save_snapshots=False)
    events = pipeline.run_video(video)

    expected = expected_counts(DEFAULT_SCENES)
    # シーン数ぶんのイベントが立つ（1通過=1イベント）
    assert len(events) == len(expected), (
        f"イベント数 {len(events)} != シーン数 {len(expected)}"
    )
    for ev, exp in zip(events, expected):
        assert ev.direction.value == exp["direction"], exp["name"]
        assert ev.count == exp["count"], (
            f"{exp['name']}: 期待 {exp['count']} != 実際 {ev.count}"
        )
