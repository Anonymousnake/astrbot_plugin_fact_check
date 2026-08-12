from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from astrbot_plugin_fact_check.storage import (
    FactCheckMetricsStore,
    atomic_write_json,
    read_json_file,
)


class StorageMetricsTests(unittest.TestCase):
    def test_corrupt_json_is_preserved_before_falling_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.json"
            path.write_text("{broken", encoding="utf-8")

            self.assertEqual(read_json_file(path, {"fresh": True}), {"fresh": True})

            self.assertFalse(path.exists())
            self.assertEqual(len(list(Path(temp_dir).glob("state.json.corrupt-*"))), 1)

    def test_atomic_json_store_round_trips_without_leaving_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sessions.json"

            atomic_write_json(path, {"sessions": [{"session_id": "fc_abcd1234"}]})

            self.assertEqual(
                read_json_file(path),
                {"sessions": [{"session_id": "fc_abcd1234"}]},
            )
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_metrics_survive_restart_and_distinguish_partial_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.json"
            store = FactCheckMetricsStore(path)
            store.record(outcome="success", elapsed=2.0, cache_hit=True)
            store.record(outcome="partial", elapsed=4.0)
            store.record(outcome="failure", elapsed=6.0, failure_stage="evidence")
            store.record_delivery(success=True)
            store.record_delivery(success=False)

            reloaded = FactCheckMetricsStore(path)
            snapshot = reloaded.snapshot()

        self.assertEqual(snapshot["requests"], 3)
        self.assertEqual(snapshot["success"], 1)
        self.assertEqual(snapshot["partial"], 1)
        self.assertEqual(snapshot["failure"], 1)
        self.assertEqual(snapshot["cache_hits"], 1)
        self.assertEqual(snapshot["failure_stages"], {"evidence": 1})
        self.assertEqual(snapshot["delivery_success"], 1)
        self.assertEqual(snapshot["delivery_failure"], 1)
        self.assertEqual(snapshot["average_seconds"], 4.0)

    def test_metrics_status_contains_counts_but_no_request_content(self) -> None:
        store = FactCheckMetricsStore(None)
        store.record(outcome="partial", elapsed=1.5, followup=True)

        rendered = store.render_status()

        self.assertIn("部分完成：1", rendered)
        self.assertIn("追问：1", rendered)
        self.assertNotIn("request_text", rendered)


if __name__ == "__main__":
    unittest.main()
