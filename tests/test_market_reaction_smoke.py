#!/usr/bin/env python3
"""Smoke tests for schema-only market-reaction tooling using synthetic samples."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = REPO_ROOT / "data" / "samples" / "market_reaction"
SCRIPTS = REPO_ROOT / "scripts" / "market_reaction"


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class MarketReactionSmokeTests(unittest.TestCase):
    def test_check_market_data(self) -> None:
        result = run_script(
            str(SCRIPTS / "check_market_data.py"),
            "--crsp-daily-returns",
            str(SAMPLES / "crsp_daily_returns_sample.csv"),
            "--crsp-compustat-link",
            str(SAMPLES / "crsp_compustat_link_sample.csv"),
            "--market-benchmark-returns",
            str(SAMPLES / "market_benchmark_returns_sample.csv"),
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("[PASS] market-data validation completed", result.stdout)

    def test_build_event_panel_validation(self) -> None:
        result = run_script(
            str(SCRIPTS / "build_event_panel.py"),
            "--ibes-gold-events",
            str(SAMPLES / "ibes_gold_events_sample.csv"),
            "--crsp-daily-returns",
            str(SAMPLES / "crsp_daily_returns_sample.csv"),
            "--crsp-compustat-link",
            str(SAMPLES / "crsp_compustat_link_sample.csv"),
            "--market-benchmark-returns",
            str(SAMPLES / "market_benchmark_returns_sample.csv"),
            "--output-path",
            str(SAMPLES / "planned_event_panel.csv"),
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("validation completed only", result.stdout.lower())

    def test_compute_event_windows_validation(self) -> None:
        result = run_script(
            str(SCRIPTS / "compute_event_windows.py"),
            "--event-panel",
            str(SAMPLES / "event_panel_sample.csv"),
            "--crsp-daily",
            str(SAMPLES / "crsp_daily_returns_sample.csv"),
            "--benchmark-returns",
            str(SAMPLES / "market_benchmark_returns_sample.csv"),
            "--windows",
            "0:1",
            "0:3",
            "0:5",
            "-1:1",
            "--output-file",
            str(SAMPLES / "planned_event_windows.csv"),
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("raw_return_w_p0_p1", result.stdout)

    def test_score_label_alignment_validation(self) -> None:
        result = run_script(
            str(SCRIPTS / "score_label_alignment.py"),
            "--event-window-file",
            str(SAMPLES / "event_windows_sample.csv"),
            "--label-columns",
            "gold_direction_label",
            "adapter_1k_direction_label",
            "adapter_10k_direction_label",
            "--windows",
            "0:1",
            "0:3",
            "0:5",
            "-1:1",
            "--output-dir",
            str(SAMPLES / "planned_alignment_report"),
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("planned metrics", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
