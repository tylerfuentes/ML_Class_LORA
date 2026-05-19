#!/usr/bin/env python3
"""Validate the label-alignment evaluation contract for market-reaction analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    DEFAULT_ALIGNMENT_REPORT,
    DEFAULT_EVENT_WINDOWS,
    DEFAULT_LABEL_COLUMNS,
    EVENT_WINDOWS_REQUIRED_BASE_ALIASES,
    ValidationIssue,
    abnormal_return_column,
    inspect_tabular_file,
    parse_window_spec,
    print_inspection,
    raw_return_column,
    require_columns,
    resolve_aliases,
    status_prefix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the planned evaluation inputs for label-vs-return alignment."
    )
    parser.add_argument("--event-window-file", type=Path, default=DEFAULT_EVENT_WINDOWS, help="Event-window panel file.")
    parser.add_argument(
        "--label-columns",
        nargs="+",
        default=DEFAULT_LABEL_COLUMNS,
        help="Label columns to evaluate, for example gold_direction_label adapter_1k_direction_label adapter_10k_direction_label.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        default=["0:1", "0:3", "0:5", "-1:1"],
        help="Event-window specs that must already exist as return columns in the input file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ALIGNMENT_REPORT,
        help="Directory for future label-alignment reports.",
    )
    parser.add_argument("--sample-rows", type=int, default=3)
    return parser.parse_args()


def validate_event_windows_file(path: Path, label_columns: list[str], windows: list[str], sample_rows: int) -> tuple[bool, list[str]]:
    result = inspect_tabular_file(path, sample_rows=sample_rows)
    print_inspection("Event-window file", result)
    if not result.exists:
        raise ValidationIssue(f"Missing event-window file: {path}")
    mapping = require_columns(result.columns, EVENT_WINDOWS_REQUIRED_BASE_ALIASES, "Event-window file")
    print(f"{status_prefix('PASS')} base columns: {mapping}")

    missing_labels = [column for column in label_columns if column not in result.columns]
    if missing_labels:
        raise ValidationIssue(f"Event-window file is missing label columns: {', '.join(missing_labels)}")

    missing_returns = [raw_return_column(spec) for spec in windows if raw_return_column(spec) not in result.columns]
    if missing_returns:
        raise ValidationIssue(f"Event-window file is missing required return columns: {', '.join(missing_returns)}")

    abnormal_columns = [abnormal_return_column(spec) for spec in windows]
    abnormal_available = all(column in result.columns for column in abnormal_columns)
    print(f"{status_prefix('PASS')} label columns present: {', '.join(label_columns)}")
    print(f"{status_prefix('PASS')} raw return columns present for windows: {', '.join(windows)}")
    if abnormal_available:
        print(f"{status_prefix('PASS')} abnormal-return columns available for all requested windows")
    else:
        print(f"{status_prefix('WARN')} abnormal-return columns are not available for every requested window")
    return abnormal_available, result.columns


def print_metric_plan(windows: list[str], label_columns: list[str], abnormal_available: bool, output_dir: Path) -> None:
    print(f"{status_prefix('INFO')} planned output directory: {output_dir}")
    print(f"{status_prefix('INFO')} planned metrics:")
    print("  - average return by predicted label")
    print("  - hit rate by predicted label")
    print("  - return spread between positive and negative labels")
    if abnormal_available:
        print("  - market-adjusted return by predicted label")
    else:
        print("  - market-adjusted return by predicted label (blocked until benchmark-adjusted columns exist)")
    print("  - simple long/avoid or long/short toy strategy metrics")
    print(f"{status_prefix('INFO')} label sources under evaluation: {', '.join(label_columns)}")
    print(f"{status_prefix('INFO')} windows under evaluation: {', '.join(windows)}")
    print(f"{status_prefix('WARN')} validation completed only. No report was written and no alpha claim is permitted.")


def main() -> int:
    args = parse_args()
    try:
        windows = [f"{start}:{end}" for start, end in (parse_window_spec(spec) for spec in args.windows)]
        abnormal_available, _ = validate_event_windows_file(
            args.event_window_file, args.label_columns, windows, args.sample_rows
        )
        print_metric_plan(windows, args.label_columns, abnormal_available, args.output_dir)
        return 0
    except ValidationIssue as exc:
        print(f"{status_prefix('FAIL')} {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
