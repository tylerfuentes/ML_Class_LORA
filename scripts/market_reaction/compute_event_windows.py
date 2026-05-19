#!/usr/bin/env python3
"""Validate event-window computation inputs and print the output contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    BENCHMARK_ALIASES,
    DEFAULT_BENCHMARK_RETURNS,
    DEFAULT_CRSP_DAILY_RETURNS,
    DEFAULT_EVENT_PANEL,
    DEFAULT_EVENT_WINDOWS,
    EVENT_PANEL_REQUIRED_ALIASES,
    ValidationIssue,
    abnormal_return_column,
    benchmark_return_column,
    inspect_tabular_file,
    parse_window_spec,
    print_inspection,
    raw_return_column,
    require_columns,
    status_prefix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the contract for future event-window return computation."
    )
    parser.add_argument("--event-panel", type=Path, default=DEFAULT_EVENT_PANEL, help="Normalized event panel file.")
    parser.add_argument("--crsp-daily", type=Path, default=DEFAULT_CRSP_DAILY_RETURNS, help="CRSP daily returns export.")
    parser.add_argument(
        "--benchmark-returns",
        type=Path,
        default=DEFAULT_BENCHMARK_RETURNS,
        help="Optional benchmark returns export for market-adjusted windows.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        default=["0:1", "0:3", "0:5", "-1:1"],
        help="Event-window specs such as 0:1 0:3 0:5 -1:1.",
    )
    parser.add_argument("--output-file", type=Path, default=DEFAULT_EVENT_WINDOWS, help="Output path for event-window labels.")
    parser.add_argument("--sample-rows", type=int, default=3)
    return parser.parse_args()


def validate_event_panel(path: Path, sample_rows: int) -> None:
    result = inspect_tabular_file(path, sample_rows=sample_rows)
    print_inspection("Event panel", result)
    if not result.exists:
        raise ValidationIssue(f"Missing event panel file: {path}")
    mapping = require_columns(result.columns, EVENT_PANEL_REQUIRED_ALIASES, "Event panel")
    print(f"{status_prefix('PASS')} event-panel columns: {mapping}")


def validate_crsp_daily(path: Path, sample_rows: int) -> None:
    from common import CRSP_DAILY_ALIASES

    result = inspect_tabular_file(path, sample_rows=sample_rows)
    print_inspection("CRSP daily returns", result)
    if not result.exists:
        raise ValidationIssue(f"Missing CRSP daily returns file: {path}")
    mapping = require_columns(result.columns, CRSP_DAILY_ALIASES, "CRSP daily returns")
    print(f"{status_prefix('PASS')} CRSP daily columns: {mapping}")


def validate_benchmark(path: Path, sample_rows: int) -> None:
    result = inspect_tabular_file(path, sample_rows=sample_rows)
    print_inspection("Market benchmark returns", result)
    if not result.exists:
        print(f"{status_prefix('WARN')} benchmark returns file not found: {path}")
        return
    mapping = require_columns(result.columns, BENCHMARK_ALIASES, "Market benchmark returns")
    print(f"{status_prefix('PASS')} benchmark columns: {mapping}")


def print_output_contract(windows: list[str], output_file: Path, benchmark_available: bool) -> None:
    print(f"{status_prefix('INFO')} planned output file: {output_file}")
    print(f"{status_prefix('INFO')} expected passthrough columns:")
    print("  - event_id")
    print("  - event_date")
    print("  - permno")
    print("  - label_direction")
    print("  - label_magnitude")
    print("  - join_key_type")
    print("  - join_status")
    print(f"{status_prefix('INFO')} expected event-window columns:")
    for spec in windows:
        print(f"  - {raw_return_column(spec)}")
        if benchmark_available:
            print(f"  - {benchmark_return_column(spec)}")
            print(f"  - {abnormal_return_column(spec)}")
    print(f"{status_prefix('WARN')} validation completed only. No output file was written.")


def main() -> int:
    args = parse_args()
    try:
        windows = [parse_window_spec(spec) for spec in args.windows]
        validated_specs = [f"{start}:{end}" for start, end in windows]
        validate_event_panel(args.event_panel, args.sample_rows)
        validate_crsp_daily(args.crsp_daily, args.sample_rows)
        benchmark_available = args.benchmark_returns.exists()
        validate_benchmark(args.benchmark_returns, args.sample_rows)
        print(f"{status_prefix('PASS')} validated windows: {', '.join(validated_specs)}")
        print_output_contract(validated_specs, args.output_file, benchmark_available)
        return 0
    except ValidationIssue as exc:
        print(f"{status_prefix('FAIL')} {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
