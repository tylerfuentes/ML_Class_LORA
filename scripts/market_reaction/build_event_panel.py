#!/usr/bin/env python3
"""Validate event-panel inputs and print the planned join contract.

This tool is intentionally validation-first. It does not fabricate outputs and
does not attempt the full join until the required market datasets are present.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    CRSP_DAILY_ALIASES,
    CRSP_LINK_ALIASES,
    DEFAULT_BENCHMARK_RETURNS,
    DEFAULT_CRSP_DAILY_RETURNS,
    DEFAULT_CRSP_LINK,
    DEFAULT_EVENT_PANEL,
    DEFAULT_IBES_GOLD_EVENTS,
    IBES_IDENTIFIER_ALIASES,
    IBES_REQUIRED_ALIASES,
    ValidationIssue,
    inspect_tabular_file,
    print_inspection,
    require_columns,
    resolve_aliases,
    status_prefix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate schemas for the future market-reaction event-panel build."
    )
    parser.add_argument(
        "--ibes-gold-events",
        type=Path,
        default=DEFAULT_IBES_GOLD_EVENTS,
        help="Path to the IBES gold event file in CSV/TSV/JSONL format.",
    )
    parser.add_argument(
        "--crsp-daily-returns",
        type=Path,
        default=DEFAULT_CRSP_DAILY_RETURNS,
        help="Path to the CRSP daily returns export.",
    )
    parser.add_argument(
        "--crsp-compustat-link",
        type=Path,
        default=DEFAULT_CRSP_LINK,
        help="Path to the CRSP/Compustat link-table export.",
    )
    parser.add_argument(
        "--market-benchmark-returns",
        type=Path,
        default=DEFAULT_BENCHMARK_RETURNS,
        help="Optional benchmark returns file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_EVENT_PANEL,
        help="Planned output path for the normalized event panel.",
    )
    parser.add_argument("--sample-rows", type=int, default=3)
    return parser.parse_args()


def validate_ibes_events(path: Path, sample_rows: int) -> dict[str, str]:
    result = inspect_tabular_file(path, sample_rows=sample_rows)
    print_inspection("IBES gold events", result)
    if not result.exists:
        raise ValidationIssue(f"Missing IBES gold events file: {path}")
    mapping = require_columns(result.columns, IBES_REQUIRED_ALIASES, "IBES gold events")
    identifier_mapping, _ = resolve_aliases(result.columns, IBES_IDENTIFIER_ALIASES)
    if "gvkey" in identifier_mapping:
        print(f"{status_prefix('PASS')} join-ready identifier present: gvkey -> {identifier_mapping['gvkey']}")
    elif "cusip" in identifier_mapping:
        print(
            f"{status_prefix('WARN')} gvkey is missing; only cusip is available "
            f"({identifier_mapping['cusip']}). A pre-join enrichment step may still be required."
        )
    elif "ticker" in identifier_mapping:
        print(
            f"{status_prefix('WARN')} only ticker is available ({identifier_mapping['ticker']}). "
            "Ticker-only joins are diagnostic-only and not production-safe."
        )
    else:
        raise ValidationIssue(
            "IBES gold events must include at least one identifier column from {gvkey, cusip, ticker}."
        )
    return {**mapping, **identifier_mapping}


def validate_required_dataset(label: str, path: Path, aliases: dict[str, tuple[str, ...]], sample_rows: int) -> dict[str, str]:
    result = inspect_tabular_file(path, sample_rows=sample_rows)
    print_inspection(label, result)
    if not result.exists:
        raise ValidationIssue(f"Missing required dataset: {path}")
    return require_columns(result.columns, aliases, label)


def validate_optional_benchmark(path: Path, sample_rows: int) -> None:
    result = inspect_tabular_file(path, sample_rows=sample_rows)
    print_inspection("Market benchmark returns", result)
    if not result.exists:
        print(f"{status_prefix('WARN')} optional benchmark file not found: {path}")
        return
    print(f"{status_prefix('PASS')} optional benchmark file is present and can be validated separately.")


def print_join_plan(output_path: Path) -> None:
    print(f"{status_prefix('INFO')} planned output path: {output_path}")
    print(f"{status_prefix('INFO')} intended join logic:")
    print("  1. Normalize one row per IBES event with canonical event_id and event_date.")
    print("  2. Prefer gvkey -> link table -> permno when gvkey exists on the event data.")
    print("  3. If gvkey is absent but cusip exists, require an explicit enrichment step before production joins.")
    print("  4. Restrict link matches to rows where event_date falls within [linkdt, linkenddt].")
    print("  5. Carry join_key_type and join_status into the event panel for auditability.")
    print(f"{status_prefix('INFO')} expected event-panel columns:")
    print("  - event_id")
    print("  - event_date")
    print("  - gvkey")
    print("  - cusip")
    print("  - ticker")
    print("  - permno")
    print("  - label_direction")
    print("  - label_magnitude")
    print("  - join_key_type")
    print("  - join_status")
    print("  - crsp_daily_file")
    print(
        f"{status_prefix('WARN')} validation completed only. No output file was written because "
        "the join implementation is intentionally blocked from fabricating data."
    )


def main() -> int:
    args = parse_args()
    try:
        validate_ibes_events(args.ibes_gold_events, args.sample_rows)
        crsp_mapping = validate_required_dataset(
            "CRSP daily returns", args.crsp_daily_returns, CRSP_DAILY_ALIASES, args.sample_rows
        )
        link_mapping = validate_required_dataset(
            "CRSP/Compustat link", args.crsp_compustat_link, CRSP_LINK_ALIASES, args.sample_rows
        )
        print(f"{status_prefix('PASS')} CRSP daily returns columns: {crsp_mapping}")
        print(f"{status_prefix('PASS')} CRSP/Compustat link columns: {link_mapping}")
        validate_optional_benchmark(args.market_benchmark_returns, args.sample_rows)
        print_join_plan(args.output_path)
        return 0
    except ValidationIssue as exc:
        print(f"{status_prefix('FAIL')} {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
