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
    CRSP_DAILY_IDENTIFIER_ALIASES,
    CRSP_LINK_ALIASES,
    CRSP_STOCK_HEADER_ALIASES,
    DEFAULT_BENCHMARK_RETURNS,
    DEFAULT_CRSP_DAILY_RETURNS,
    DEFAULT_CRSP_LINK,
    DEFAULT_CRSP_STOCK_HEADER,
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
        "--crsp-stock-header",
        type=Path,
        default=DEFAULT_CRSP_STOCK_HEADER,
        help="Optional CRSP stock-header extract for non-CCM fallback joins.",
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


def validate_optional_dataset(label: str, path: Path, aliases: dict[str, tuple[str, ...]], sample_rows: int) -> dict[str, str]:
    result = inspect_tabular_file(path, sample_rows=sample_rows)
    print_inspection(label, result)
    if not result.exists:
        print(f"{status_prefix('WARN')} optional dataset missing: {path}")
        return {}
    mapping = require_columns(result.columns, aliases, label)
    print(f"{status_prefix('PASS')} optional columns present: {mapping}")
    return mapping


def fallback_identifier_mapping(daily_columns: list[str]) -> dict[str, str]:
    resolved, _ = resolve_aliases(daily_columns, CRSP_DAILY_IDENTIFIER_ALIASES)
    return resolved


def print_join_plan(output_path: Path, has_ccm: bool, has_daily_id_fallback: bool, has_stock_header: bool) -> None:
    print(f"{status_prefix('INFO')} planned output path: {output_path}")
    print(f"{status_prefix('INFO')} intended join logic:")
    print("  1. Normalize one row per IBES event with canonical event_id and event_date.")
    if has_ccm:
        print("  2. Prefer gvkey -> CRSP/Compustat link -> permno when gvkey exists on the event data.")
        print("  3. Restrict link matches to rows where event_date falls within [linkdt, linkenddt].")
    else:
        print("  2. CRSP/Compustat link is unavailable, so use fallback mapping in diagnostic mode.")
    if has_daily_id_fallback:
        print("  4. Fallback A: match IBES cusip/cusip8 against CRSP CUSIP or NCUSIP around the event date.")
    if has_stock_header:
        print("  5. Fallback B: use ticker/name history from CRSP stock header to audit or disambiguate joins.")
    print("  6. Carry join_key_type, join_method, join_confidence, and join_status into the event panel.")
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
    print("  - join_method")
    print("  - join_confidence")
    print("  - join_status")
    print("  - crsp_daily_file")
    print(
        f"{status_prefix('WARN')} validation completed only. No output file was written because "
        "the join implementation is intentionally blocked from fabricating data."
    )


def main() -> int:
    args = parse_args()
    try:
        ibes_mapping = validate_ibes_events(args.ibes_gold_events, args.sample_rows)
        crsp_mapping = validate_required_dataset(
            "CRSP daily returns", args.crsp_daily_returns, CRSP_DAILY_ALIASES, args.sample_rows
        )
        daily_result = inspect_tabular_file(args.crsp_daily_returns, sample_rows=args.sample_rows)
        daily_id_mapping = fallback_identifier_mapping(daily_result.columns)
        link_result = inspect_tabular_file(args.crsp_compustat_link, sample_rows=args.sample_rows)
        print_inspection("CRSP/Compustat link", link_result)
        link_mapping: dict[str, str] = {}
        if link_result.exists:
            link_mapping = require_columns(link_result.columns, CRSP_LINK_ALIASES, "CRSP/Compustat link")
        else:
            print(f"{status_prefix('WARN')} CRSP/Compustat link is unavailable: {args.crsp_compustat_link}")
        stock_header_mapping = validate_optional_dataset(
            "CRSP stock header", args.crsp_stock_header, CRSP_STOCK_HEADER_ALIASES, args.sample_rows
        )
        print(f"{status_prefix('PASS')} CRSP daily returns columns: {crsp_mapping}")
        if daily_id_mapping:
            print(f"{status_prefix('PASS')} CRSP daily identifier fallback columns: {daily_id_mapping}")
        if link_mapping:
            print(f"{status_prefix('PASS')} CRSP/Compustat link columns: {link_mapping}")
        elif not daily_id_mapping:
            raise ValidationIssue(
                "Missing CRSP/Compustat link and no fallback identifier columns found in the CRSP daily file."
            )
        elif "cusip" not in ibes_mapping and "ticker" not in ibes_mapping:
            raise ValidationIssue(
                "Fallback join mode requires at least one IBES identifier from {cusip, ticker}."
            )
        else:
            print(
                f"{status_prefix('WARN')} proceeding in fallback join mode only. Results will be less reliable "
                "than a gvkey->CCM join and should be treated as diagnostic until the link table exists."
            )
        if stock_header_mapping:
            print(f"{status_prefix('PASS')} CRSP stock header columns: {stock_header_mapping}")
        validate_optional_benchmark(args.market_benchmark_returns, args.sample_rows)
        print_join_plan(
            args.output_path,
            has_ccm=bool(link_mapping),
            has_daily_id_fallback=bool(daily_id_mapping),
            has_stock_header=bool(stock_header_mapping),
        )
        return 0
    except ValidationIssue as exc:
        print(f"{status_prefix('FAIL')} {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
