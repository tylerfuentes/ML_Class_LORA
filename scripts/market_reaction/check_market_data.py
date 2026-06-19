#!/usr/bin/env python3
"""Validate expected market-reaction input files without loading full datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    BENCHMARK_ALIASES,
    CRSP_DAILY_ALIASES,
    CRSP_DAILY_IDENTIFIER_ALIASES,
    CRSP_LINK_ALIASES,
    CRSP_STOCK_HEADER_ALIASES,
    DEFAULT_BENCHMARK_RETURNS,
    DEFAULT_CRSP_DAILY_RETURNS,
    DEFAULT_CRSP_LINK,
    DEFAULT_CRSP_STOCK_HEADER,
    ValidationIssue,
    inspect_tabular_file,
    print_inspection,
    require_columns,
    resolve_aliases,
    status_prefix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate CRSP/link/benchmark files needed for the market-reaction pipeline."
    )
    parser.add_argument("--crsp-daily-returns", type=Path, default=DEFAULT_CRSP_DAILY_RETURNS)
    parser.add_argument("--crsp-compustat-link", type=Path, default=DEFAULT_CRSP_LINK)
    parser.add_argument("--market-benchmark-returns", type=Path, default=DEFAULT_BENCHMARK_RETURNS)
    parser.add_argument("--crsp-stock-header", type=Path, default=DEFAULT_CRSP_STOCK_HEADER)
    parser.add_argument("--sample-rows", type=int, default=3)
    return parser.parse_args()


def validate_required_dataset(label: str, path: Path, aliases: dict[str, tuple[str, ...]], sample_rows: int) -> bool:
    result = inspect_tabular_file(path, sample_rows=sample_rows)
    print_inspection(label, result)
    if not result.exists:
        print(f"{status_prefix('FAIL')} required dataset missing: {path}")
        return False
    try:
        mapping = require_columns(result.columns, aliases, label)
    except ValidationIssue as exc:
        print(f"{status_prefix('FAIL')} {exc}")
        return False
    print(f"{status_prefix('PASS')} required columns present: {mapping}")
    return True


def validate_daily_identifier_fallback(path: Path, sample_rows: int) -> bool:
    result = inspect_tabular_file(path, sample_rows=sample_rows)
    if not result.exists:
        return False
    resolved, _ = resolve_aliases(result.columns, CRSP_DAILY_IDENTIFIER_ALIASES)
    if not resolved:
        print(
            f"{status_prefix('WARN')} CRSP daily returns file has core return columns but no identifier "
            "fallback columns such as cusip/ncusip/ticker/comnam."
        )
        return False
    print(f"{status_prefix('PASS')} CRSP daily identifier fallback columns present: {resolved}")
    return True


def validate_optional_dataset(label: str, path: Path, aliases: dict[str, tuple[str, ...]], sample_rows: int) -> bool:
    result = inspect_tabular_file(path, sample_rows=sample_rows)
    print_inspection(label, result)
    if not result.exists:
        print(f"{status_prefix('WARN')} optional dataset missing: {path}")
        return True
    try:
        mapping = require_columns(result.columns, aliases, label)
    except ValidationIssue as exc:
        print(f"{status_prefix('FAIL')} {exc}")
        return False
    print(f"{status_prefix('PASS')} optional columns present: {mapping}")
    return True


def main() -> int:
    args = parse_args()
    checks = [
        validate_required_dataset("CRSP daily returns", args.crsp_daily_returns, CRSP_DAILY_ALIASES, args.sample_rows),
        validate_optional_dataset(
            "Market benchmark returns",
            args.market_benchmark_returns,
            BENCHMARK_ALIASES,
            args.sample_rows,
        ),
        validate_optional_dataset(
            "CRSP stock header",
            args.crsp_stock_header,
            CRSP_STOCK_HEADER_ALIASES,
            args.sample_rows,
        ),
    ]
    link_ok = validate_required_dataset(
        "CRSP/Compustat link",
        args.crsp_compustat_link,
        CRSP_LINK_ALIASES,
        args.sample_rows,
    )
    fallback_ok = validate_daily_identifier_fallback(args.crsp_daily_returns, args.sample_rows)
    checks.append(link_ok or fallback_ok)
    if not link_ok and fallback_ok:
        print(
            f"{status_prefix('WARN')} CRSP/Compustat link table is missing, but fallback identifier columns "
            "exist in the CRSP daily file. Event-panel joins can proceed only in fallback mode."
        )
    if all(checks):
        print(f"{status_prefix('PASS')} market-data validation completed")
        return 0
    print(f"{status_prefix('FAIL')} market-data validation found blocking issues")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
