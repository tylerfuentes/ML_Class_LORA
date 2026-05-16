#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from ibes_pipeline import (
    DEFAULT_RAW_IBES,
    build_bronze,
    build_gold,
    build_silver,
    dataset_date_range,
    format_size,
    key_missingness,
    load_raw_ibes,
    resolve_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the raw WRDS/IBES CSV and its bronze/silver/gold filters.")
    parser.add_argument("--input", default=str(DEFAULT_RAW_IBES), help="Path to the raw WRDS IBES CSV.")
    parser.add_argument("--json-out", help="Optional path to write the summary report as JSON.")
    args = parser.parse_args()

    path = resolve_path(args.input)
    if not path.exists():
        print(f"FAIL missing file: {path}")
        return 1

    raw = load_raw_ibes(path)
    bronze = build_bronze(raw)
    silver, silver_stats = build_silver(bronze)
    gold, gold_stats = build_gold(silver)

    raw_rows = len(raw)
    duplicate_pattern = int(
        bronze.duplicated(
            subset=["ticker_norm", "oftic_norm", "cusip8", "event_date", "VALUE", "ANALYS", "ESTIMATOR"]
        ).sum()
    )
    summary = {
        "input_path": str(path),
        "input_size_bytes": path.stat().st_size,
        "raw": {
            "row_count": int(raw_rows),
            "date_ranges": {
                "ACTDATS": dataset_date_range(bronze, "ACTDATS"),
                "REVDATS": dataset_date_range(bronze, "REVDATS"),
                "ANNDATS": dataset_date_range(bronze, "ANNDATS"),
                "event_date": dataset_date_range(bronze, "event_date"),
            },
            "unique_identifiers": {
                "ticker_norm": int(bronze["ticker_norm"].dropna().nunique()),
                "oftic_norm": int(bronze["oftic_norm"].dropna().nunique()),
                "cusip8": int(bronze["cusip8"].dropna().nunique()),
            },
            "missingness": key_missingness(
                bronze,
                [
                    "ticker_norm",
                    "oftic_norm",
                    "cusip8",
                    "event_date",
                    "VALUE",
                    "report_currency_norm",
                    "currfl_norm",
                    "FPEDATS",
                ],
            ),
            "duplicate_pattern_rows": duplicate_pattern,
        },
        "silver": silver_stats,
        "gold": gold_stats,
    }

    print("WRDS / IBES audit")
    print(f"- input: {path}")
    print(f"- size: {format_size(path.stat().st_size)}")
    print(f"- raw rows: {summary['raw']['row_count']:,}")
    print(f"- event date range: {summary['raw']['date_ranges']['event_date']['min']} -> {summary['raw']['date_ranges']['event_date']['max']}")
    print("- identifier coverage:")
    for key, value in summary["raw"]["unique_identifiers"].items():
        print(f"  - {key}: {value:,}")
    print("- missingness:")
    for key, value in summary["raw"]["missingness"].items():
        print(f"  - {key}: {value:.2%}")
    print(f"- duplicate pattern rows: {summary['raw']['duplicate_pattern_rows']:,}")
    print("- silver rows after filters:")
    for key, value in silver_stats.items():
        print(f"  - {key}: {value:,}" if isinstance(value, int) else f"  - {key}: {value}")
    print("- gold event stats:")
    for key, value in gold_stats.items():
        print(f"  - {key}: {value:,}" if isinstance(value, int) else f"  - {key}: {value}")

    if args.json_out:
        out_path = resolve_path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"- json report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
