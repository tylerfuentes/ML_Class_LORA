#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ibes_pipeline import DEFAULT_RAW_IBES, EXPECTED_IBES_COLUMNS, format_size, load_csv_header, resolve_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that the local WRDS/IBES CSV exists and matches the expected schema.")
    parser.add_argument("--input", default=str(DEFAULT_RAW_IBES), help="Path to the raw WRDS IBES CSV.")
    args = parser.parse_args()

    path = resolve_path(args.input)
    if not path.exists():
        print(f"FAIL missing file: {path}")
        return 1

    header = load_csv_header(path)
    missing = [column for column in EXPECTED_IBES_COLUMNS if column not in header]
    unexpected = [column for column in header if column not in EXPECTED_IBES_COLUMNS]

    print(f"PASS file exists: {path}")
    print(f"size: {format_size(path.stat().st_size)}")
    print(f"column_count: {len(header)}")
    print("columns:")
    for column in header:
        print(f"  - {column}")

    if missing:
        print(f"FAIL missing expected columns: {missing}")
        return 1

    if unexpected:
        print(f"WARN unexpected extra columns: {unexpected}")

    print("PASS WRDS/IBES header matches the expected schema.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
