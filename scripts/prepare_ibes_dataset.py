#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from ibes_pipeline import (
    BASELINE_10K,
    BASELINE_1K,
    DEFAULT_RAW_IBES,
    build_bronze,
    build_gold,
    build_lora_row,
    build_silver,
    format_size,
    load_raw_ibes,
    resolve_path,
    sample_gold_splits,
    write_jsonl,
)


def write_split_bundle(gold, out_dir, spec, seed: int) -> dict:
    split_frames = sample_gold_splits(gold, split=spec, seed=seed)
    jsonl_dir = out_dir / "jsonl" / spec.name
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    split_counts = {}
    for split_name, frame in split_frames.items():
        rows = [build_lora_row(row) for _, row in frame.iterrows()]
        split_counts[split_name] = write_jsonl(rows, jsonl_dir / f"{split_name}.jsonl")
    return split_counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Build bronze/silver/gold IBES tables and small LoRA/eval JSONL splits.")
    parser.add_argument("--input", default=str(DEFAULT_RAW_IBES), help="Path to the raw WRDS IBES CSV.")
    parser.add_argument(
        "--out",
        default="data/processed/ibes_lora_baseline",
        help="Output directory for bronze/silver/gold tables and JSONL splits.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split sampling.")
    parser.add_argument(
        "--neutral-abs-delta",
        type=float,
        default=0.005,
        help="Absolute consensus delta below which labels are treated as neutral.",
    )
    parser.add_argument(
        "--skip-10k",
        action="store_true",
        help="Only export the 1k baseline split even if there are enough gold events for the 10k split.",
    )
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    if not input_path.exists():
        print(f"FAIL missing file: {input_path}")
        return 1

    out_dir = resolve_path(args.out)
    bronze_dir = out_dir / "bronze"
    silver_dir = out_dir / "silver"
    gold_dir = out_dir / "gold"
    report_dir = out_dir / "reports"
    for directory in [bronze_dir, silver_dir, gold_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    raw = load_raw_ibes(input_path)
    bronze = build_bronze(raw)
    silver, silver_stats = build_silver(bronze)
    gold, gold_stats = build_gold(silver, neutral_abs_delta=args.neutral_abs_delta)

    bronze_path = bronze_dir / "ibes_bronze.parquet"
    silver_path = silver_dir / "ibes_eps_us_current.parquet"
    gold_path = gold_dir / "ibes_revision_events.parquet"
    bronze.to_parquet(bronze_path, index=False)
    silver.to_parquet(silver_path, index=False)
    gold.to_parquet(gold_path, index=False)

    split_reports: dict[str, dict] = {}
    split_reports[BASELINE_1K.name] = write_split_bundle(gold, out_dir, spec=BASELINE_1K, seed=args.seed)
    if not args.skip_10k and len(gold) >= BASELINE_10K.total:
        split_reports[BASELINE_10K.name] = write_split_bundle(
            gold, out_dir, spec=BASELINE_10K, seed=args.seed + 1
        )

    report = {
        "input_path": str(input_path),
        "input_size": format_size(input_path.stat().st_size),
        "rows": {
            "raw": int(len(raw)),
            "bronze": int(len(bronze)),
            "silver": int(len(silver)),
            "gold": int(len(gold)),
        },
        "silver_stats": silver_stats,
        "gold_stats": gold_stats,
        "artifacts": {
            "bronze": str(bronze_path),
            "silver": str(silver_path),
            "gold": str(gold_path),
        },
        "splits": split_reports,
    }
    report_path = report_dir / "ibes_pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("WRDS / IBES pipeline complete")
    print(f"- input: {input_path}")
    print(f"- bronze: {bronze_path}")
    print(f"- silver: {silver_path}")
    print(f"- gold: {gold_path}")
    print(f"- report: {report_path}")
    for split_name, counts in split_reports.items():
        print(f"- {split_name}: train={counts['train']}, eval={counts['eval']}, holdout={counts['holdout']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
