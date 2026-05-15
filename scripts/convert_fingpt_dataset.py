#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys


def load_fingpt_loader():
    repo_root = Path(__file__).resolve().parents[1]
    benchmark_dir = repo_root / "external" / "FinGPT" / "fingpt" / "FinGPT_Benchmark"
    if not benchmark_dir.is_dir():
        raise SystemExit(
            "FinGPT submodule not found. Run: git submodule update --init --recursive"
        )
    benchmark_dir_str = str(benchmark_dir)
    if benchmark_dir_str not in sys.path:
        sys.path.insert(0, benchmark_dir_str)
    from utils import load_dataset  # type: ignore

    return load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="sentiment-cls",
        help="FinGPT dataset suffix, e.g. sentiment-cls or headline.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to convert.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional row limit. Use 0 for all rows in the split.",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include source_dataset and source_split fields in output rows.",
    )
    parser.add_argument(
        "--use-local-fingpt-cache",
        action="store_true",
        help="Load datasets from FinGPT's local saved-to-disk cache instead of Hugging Face.",
    )
    args = parser.parse_args()

    load_dataset = load_fingpt_loader()
    dataset_list = load_dataset(args.dataset, from_remote=not args.use_local_fingpt_cache)
    if len(dataset_list) != 1:
        raise SystemExit(
            "This export script expects exactly one FinGPT dataset name at a time. "
            f"Resolved datasets: {len(dataset_list)}"
        )

    dataset_dict = dataset_list[0]
    if args.split not in dataset_dict:
        raise SystemExit(
            f"Split '{args.split}' not found in FinGPT dataset '{args.dataset}'. "
            f"Available: {list(dataset_dict.keys())}"
        )

    dataset = dataset_dict[args.split]
    if args.max_rows and args.max_rows > 0:
        dataset = dataset.select(range(min(args.max_rows, len(dataset))))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            converted = {
                "instruction": row["instruction"],
                "input": row["input"],
                "output": row["output"],
            }
            if args.include_metadata:
                converted["source_dataset"] = f"FinGPT/fingpt-{args.dataset}"
                converted["source_split"] = args.split
            handle.write(json.dumps(converted, ensure_ascii=True) + "\n")
            count += 1

    print(f"Wrote {count} rows to {output_path}")


if __name__ == "__main__":
    main()
