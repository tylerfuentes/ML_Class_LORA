#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ("fiqa", "fpb", "tfns", "nwgi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-training evals for the Colab workflow.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "colab_paths.example.yaml")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--adapter-path", default="")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    training_cfg = cfg["training"]
    evaluation_cfg = cfg["evaluation"]
    model_cfg = cfg["model"]
    drive_cfg = cfg["drive"]
    datasets_cfg = cfg["datasets"]
    run_name = args.run_name or training_cfg["run_name"]

    adapter_path = Path(args.adapter_path) if args.adapter_path else Path(drive_cfg["adapters"]) / run_name
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    eval_root = Path(drive_cfg["outputs"]) / evaluation_cfg["output_subdir"] / run_name
    eval_root.mkdir(parents=True, exist_ok=True)
    holdout_file = Path(datasets_cfg["test_file"])
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_name": run_name,
        "adapter_path": str(adapter_path),
        "tasks": [],
    }

    tasks: list[tuple[str, list[str]]] = [
        (
            "ibes_test",
            [
                "--holdout-file",
                str(holdout_file),
            ],
        )
    ]
    tasks.extend((benchmark, ["--benchmark", benchmark]) for benchmark in BENCHMARKS)

    for task_name, task_args in tasks:
        output_dir = eval_root / task_name
        cmd = [
            sys.executable,
            str(REPO_ROOT / "eval" / "evaluate_base_vs_adapter.py"),
            "--model-id",
            model_cfg["model_id"],
            "--adapter-path",
            str(adapter_path),
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            str(evaluation_cfg["max_new_tokens"]),
            "--benchmark-split-size",
            str(evaluation_cfg["benchmark_split_size"]),
            "--qwen-thinking-mode",
            str(evaluation_cfg["qwen_thinking_mode"]),
        ]
        if model_cfg.get("local_files_only", False):
            cmd.append("--local-files-only")
        cmd.extend(task_args)
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            return result.returncode
        manifest["tasks"].append({"task_name": task_name, "output_dir": str(output_dir)})

    manifest_path = Path(drive_cfg["manifests"]) / f"{run_name}_evals.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
