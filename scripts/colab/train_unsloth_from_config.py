#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_ADAPTER_FILES = (
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_model.pt",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "run_summary.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Unsloth training from a Colab/Drive YAML config.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "colab_paths.example.yaml")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--resume-latest", action="store_true", default=False)
    parser.add_argument("--max-steps-override", type=int, default=0)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    return path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_checkpoint(path: Path) -> Path | None:
    checkpoints = sorted(path.glob("checkpoint-*"), key=lambda item: int(item.name.split("-")[-1]))
    return checkpoints[-1] if checkpoints else None


def copy_final_adapter(output_dir: Path, adapters_root: Path, run_name: str) -> Path:
    adapter_dir = ensure_dir(adapters_root / run_name)
    for name in FINAL_ADAPTER_FILES:
        source = output_dir / name
        if source.exists():
            shutil.copy2(source, adapter_dir / name)
    return adapter_dir


def manifest_path(manifests_root: Path, run_name: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return manifests_root / f"{run_name}_{stamp}.json"


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)

    training_cfg = cfg["training"]
    datasets_cfg = cfg["datasets"]
    drive_cfg = cfg["drive"]
    model_cfg = cfg["model"]
    run_name = args.run_name or training_cfg["run_name"]

    checkpoints_root = ensure_dir(Path(drive_cfg["checkpoints"]))
    adapters_root = ensure_dir(Path(drive_cfg["adapters"]))
    outputs_root = ensure_dir(Path(drive_cfg["outputs"]))
    manifests_root = ensure_dir(Path(drive_cfg["manifests"]))

    train_file = ensure_file(Path(datasets_cfg["train_file"]))
    train_eval_file = ensure_file(Path(datasets_cfg["train_eval_file"]))
    test_file = ensure_file(Path(datasets_cfg["test_file"]))

    output_dir = checkpoints_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ensure_dir(outputs_root / "train_logs")
    log_path = log_dir / f"{run_name}.log"

    resume_from_checkpoint = args.resume_from_checkpoint.strip()
    if args.resume_latest and not resume_from_checkpoint:
        checkpoint = latest_checkpoint(output_dir)
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found under {output_dir} for --resume-latest.")
        resume_from_checkpoint = str(checkpoint)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "training" / "train_finance_lora_unsloth.py"),
        "--model-id",
        model_cfg["model_id"],
        "--train-file",
        str(train_file),
        "--eval-file",
        str(train_eval_file),
        "--test-file",
        str(test_file),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(training_cfg["epochs"]),
        "--lr",
        str(training_cfg["learning_rate"]),
        "--per-device-train-batch-size",
        str(training_cfg["per_device_train_batch_size"]),
        "--gradient-accumulation-steps",
        str(training_cfg["gradient_accumulation_steps"]),
        "--save-steps",
        str(training_cfg["save_steps"]),
        "--logging-steps",
        str(training_cfg["logging_steps"]),
        "--eval-steps",
        str(training_cfg["eval_steps"]),
        "--lora-r",
        str(training_cfg["lora_r"]),
        "--lora-alpha",
        str(training_cfg["lora_alpha"]),
        "--lora-dropout",
        str(training_cfg["lora_dropout"]),
        "--max-seq-length",
        str(training_cfg["max_seq_length"]),
        "--gpu-memory-utilization",
        str(training_cfg["gpu_memory_utilization"]),
        "--max-total-examples",
        str(training_cfg["max_total_examples"]),
    ]
    if model_cfg.get("local_files_only", False):
        cmd.append("--local-files-only")
    if training_cfg.get("disable_qlora", False):
        cmd.append("--disable-qlora")
    if resume_from_checkpoint:
        cmd.extend(["--resume-from-checkpoint", resume_from_checkpoint])
    if args.max_steps_override:
        cmd.extend(["--max-steps-override", str(args.max_steps_override)])

    print("Running:", " ".join(cmd))
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{datetime.now(UTC).isoformat()}] command: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        print(f"Training failed; see {log_path}", file=sys.stderr)
        return result.returncode

    adapter_dir = copy_final_adapter(output_dir, adapters_root, run_name)
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_name": run_name,
        "train_file": str(train_file),
        "train_eval_file": str(train_eval_file),
        "test_file": str(test_file),
        "output_dir": str(output_dir),
        "adapter_dir": str(adapter_dir),
        "log_path": str(log_path),
        "resume_from_checkpoint": resume_from_checkpoint or None,
        "model_id": model_cfg["model_id"],
    }
    path = manifest_path(manifests_root, run_name)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"adapter_dir={adapter_dir}")
    print(f"manifest={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
