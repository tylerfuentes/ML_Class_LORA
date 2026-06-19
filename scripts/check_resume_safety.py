#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from safety import ensure_safe_output_dir, validate_resume_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that a Trainer LoRA checkpoint can be resumed safely."
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        required=True,
        help="Checkpoint directory to validate, for example outputs/.../checkpoint-500.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory to validate for overwrite safety.",
    )
    parser.add_argument(
        "--allow-overwrite-output-dir",
        action="store_true",
        default=False,
        help="Allow validation to pass even when output_dir already contains adapter artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = validate_resume_checkpoint(args.resume_from_checkpoint)
    print(f"[ok] checkpoint exists: {checkpoint}")
    print(f"[ok] adapter config exists: {checkpoint / 'adapter_config.json'}")

    weight_files = [path for path in checkpoint.iterdir() if path.name.startswith("adapter_model.")]
    for weight_file in weight_files:
        print(f"[ok] adapter weights exist: {weight_file}")

    print(f"[ok] trainer state exists: {checkpoint / 'trainer_state.json'}")

    if args.output_dir:
        output_dir = ensure_safe_output_dir(
            args.output_dir,
            resume_from_checkpoint=checkpoint,
            allow_overwrite_output_dir=args.allow_overwrite_output_dir,
        )
        print(f"[ok] output_dir passed overwrite safety check: {output_dir}")


if __name__ == "__main__":
    main()
