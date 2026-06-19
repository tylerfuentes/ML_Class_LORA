#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push only the final adapter artifacts to a private Hugging Face repo.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "colab_paths.example.yaml")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--enable-upload", action="store_true", default=False)
    parser.add_argument("--repo-id", default="")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    hf_cfg = cfg["huggingface"]
    drive_cfg = cfg["drive"]
    run_name = args.run_name or cfg["training"]["run_name"]
    adapter_path = Path(args.adapter_path) if args.adapter_path else Path(drive_cfg["adapters"]) / run_name
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    if not args.enable_upload:
        print("Upload disabled. Re-run with --enable-upload when you explicitly want to publish the final adapter.")
        return 0

    token = os.environ.get(hf_cfg["token_env"], "")
    if not token:
        print(f"Missing {hf_cfg['token_env']} in the environment; refusing upload.", file=sys.stderr)
        return 1

    repo_id = args.repo_id or hf_cfg["final_adapter_repo"]
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=bool(hf_cfg.get("private", True)), exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(adapter_path),
        commit_message=f"Upload final adapter for {run_name}",
        allow_patterns=[
            "README.md",
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "adapter_model.pt",
            "chat_template.jinja",
            "tokenizer.json",
            "tokenizer_config.json",
            "run_summary.json",
        ],
    )
    print(f"uploaded_repo={repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
