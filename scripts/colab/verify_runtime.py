#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import sys

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify that the Colab runtime is suitable for Qwen 27B Unsloth training.")
    parser.add_argument("--require-gpu-substring", default="A100")
    parser.add_argument("--min-memory-gb", type=float, default=70.0)
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "<not installed>"


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA is not available; Colab runtime is not suitable.", file=sys.stderr)
        return 1

    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name
    total_gb = props.total_memory / (1024 ** 3)

    print(f"gpu_name: {gpu_name}")
    print(f"gpu_memory_gb: {total_gb:.2f}")
    print(f"torch_version: {torch.__version__}")
    print(f"torch_cuda_version: {torch.version.cuda}")
    print(f"cuda_bf16_supported: {torch.cuda.is_bf16_supported()}")
    print(f"unsloth_version: {package_version('unsloth')}")

    if args.require_gpu_substring.lower() not in gpu_name.lower():
        print(
            f"Refusing to continue: expected GPU containing '{args.require_gpu_substring}', got '{gpu_name}'.",
            file=sys.stderr,
        )
        return 2
    if total_gb < args.min_memory_gb:
        print(
            f"Refusing to continue: need at least {args.min_memory_gb:.1f} GB VRAM, found {total_gb:.2f} GB.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
