#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
from importlib import metadata


TORCH_VERSION = "2.10.0"
TORCHVISION_VERSION = "0.25.0"
UNSLOTH_VERSION = "2026.6.7"
TRANSFORMERS_VERSION = "5.5.0"
DATASETS_VERSION = "4.3.0"
TRL_VERSION = "0.24.0"
PEFT_VERSION = "0.19.1"
BITSANDBYTES_VERSION = "0.49.2"
ACCELERATE_VERSION = "1.13.0"
HUGGINGFACE_HUB_VERSION = "1.14.0"

SUPPORTED_CUDA_TAGS = {"cu124", "cu126", "cu128", "cu130"}
DEFAULT_CUDA_TAG = "cu128"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def detect_cuda_tag() -> str:
    torch_version = installed_version("torch")
    if torch_version:
        match = re.search(r"\+(cu\d+)$", torch_version)
        if match and match.group(1) in SUPPORTED_CUDA_TAGS:
            return match.group(1)
    return DEFAULT_CUDA_TAG


def pip_install(*args: str) -> None:
    run([sys.executable, "-m", "pip", "install", "-q", *args])


def main() -> int:
    cuda_tag = detect_cuda_tag()
    print(f"Using PyTorch wheel index for {cuda_tag}")

    pip_install("--upgrade", "pip", "setuptools", "wheel")

    # Install torch and torchvision together from one CUDA-specific index so
    # Colab does not keep a stale torchvision after downgrading torch for
    # Unsloth's declared torch<2.11 compatibility window.
    pip_install(
        "--force-reinstall",
        "--index-url",
        f"https://download.pytorch.org/whl/{cuda_tag}",
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
    )

    # Keep the rest of the stack inside Unsloth's published compatibility
    # bounds instead of installing newer versions and overriding them later.
    pip_install(
        "--upgrade",
        "--force-reinstall",
        f"pyyaml",
        f"unsloth=={UNSLOTH_VERSION}",
        f"transformers=={TRANSFORMERS_VERSION}",
        f"datasets=={DATASETS_VERSION}",
        f"trl=={TRL_VERSION}",
        f"peft=={PEFT_VERSION}",
        f"bitsandbytes=={BITSANDBYTES_VERSION}",
        f"accelerate=={ACCELERATE_VERSION}",
        f"huggingface_hub=={HUGGINGFACE_HUB_VERSION}",
    )

    verify = """
import importlib.metadata
import torch
for name in [
    "unsloth",
    "transformers",
    "datasets",
    "trl",
    "peft",
    "accelerate",
    "bitsandbytes",
    "huggingface_hub",
]:
    print(f"{name}={importlib.metadata.version(name)}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"bf16={torch.cuda.is_bf16_supported()}")
"""
    run([sys.executable, "-c", verify])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
