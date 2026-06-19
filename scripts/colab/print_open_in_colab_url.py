#!/usr/bin/env python3
from __future__ import annotations

from urllib.parse import quote


NOTEBOOK_PATH = "notebooks/colab_a100_unsloth_qwen_finance.ipynb"
REPO = "nathanaelguitar/ML_Class_LORA"
BRANCH = "main"


def build_url() -> str:
    path = quote(NOTEBOOK_PATH)
    return f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{path}"


def main() -> None:
    print(build_url())


if __name__ == "__main__":
    main()
