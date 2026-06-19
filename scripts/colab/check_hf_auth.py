#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Hugging Face authentication without printing the token.")
    parser.add_argument("--token-env", default="HF_TOKEN")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get(args.token_env, "")
    if not token:
        print(f"{args.token_env} is not set.", file=sys.stderr)
        return 1
    profile = HfApi(token=token).whoami()
    print(f"hf_user: {profile.get('name') or profile.get('fullname') or '<unknown>'}")
    print("hf_auth_ok: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
