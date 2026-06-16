#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_NAME="${1:-qwen36-27b-wrds-500k-unsloth-gb10}"
exec "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/start_wrds_unsloth_gb10_detached.py" --run-name "$RUN_NAME"
