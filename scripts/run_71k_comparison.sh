#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

HOLDOUT=data/processed/wrds_qwen_pipeline/jsonl/test.jsonl
COMMON_ARGS=(--model-id Qwen/Qwen3.6-27B --holdout-file "$HOLDOUT" --qwen-thinking-mode both --max-examples 50 --max-new-tokens 256 --batch-size 1 --local-files-only)

echo "=== [1/3] checkpoint-4500 (71k, wrds-native adapter) ==="
.venv/bin/python eval/evaluate_base_vs_adapter.py \
  --adapter-path outputs/wrds_qwen_pipeline/train/qwen36-27b-wrds-500k-unsloth-gb10-rerun-20260616T2330Z/checkpoint-4500 \
  --output-dir outputs/evals/wrds_holdout_comparison/checkpoint-4500-71k \
  --run-label wrds-holdout-71k-vs-base \
  "${COMMON_ARGS[@]}"

echo "=== [2/3] ibes-baseline (1k adapter) on wrds holdout ==="
.venv/bin/python eval/evaluate_base_vs_adapter.py \
  --adapter-path outputs/qwen36-27b-ibes-baseline \
  --output-dir outputs/evals/wrds_holdout_comparison/ibes-baseline-1k \
  --run-label wrds-holdout-1k-vs-base \
  "${COMMON_ARGS[@]}"

echo "=== [3/3] ibes-10k-controlled checkpoint-500 (10k adapter) on wrds holdout ==="
.venv/bin/python eval/evaluate_base_vs_adapter.py \
  --adapter-path outputs/qwen36-27b-ibes-10k-controlled/checkpoint-500 \
  --output-dir outputs/evals/wrds_holdout_comparison/ibes-10k-controlled \
  --run-label wrds-holdout-10k-vs-base \
  "${COMMON_ARGS[@]}"

echo "=== all done ==="
