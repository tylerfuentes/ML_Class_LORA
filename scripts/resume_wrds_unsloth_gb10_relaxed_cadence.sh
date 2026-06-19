#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_NAME="${1:-qwen36-27b-wrds-500k-unsloth-gb10-rerun-20260616T2330Z}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/wrds_qwen_pipeline/train/$RUN_NAME}"
TRAIN_FILE="${TRAIN_FILE:-$REPO_ROOT/data/processed/wrds_qwen_pipeline/jsonl/train.jsonl}"
TRAIN_EVAL_FILE="${TRAIN_EVAL_FILE:-$REPO_ROOT/data/processed/wrds_qwen_pipeline/jsonl/train_eval.jsonl}"
HOLDOUT_FILE="${HOLDOUT_FILE:-$REPO_ROOT/data/processed/wrds_qwen_pipeline/jsonl/test.jsonl}"

latest_checkpoint="$(
  find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
    | sort -t- -k2,2n \
    | tail -n 1
)"

if [[ -z "$latest_checkpoint" ]]; then
  echo "no checkpoint found under $OUTPUT_DIR" >&2
  exit 1
fi

export OUTPUT_DIR
export TRAIN_FILE
export TRAIN_EVAL_FILE
export HOLDOUT_FILE
export RESUME_FROM_CHECKPOINT="$OUTPUT_DIR/$latest_checkpoint"
export EVAL_STEPS="${EVAL_STEPS:-5000}"
export SAVE_STEPS="${SAVE_STEPS:-5000}"
export LOGGING_STEPS="${LOGGING_STEPS:-100}"
export MAX_TOTAL_EXAMPLES="${MAX_TOTAL_EXAMPLES:-0}"

echo "resuming $RUN_NAME from $RESUME_FROM_CHECKPOINT"
echo "train_file=$TRAIN_FILE"
echo "eval_steps=$EVAL_STEPS save_steps=$SAVE_STEPS logging_steps=$LOGGING_STEPS"

exec "$REPO_ROOT/scripts/start_wrds_unsloth_gb10_detached.sh" "$RUN_NAME"
