#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_NAME="${1:-qwen36-27b-wrds-500k-unsloth-gb10}"
TRAIN_FILE="${TRAIN_FILE:-$REPO_ROOT/data/processed/wrds_qwen_pipeline/jsonl/train.jsonl}"
TRAIN_EVAL_FILE="${TRAIN_EVAL_FILE:-$REPO_ROOT/data/processed/wrds_qwen_pipeline/jsonl/train_eval.jsonl}"
HOLDOUT_FILE="${HOLDOUT_FILE:-$REPO_ROOT/data/processed/wrds_qwen_pipeline/jsonl/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/wrds_qwen_pipeline/train/$RUN_NAME}"
LOG_PATH="${LOG_PATH:-$REPO_ROOT/logs/wrds_qwen_pipeline/$RUN_NAME.log}"
EXIT_CODE_PATH="${EXIT_CODE_PATH:-$OUTPUT_DIR/train.exitcode}"

mkdir -p "$(dirname "$LOG_PATH")" "$OUTPUT_DIR"
rm -f "$EXIT_CODE_PATH"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

{
  echo "[$(date --iso-8601=seconds)] starting run $RUN_NAME"
  echo "[$(date --iso-8601=seconds)] output_dir=$OUTPUT_DIR"
  echo "[$(date --iso-8601=seconds)] log_path=$LOG_PATH"
  set +e
  "$REPO_ROOT/.venv/bin/python" -u "$REPO_ROOT/training/train_finance_lora_unsloth.py" \
    --model-id Qwen/Qwen3.6-27B \
    --train-file "$TRAIN_FILE" \
    --eval-file "$TRAIN_EVAL_FILE" \
    --test-file "$HOLDOUT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 1.0 \
    --lr 0.0002 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 4 \
    --lora-dropout 0.0 \
    --eval-steps 500 \
    --save-steps 500 \
    --logging-steps 25 \
    --max-seq-length 1024 \
    --gpu-memory-utilization 0.9 \
    --max-total-examples 0 \
    --local-files-only
  exit_code=$?
  set -e
  echo "$exit_code" >"$EXIT_CODE_PATH"
  echo "[$(date --iso-8601=seconds)] run $RUN_NAME exited with code $exit_code"
  exit "$exit_code"
} >>"$LOG_PATH" 2>&1
