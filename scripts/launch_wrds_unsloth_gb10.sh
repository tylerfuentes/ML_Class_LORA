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
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B}"
EPOCHS="${EPOCHS:-1.0}"
LR="${LR:-0.0002}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
EVAL_STEPS="${EVAL_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-25}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_TOTAL_EXAMPLES="${MAX_TOTAL_EXAMPLES:-0}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

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
  echo "[$(date --iso-8601=seconds)] eval_steps=$EVAL_STEPS save_steps=$SAVE_STEPS logging_steps=$LOGGING_STEPS"
  set +e
  cmd=(
    "$REPO_ROOT/.venv/bin/python" -u "$REPO_ROOT/training/train_finance_lora_unsloth.py"
    --model-id "$MODEL_ID"
    --train-file "$TRAIN_FILE"
    --eval-file "$TRAIN_EVAL_FILE"
    --test-file "$HOLDOUT_FILE"
    --output-dir "$OUTPUT_DIR"
    --epochs "$EPOCHS"
    --lr "$LR"
    --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --lora-dropout "$LORA_DROPOUT"
    --eval-steps "$EVAL_STEPS"
    --save-steps "$SAVE_STEPS"
    --logging-steps "$LOGGING_STEPS"
    --max-seq-length "$MAX_SEQ_LENGTH"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-total-examples "$MAX_TOTAL_EXAMPLES"
    --local-files-only
  )
  if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    cmd+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
  fi
  "${cmd[@]}"
  exit_code=$?
  set -e
  echo "$exit_code" >"$EXIT_CODE_PATH"
  echo "[$(date --iso-8601=seconds)] run $RUN_NAME exited with code $exit_code"
  exit "$exit_code"
} >>"$LOG_PATH" 2>&1
