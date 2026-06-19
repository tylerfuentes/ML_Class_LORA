#!/bin/bash
set -euo pipefail
REPO_DIR="/home/nathanaelguitar/ML_Class_LORA"
LOG_FILE="$REPO_DIR/babysitter.log"
WAIT_PID="${1:-}"
BALANCED_RUN_DIR="$REPO_DIR/outputs/overnight_tournament/runs/balanced_ibes_10k"
if [ -z "$WAIT_PID" ]; then
  echo "usage: $0 <pid>" >&2
  exit 1
fi
{
  echo "$(date '+%Y-%m-%d %H:%M:%S %Z'): babysitter waiting on PID $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
  echo "$(date '+%Y-%m-%d %H:%M:%S %Z'): initial runner finished"
  required_files=(
    "$BALANCED_RUN_DIR/adapter/run_summary.json"
    "$BALANCED_RUN_DIR/eval_wrds_holdout/metrics.json"
    "$BALANCED_RUN_DIR/eval_fiqa/metrics.json"
    "$BALANCED_RUN_DIR/eval_fpb/metrics.json"
    "$BALANCED_RUN_DIR/eval_tfns/metrics.json"
    "$BALANCED_RUN_DIR/eval_nwgi/metrics.json"
  )
  for path in "${required_files[@]}"; do
    if [ ! -f "$path" ]; then
      echo "$(date '+%Y-%m-%d %H:%M:%S %Z'): balanced_ibes_10k did not finish cleanly; missing $path"
      exit 1
    fi
  done
  echo "$(date '+%Y-%m-%d %H:%M:%S %Z'): balanced_ibes_10k artifacts verified; evaluating stop rules before any resume"
  cd "$REPO_DIR"
  ./.venv/bin/python scripts/run_overnight_training_tournament.py --start-from mixed_finance_10k >> "$LOG_FILE" 2>&1
  echo "$(date '+%Y-%m-%d %H:%M:%S %Z'): resume runner completed"
} >> "$LOG_FILE" 2>&1
