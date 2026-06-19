#!/bin/bash
set -euo pipefail

REPO_DIR="/home/nathanaelguitar/ML_Class_LORA"
LOG_DIR="$REPO_DIR/admin/local/overnight-monitor"
LOG_FILE="$LOG_DIR/monitor.log"
PID="${1:-}"
INTERVAL_SECONDS="${2:-300}"

if [ -z "$PID" ]; then
  echo "usage: $0 <tournament_pid> [interval_seconds]" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" >> "$LOG_FILE"
}

snapshot() {
  local balanced_dir="$REPO_DIR/outputs/overnight_tournament/runs/balanced_ibes_10k"
  local adapter_dir="$balanced_dir/adapter"
  local summary="$adapter_dir/run_summary.json"
  local wrds_metrics="$balanced_dir/eval_wrds_holdout/metrics.json"
  local fiqa_metrics="$balanced_dir/eval_fiqa/metrics.json"
  local fpb_metrics="$balanced_dir/eval_fpb/metrics.json"
  local tfns_metrics="$balanced_dir/eval_tfns/metrics.json"
  local nwgi_metrics="$balanced_dir/eval_nwgi/metrics.json"

  local ps_line
  ps_line="$(ps -o pid,ppid,etime,pcpu,pmem,cmd -p "$PID" --no-headers 2>/dev/null || true)"
  if [ -n "$ps_line" ]; then
    log "process $ps_line"
  else
    log "process pid=$PID not found"
  fi

  local gpu_lines
  gpu_lines="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null | grep -E "(^$PID,|train_finance_lora.py|run_overnight_training_tournament.py)" || true)"
  if [ -n "$gpu_lines" ]; then
    while IFS= read -r line; do
      [ -n "$line" ] && log "gpu $line"
    done <<< "$gpu_lines"
  fi

  if [ -d "$adapter_dir" ]; then
    local file_count
    file_count="$(find "$adapter_dir" -maxdepth 2 -type f | wc -l | tr -d ' ')"
    log "adapter_dir files=$file_count path=$adapter_dir"
  fi

  for path in "$summary" "$wrds_metrics" "$fiqa_metrics" "$fpb_metrics" "$tfns_metrics" "$nwgi_metrics"; do
    if [ -f "$path" ]; then
      log "artifact present $path"
    else
      log "artifact missing $path"
    fi
  done
}

log "monitor started pid=$PID interval_seconds=$INTERVAL_SECONDS"
snapshot

while kill -0 "$PID" 2>/dev/null; do
  sleep "$INTERVAL_SECONDS"
  snapshot
done

log "process exited pid=$PID"
snapshot
