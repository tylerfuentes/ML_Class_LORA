#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_NAME="${1:-qwen36-27b-wrds-500k-unsloth-gb10}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/wrds_qwen_pipeline/train/$RUN_NAME}"
TRAIN_LOG="${TRAIN_LOG:-$REPO_ROOT/logs/wrds_qwen_pipeline/$RUN_NAME.log}"
TRAIN_STDOUT="${TRAIN_STDOUT:-$REPO_ROOT/logs/wrds_qwen_pipeline/$RUN_NAME.stdout}"
STATUS_LOG="${STATUS_LOG:-$REPO_ROOT/logs/wrds_qwen_pipeline/babysit_$RUN_NAME.log}"
BABYSIT_STDOUT="${BABYSIT_STDOUT:-$REPO_ROOT/logs/wrds_qwen_pipeline/babysit_$RUN_NAME.stdout}"
EVAL_ROOT="${EVAL_ROOT:-$REPO_ROOT/outputs/evals/wrds_qwen_pipeline/$RUN_NAME}"
REPORT_PATH="${REPORT_PATH:-$REPO_ROOT/docs/reports/$RUN_NAME.md}"
REPORT_JSON_PATH="${REPORT_JSON_PATH:-$REPO_ROOT/data/processed/wrds_qwen_pipeline/reports/$RUN_NAME.json}"
DOC_SUMMARY_PATH="${DOC_SUMMARY_PATH:-$REPO_ROOT/docs/overnight_run_summary.md}"
TRAIN_PID_PATH="$OUTPUT_DIR/train_wrapper.pid"
BABYSIT_PID_PATH="$OUTPUT_DIR/babysitter.pid"
EXIT_CODE_PATH="$OUTPUT_DIR/train.exitcode"
MANIFEST_PATH="$OUTPUT_DIR/launch_manifest.json"

mkdir -p "$(dirname "$TRAIN_LOG")" "$OUTPUT_DIR" "$EVAL_ROOT" "$(dirname "$REPORT_PATH")" "$(dirname "$REPORT_JSON_PATH")"

if [[ -f "$TRAIN_PID_PATH" ]]; then
  existing_train_pid="$(cat "$TRAIN_PID_PATH")"
  if kill -0 "$existing_train_pid" 2>/dev/null; then
    echo "refusing to start: existing train wrapper pid $existing_train_pid is still running" >&2
    exit 1
  fi
fi

if [[ -f "$BABYSIT_PID_PATH" ]]; then
  existing_babysit_pid="$(cat "$BABYSIT_PID_PATH")"
  if kill -0 "$existing_babysit_pid" 2>/dev/null; then
    echo "refusing to start: existing babysitter pid $existing_babysit_pid is still running" >&2
    exit 1
  fi
fi

rm -f "$TRAIN_STDOUT" "$BABYSIT_STDOUT" "$STATUS_LOG" "$EXIT_CODE_PATH"

nohup "$REPO_ROOT/scripts/launch_wrds_unsloth_gb10.sh" "$RUN_NAME" >"$TRAIN_STDOUT" 2>&1 &
train_pid=$!
echo "$train_pid" >"$TRAIN_PID_PATH"

nohup "$REPO_ROOT/.venv/bin/python" -u "$REPO_ROOT/scripts/babysit_wrds_qwen_run.py" \
  --train-pid "$train_pid" \
  --run-label "$RUN_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --train-log "$TRAIN_LOG" \
  --status-log "$STATUS_LOG" \
  --eval-root "$EVAL_ROOT" \
  --report-path "$REPORT_PATH" \
  --report-json-path "$REPORT_JSON_PATH" \
  --doc-summary-path "$DOC_SUMMARY_PATH" \
  --local-files-only \
  >"$BABYSIT_STDOUT" 2>&1 &
babysit_pid=$!
echo "$babysit_pid" >"$BABYSIT_PID_PATH"

cat >"$MANIFEST_PATH" <<EOF
{
  "run_name": "$RUN_NAME",
  "train_pid": $train_pid,
  "babysit_pid": $babysit_pid,
  "output_dir": "$OUTPUT_DIR",
  "train_log": "$TRAIN_LOG",
  "train_stdout": "$TRAIN_STDOUT",
  "status_log": "$STATUS_LOG",
  "babysit_stdout": "$BABYSIT_STDOUT",
  "eval_root": "$EVAL_ROOT",
  "report_path": "$REPORT_PATH",
  "report_json_path": "$REPORT_JSON_PATH",
  "train_exit_code_path": "$EXIT_CODE_PATH"
}
EOF

echo "started run $RUN_NAME"
echo "train_pid=$train_pid"
echo "babysit_pid=$babysit_pid"
echo "manifest=$MANIFEST_PATH"
