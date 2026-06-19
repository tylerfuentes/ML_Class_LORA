#!/bin/bash
# Babysitter script to continue the tournament

RUNNER_PID=555526
REPO_DIR="/home/nathanaelguitar/ML_Class_LORA"
LOG_FILE="$REPO_DIR/babysitter.log"

echo "$(date): Babysitter started. Waiting for PID $RUNNER_PID..." >> "$LOG_FILE"

# Wait for the current runner to finish
while kill -0 $RUNNER_PID 2>/dev/null; do
    sleep 60
done

echo "$(date): Initial runner (PID $RUNNER_PID) finished." >> "$LOG_FILE"

# Check if high_quality_ibes_4k finished successfully
if [ -f "$REPO_DIR/outputs/overnight_tournament/runs/high_quality_ibes_4k/adapter/run_summary.json" ]; then
    echo "$(date): high_quality_ibes_4k completed successfully." >> "$LOG_FILE"
else
    echo "$(date): WARNING: high_quality_ibes_4k summary not found. Check logs." >> "$LOG_FILE"
fi

# Launch the rest of the tournament starting from balanced_ibes_10k
echo "$(date): Launching remaining tournament candidates..." >> "$LOG_FILE"
cd "$REPO_DIR"
./.venv/bin/python scripts/run_overnight_training_tournament.py --start-from balanced_ibes_10k >> "$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "$(date): New runner launched with PID $NEW_PID." >> "$LOG_FILE"
