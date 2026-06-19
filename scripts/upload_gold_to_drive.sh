#!/usr/bin/env bash
set -euo pipefail

# Uploads the gold JSONL training data to Google Drive so the Colab
# notebook can read it from /content/drive/MyDrive/ML_Class_LORA/data/gold/.
# Authenticate the rclone remote with the SAME Google account you authorize
# in the Colab drive.mount() popup.
#
# One-time setup:
#   sudo apt-get install -y rclone
#   rclone config
#     n) New remote -> name: gdrive -> storage: drive -> accept defaults -> auto config: yes
#
# Usage:
#   ./scripts/upload_gold_to_drive.sh

REMOTE_NAME="${RCLONE_REMOTE:-gdrive}"
DRIVE_DEST="${DRIVE_DEST:-ML_Class_LORA/data/gold}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$REPO_ROOT/data/processed/wrds_qwen_pipeline/jsonl"

FILES=(train.jsonl train_eval.jsonl val.jsonl test.jsonl)

if ! command -v rclone >/dev/null 2>&1; then
    echo "rclone not found. Install it first:" >&2
    echo "  sudo apt-get install -y rclone" >&2
    exit 1
fi

if ! rclone listremotes | grep -qx "${REMOTE_NAME}:"; then
    echo "rclone remote '${REMOTE_NAME}' is not configured." >&2
    echo "Run this once and follow the browser auth flow (use the same Google account you'll authorize in Colab):" >&2
    echo "  rclone config" >&2
    echo "    n) New remote -> name: ${REMOTE_NAME} -> storage: drive -> accept defaults -> auto config: yes" >&2
    exit 1
fi

for f in "${FILES[@]}"; do
    path="$SRC_DIR/$f"
    if [[ ! -f "$path" ]]; then
        echo "Missing expected file: $path" >&2
        exit 1
    fi
done

echo "Uploading to ${REMOTE_NAME}:${DRIVE_DEST}/ ..."
for f in "${FILES[@]}"; do
    rclone copy "$SRC_DIR/$f" "${REMOTE_NAME}:${DRIVE_DEST}/" --progress
done

echo
echo "Verifying remote contents:"
rclone lsl "${REMOTE_NAME}:${DRIVE_DEST}/"
