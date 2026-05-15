#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.12}"
VENV_DIR="${REPO_ROOT}/.venv"

echo "[setup] repo root: ${REPO_ROOT}"
echo "[setup] python bin: ${PYTHON_BIN}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
pip install \
  "torch>=2.4" \
  "transformers>=4.57" \
  "accelerate>=0.34" \
  "peft>=0.17" \
  "trl>=0.20" \
  "datasets>=2.20" \
  "bitsandbytes>=0.43" \
  "safetensors>=0.4" \
  "huggingface_hub>=0.24" \
  "tensorboard>=2.17"

if [[ "${INSTALL_UNSLOTH:-0}" == "1" ]]; then
  echo "[warn] INSTALL_UNSLOTH=1 requested."
  echo "[warn] On this DGX ARM64 stack, a direct pip install of unsloth previously"
  echo "[warn] resolved a CPU-only torch build and broke CUDA visibility."
  echo "[warn] Use a separate experimental environment unless this has been re-validated."
fi

python - <<'PY'
import torch, transformers, peft, trl, datasets
print("[ok] torch:", torch.__version__)
print("[ok] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[ok] gpu:", torch.cuda.get_device_name(0))
    print("[ok] bf16 support:", torch.cuda.is_bf16_supported())
print("[ok] transformers:", transformers.__version__)
print("[ok] peft:", peft.__version__)
print("[ok] trl:", trl.__version__)
print("[ok] datasets:", datasets.__version__)
PY

echo "[done] environment ready at ${VENV_DIR}"
