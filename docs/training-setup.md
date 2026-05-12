# Qwen 3.6 LoRA Setup

This repo now includes a local training workflow for `Qwen/Qwen3.6-27B` on the DGX.

## Layout

- `training/`: environment bootstrap and training scripts
- `data/samples/`: starter finance JSONL data
- `outputs/`: ignored run outputs

## One-time environment setup

From the repo root:

```bash
bash training/setup_env.sh
source .venv/bin/activate
```

This creates a repo-local virtual environment at `.venv/`.

## Smoke run

Use this first to confirm the model can load and a LoRA adapter can be written:

```bash
source .venv/bin/activate
python training/train_smoke.py \
  --model-id Qwen/Qwen3.6-27B \
  --max-steps 1 \
  --max-seq-length 1024
```

Outputs land under:

```text
./outputs/qwen36-27b-finance-lora-smoke
```

## Validate the adapter

```bash
source .venv/bin/activate
python training/validate_adapter.py \
  --base-model Qwen/Qwen3.6-27B \
  --adapter-dir ./outputs/qwen36-27b-finance-lora-smoke
```

## Train on repo data

Starter sample data lives at:

```text
data/samples/finance_train.jsonl
```

Example run:

```bash
source .venv/bin/activate
python training/train_finance_lora.py \
  --model-id Qwen/Qwen3.6-27B \
  --train-file data/samples/finance_train.jsonl \
  --output-dir ./outputs/qwen36-27b-finance-lora-sample \
  --epochs 1 \
  --max-seq-length 1024
```

## Notes

- The base model already exists locally on this machine from the earlier setup.
- `outputs/` is gitignored and should hold adapters, checkpoints, and run summaries.
- This is a class-project baseline, not a fully tuned production pipeline.
