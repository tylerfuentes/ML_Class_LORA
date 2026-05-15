# Qwen 3.6 LoRA Setup

This repo includes a local finance QLoRA workflow for `Qwen/Qwen3.6-27B` on the DGX.

The current verified path is:

- FinGPT-native task/data structure
- repo-local Hugging Face / PEFT training
- 4-bit QLoRA adapters only

Related references:

- `docs/fingpt-integration.md`
- `docs/sources.md`

## Layout

- `training/`: environment bootstrap and training scripts
- `data/samples/`: starter finance JSONL data
- `outputs/`: ignored run outputs
- `external/FinGPT/`: upstream finance benchmark/data submodule

## One-time environment setup

From the repo root:

```bash
bash training/setup_env.sh
source .venv/bin/activate
```

This creates a repo-local virtual environment at `.venv/`.

## FinGPT-native data path

Before inventing local finance datasets, use FinGPT's real task structure where possible.

Make sure the submodule is present:

```bash
git submodule update --init --recursive
```

Export a small FinGPT slice for inspection or a baseline:

```bash
source .venv/bin/activate
python scripts/convert_fingpt_dataset.py \
  --dataset sentiment-cls \
  --split train \
  --max-rows 3 \
  --include-metadata \
  --output data/public/fingpt_sentiment_cls_sample.jsonl
```

The export stays in `instruction` / `input` / `output`.
At training time, `training/common.py` renders FinGPT's native prompt template through the submodule.

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

That sample file is for smoke testing and script validation only. Do not treat it as the first real finance baseline.

## First real baseline

For the first real finance QLoRA run, keep the dataset small and clean:

- `800` train examples
- `100` eval examples
- `100` holdout/test examples
- `1,000` total examples maximum

Do not scale beyond `1,000` examples until:

- the `r=16` adapter trains successfully
- the adapter saves correctly
- the adapter loads for inference without errors

Suggested paths:

```text
data/baseline/train.jsonl
data/baseline/eval.jsonl
data/baseline/test.jsonl
```

Example run:

```bash
source .venv/bin/activate
python training/train_finance_lora.py \
  --model-id Qwen/Qwen3.6-27B \
  --train-file data/baseline/train.jsonl \
  --eval-file data/baseline/eval.jsonl \
  --test-file data/baseline/test.jsonl \
  --output-dir ./outputs/qwen36-27b-finance-lora-baseline \
  --epochs 1 \
  --max-seq-length 1024
```

## Notes

- The base model already exists locally on this machine from the earlier setup.
- `outputs/` is gitignored and should hold adapters, checkpoints, and run summaries.
- This is a class-project baseline, not a fully tuned production pipeline.
- The trainer now defaults to a `1,000` example safety cap. Use `--max-total-examples 0` only if you intentionally want to override it later.
- Research direction notes and arXiv references live in `docs/research-directions.md`.
- `external/FinGPT/` is the upstream finance reference layer. Prefer its native task format before adding local wrappers.

## Unsloth note

Unsloth remains the intended acceleration path, but it is not the repo default environment today.

On this DGX's ARM64 stack, a direct `pip install unsloth` resolved a CPU-only `torch 2.10.0`, which broke CUDA visibility. The verified training environment was restored to `torch 2.12.0+cu130`.

So for now:

- verified training path: repo-local HF/PEFT QLoRA
- intended later path: isolated Unsloth environment once the ARM64/CUDA package resolution issue is solved cleanly

Official Unsloth references:

- https://docs.unsloth.ai/models/qwen3-how-to-run-and-fine-tune
- https://docs.unsloth.ai/basics/datasets-guide
