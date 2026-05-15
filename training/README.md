# Training

This directory contains the repo-local Qwen 3.6 LoRA/QLoRA workflow for the DGX.

## What is ready

- isolated repo-local Python environment via `training/setup_env.sh`
- smoke-test QLoRA training via `training/train_smoke.py`
- adapter validation via `training/validate_adapter.py`
- dataset-driven finance fine-tuning via `training/train_finance_lora.py`
- optional adapter merge via `training/merge_adapter.py`

## FinGPT-native prompt path

This repo now uses FinGPT's real benchmark prompt template at training time.

Source:

- `external/FinGPT/fingpt/FinGPT_Benchmark/utils.py`

Specifically:

- dataset rows stay in `instruction` / `input` / `output`
- `training/common.py` calls FinGPT's `get_prompt("default", ...)`
- Qwen is then fine-tuned on that prompt inside the model chat wrapper

That means the repo is no longer inventing a parallel local prompt format for FinGPT tasks.

## Baseline policy

The first real finance baseline should use a small clean split:

- `800` train
- `100` eval
- `100` holdout/test

Keep the existing `data/samples/finance_train.jsonl` file for smoke testing only.

Do not scale beyond `1,000` total examples until:

- the `r=16` adapter trains successfully
- adapter checkpoints save correctly
- the saved adapter reloads for inference

## What is not ready yet

- a final course dataset
- quantitative evaluation scripts
- experiment tracking beyond per-run JSON summaries

## Required data format

`train_finance_lora.py` expects JSONL rows with:

```json
{"instruction":"...","input":"...","output":"..."}
```

`input` may be empty, but `instruction` and `output` are required.

This matches the structure used by FinGPT benchmark datasets, so export can stay almost lossless.
