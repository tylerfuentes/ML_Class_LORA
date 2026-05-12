# Training

This directory contains the repo-local Qwen 3.6 LoRA/QLoRA workflow for the DGX.

## What is ready

- isolated repo-local Python environment via `training/setup_env.sh`
- smoke-test QLoRA training via `training/train_smoke.py`
- adapter validation via `training/validate_adapter.py`
- dataset-driven finance fine-tuning via `training/train_finance_lora.py`
- optional adapter merge via `training/merge_adapter.py`

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
