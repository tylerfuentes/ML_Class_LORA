# Colab A100 Storage Policy

This document defines what lives in GitHub, Google Drive, and Hugging Face for the Colab A100 workflow.

## Summary

GitHub:

- code
- docs
- configs
- notebooks with cleared outputs

Google Drive:

- gold JSONL data
- large eval outputs
- logs
- intermediate checkpoints
- local copies of final adapters
- manifests

Hugging Face private repo:

- final selected LoRA adapter weights only
- adapter metadata only

## Drive layout

Use this layout in Google Drive:

```text
/content/drive/MyDrive/ML_Class_LORA/
  data/gold/
    train.jsonl
    train_eval.jsonl
    val.jsonl
    test.jsonl
  adapters/
  checkpoints/
  outputs/
  manifests/
```

## GitHub policy

Allowed in GitHub:

- Python source
- shell scripts
- YAML config examples
- Markdown docs
- notebooks with outputs cleared
- tiny public samples

Forbidden in GitHub:

- WRDS/CRSP-derived gold JSONL
- adapter weights
- checkpoint folders
- eval output directories
- notebook outputs
- secrets
- logs

## Hugging Face policy

Allowed in Hugging Face private repo:

- `adapter_model.safetensors`
- `adapter_config.json`
- tokenizer metadata if needed for serving
- concise README / manifest metadata

Forbidden in Hugging Face:

- WRDS raw files
- CRSP raw files
- gold JSONL files
- parquet tables
- eval dumps
- training logs
- intermediate checkpoints unless there is a separate explicit private archival need

## Notebook policy

- notebooks are orchestration surfaces, not artifact stores
- clear all outputs before commit
- do not save training logs, stack traces, or generated text inside committed notebook outputs

## Local and Drive paths

Colab should read path defaults from:

- [config/colab_paths.example.yaml](/home/nathanaelguitar/ML_Class_LORA/config/colab_paths.example.yaml)

The active training and eval helpers for Colab are:

- [scripts/colab/train_unsloth_from_config.py](/home/nathanaelguitar/ML_Class_LORA/scripts/colab/train_unsloth_from_config.py)
- [scripts/colab/run_post_training_evals.py](/home/nathanaelguitar/ML_Class_LORA/scripts/colab/run_post_training_evals.py)
- [scripts/colab/push_final_adapter.py](/home/nathanaelguitar/ML_Class_LORA/scripts/colab/push_final_adapter.py)
- [scripts/colab/verify_runtime.py](/home/nathanaelguitar/ML_Class_LORA/scripts/colab/verify_runtime.py)

## Resume policy

When a Colab runtime disconnects:

- keep checkpoints on Drive
- reconnect to a fresh Colab runtime
- reinstall runtime packages
- resume from the latest checkpoint

Do not restart from scratch unless the checkpoint lineage is known bad.
