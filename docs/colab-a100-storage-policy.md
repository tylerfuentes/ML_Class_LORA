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

## Promotion to Hugging Face: decision rule

Hugging Face is the last step, not a parking spot for in-progress work.
Follow this order for every candidate adapter/checkpoint, including ones
produced on Colab:

1. **Back it up to Drive first.** Copy the checkpoint (adapter weights +
   `trainer_state.json` + optimizer/scheduler/RNG state if you may resume
   it) plus its training config, loss history, and exact resume/eval
   commands into a self-contained package under
   `Drive/MyDrive/ML_Class_LORA/local_backup/<run_name>_<checkpoint>/`.
   Do this before anything else touches the run — a crashed or interrupted
   training process does not mean the checkpoint is disposable.
2. **Verify it, don't assume it.** Reload the adapter and confirm:
   - it loads without error
   - it generates coherent, on-task output on a handful of examples
     (`eval/evaluate_base_vs_adapter.py` with a small `--max-examples`)
3. **Compare it against the existing reference adapters** before forming an
   opinion about quality — run the same holdout file, same
   `--max-new-tokens` (256+, not the 80 default — see
   [eval-findings.md](/home/nathanaelguitar/ML_Class_LORA/docs/eval-findings.md)
   for why 80 silently truncates thinking-mode output and looks like a
   failure when it is not), same `--max-examples`, against:
   - the prior `1k` adapter (`outputs/qwen36-27b-ibes-baseline`)
   - the prior `10k` adapter (`outputs/qwen36-27b-ibes-10k-controlled/checkpoint-500`)
   - base Qwen (the script does this automatically)
4. **Only push to Hugging Face if the comparison supports it**, and only the
   one selected adapter's final files (`adapter_model.safetensors`,
   `adapter_config.json`, tokenizer metadata, a short README) via
   `scripts/colab/push_final_adapter.py --enable-upload`. Never push full
   checkpoint history, optimizer state, or every candidate run.
5. **If it does not evaluate well, leave it archived in Drive.** Do not
   delete it and do not push it. A losing candidate is still useful history.

See [eval-findings.md](/home/nathanaelguitar/ML_Class_LORA/docs/eval-findings.md)
for worked examples of this comparison methodology and current adapter
standings.

## Resume policy

When a Colab runtime disconnects:

- keep checkpoints on Drive
- reconnect to a fresh Colab runtime
- reinstall runtime packages
- resume from the latest checkpoint

Do not restart from scratch unless the checkpoint lineage is known bad.
