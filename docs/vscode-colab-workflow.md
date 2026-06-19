# VS Code + Colab Workflow

This project supports an edit-in-VS-Code and run-in-Colab workflow for A100 training.

## Goal

- edit notebooks, scripts, configs, and docs from VS Code
- execute the notebook on a Google Colab runtime through the VS Code Google Colab extension
- keep the notebook thin and push reusable logic into repo scripts

## Expected setup

1. Install the VS Code Google Colab extension.
2. Open this repository locally in VS Code.
3. Open [colab_a100_unsloth_qwen_finance.ipynb](/home/nathanaelguitar/ML_Class_LORA/notebooks/colab_a100_unsloth_qwen_finance.ipynb) from VS Code.
4. Connect the notebook to a Colab runtime from the extension UI.
5. Select an A100 runtime. For Qwen 27B, prefer the `A100 80GB` tier.

Authentication checklist:

- [colab-auth-checklist.md](/home/nathanaelguitar/ML_Class_LORA/docs/colab-auth-checklist.md)

## What Codex should edit

Safe to edit:

- `notebooks/*.ipynb` with outputs cleared
- `training/*.py`
- `eval/*.py`
- `scripts/colab/*.py`
- `config/*.yaml`
- `docs/*.md`

Safe to commit:

- code
- docs
- configs
- notebooks with cleared outputs
- tiny public or sample data already tracked under `data/public/` and `data/samples/`

Do not commit:

- WRDS or CRSP raw files
- gold JSONL training data
- notebook outputs
- adapters
- checkpoints
- eval outputs
- logs
- base model weights
- Hugging Face tokens or Colab secrets

## Notebook design rules

The notebook should only do:

- Google Drive mount
- repo clone/update
- runtime checks
- package installs
- small config inspection
- sanity-gate execution
- calls into repo scripts

Do not put large training loops or long reusable Python blocks directly in notebook cells. Keep that logic in:

- `scripts/colab/train_unsloth_from_config.py`
- `scripts/colab/run_post_training_evals.py`
- `scripts/colab/push_final_adapter.py`
- `scripts/colab/verify_runtime.py`

## Storage model

GitHub stores:

- source code
- docs
- configs
- notebooks with outputs cleared

Google Drive stores:

- gold JSONL data
- intermediate checkpoints
- logs
- eval outputs
- manifests
- copied final adapter artifacts

Hugging Face private model repo stores:

- final selected LoRA adapter weights only
- adapter metadata only

Do not upload WRDS, CRSP, or gold JSONL files to Hugging Face.

## Resume training

Training on Colab should write checkpoints to Drive under:

- `/content/drive/MyDrive/ML_Class_LORA/checkpoints/<run_name>/`

To resume:

1. reconnect the notebook to a suitable A100 runtime
2. remount Drive
3. rerun package setup
4. rerun the training helper with `--resume-latest` or `--resume-from-checkpoint`

The helper script keeps the training dataset on Drive and resumes from the last saved checkpoint instead of starting from zero.

## Sanity gate before real training

Do not start the full run first.

Run the sanity gate first:

- [scripts/colab/sanity_gate.py](/home/nathanaelguitar/ML_Class_LORA/scripts/colab/sanity_gate.py)

The sanity gate:

- runs a `1` to `5` step adapter train
- saves the adapter to Drive
- reloads the adapter
- generates on `5` examples

Only launch the long run after that gate succeeds.

## Private adapter upload

Private upload is opt-in.

Requirements:

- `HF_TOKEN` present in Colab secrets or environment
- explicit upload enablement
- private Hugging Face model repo

Recommended flow:

1. finish training
2. inspect eval outputs on Drive
3. decide which adapter is the selected final artifact
4. run the push helper with upload explicitly enabled

## Avoid committing notebook outputs

Before commit:

1. use VS Code notebook commands to clear all outputs
2. save the notebook
3. confirm the diff contains source changes only

This repo treats notebooks as source, not as result artifacts.
