# Google Colab Browser Quickstart

This repo already includes a Colab-ready notebook and Colab helper scripts.
The only part that cannot be automated from this machine is your Google sign-in.

Use this flow with the Google account:

- `iamrapidrocket@gmail.com`

## What to open

Primary notebook:

- [notebooks/colab_a100_unsloth_qwen_finance.ipynb](/home/nathanaelguitar/ML_Class_LORA/notebooks/colab_a100_unsloth_qwen_finance.ipynb)

Direct Colab URL pattern:

```text
https://colab.research.google.com/github/nathanaelguitar/ML_Class_LORA/blob/main/notebooks/colab_a100_unsloth_qwen_finance.ipynb
```

You can also print the current URL locally with:

```bash
python3 scripts/colab/print_open_in_colab_url.py
```

## Browser setup steps

1. Open `https://colab.research.google.com/`.
2. Sign in with `iamrapidrocket@gmail.com`.
3. Open the repo notebook from GitHub using the URL above.
4. In `Runtime -> Change runtime type`, choose `GPU`.
5. If your plan offers it, select `A100`, ideally `A100 80GB`.
6. Run the Drive mount cell and approve access for the same Google account.

Success signals:

- the notebook kernel is connected
- `nvidia-smi` shows an NVIDIA GPU
- `/content/drive/MyDrive/ML_Class_LORA/` exists after mount

## Required Drive layout

The default config expects:

- `/content/drive/MyDrive/ML_Class_LORA/data/gold/train.jsonl`
- `/content/drive/MyDrive/ML_Class_LORA/data/gold/train_eval.jsonl`
- `/content/drive/MyDrive/ML_Class_LORA/data/gold/val.jsonl`
- `/content/drive/MyDrive/ML_Class_LORA/data/gold/test.jsonl`

It also writes outputs under:

- `/content/drive/MyDrive/ML_Class_LORA/adapters`
- `/content/drive/MyDrive/ML_Class_LORA/checkpoints`
- `/content/drive/MyDrive/ML_Class_LORA/outputs`
- `/content/drive/MyDrive/ML_Class_LORA/manifests`

See:

- [config/colab_paths.example.yaml](/home/nathanaelguitar/ML_Class_LORA/config/colab_paths.example.yaml)

## Authentication notes

Google auth:

- required for Colab and Drive mount

Hugging Face auth:

- only required if model access or private adapter upload needs it
- set `HF_TOKEN` in Colab secrets or the runtime environment

Reference docs:

- [docs/colab-auth-checklist.md](/home/nathanaelguitar/ML_Class_LORA/docs/colab-auth-checklist.md)
- [docs/vscode-colab-workflow.md](/home/nathanaelguitar/ML_Class_LORA/docs/vscode-colab-workflow.md)

## Important project rules

- do not upload WRDS, CRSP, or Compustat data to GitHub or Hugging Face
- keep large datasets and outputs in Drive or ignored local folders
- do not mutate the repo `.venv` to test Unsloth locally
