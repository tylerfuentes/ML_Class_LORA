# Colab Auth Checklist

Use this checklist before running the Colab notebook through the VS Code Google Colab extension.

## 1. VS Code Colab extension connection

Manual steps:

1. Open VS Code.
2. Install and enable the Google Colab extension.
3. Open [colab_a100_unsloth_qwen_finance.ipynb](/home/nathanaelguitar/ML_Class_LORA/notebooks/colab_a100_unsloth_qwen_finance.ipynb).
4. In the notebook toolbar, choose the Google Colab kernel / connect option.
5. Complete any Google sign-in flow the extension prompts for.
6. Confirm the notebook status changes from disconnected to connected.

Success signal:

- the notebook shows an active remote Colab kernel

## 2. Select an A100 runtime

Manual steps:

1. In the Colab runtime panel, open runtime settings.
2. Choose GPU hardware acceleration.
3. Select an `A100` runtime.
4. Prefer `A100 80GB` if your subscription offers it.

Success signal:

- running `nvidia-smi` shows an `A100` with about `80 GB` VRAM

## 3. Mount Google Drive

Manual steps:

1. Run the notebook cell `mount_drive`.
2. In the Google popup, grant Drive access.
3. Return to VS Code and wait for the cell to finish.

Success signal:

- `/content/drive/MyDrive/ML_Class_LORA/` is accessible in the runtime

## 4. Authenticate Hugging Face if needed

Manual steps:

Only required if:

- the model download needs authentication
- or you want to push a final adapter later

Recommended approach:

1. In Colab secrets or runtime environment, set `HF_TOKEN`.
2. Do not paste the token into committed files.
3. Run:

```bash
python scripts/colab/check_hf_auth.py --token-env HF_TOKEN
```

Success signal:

- it prints `hf_auth_ok: True`

Do not enable upload unless you explicitly want a private adapter push.

## 5. Authenticate GitHub if clone/pull needs it

For a public repo, no GitHub auth is normally required.

If GitHub auth is required in your environment:

1. use a PAT in a temporary credential helper
2. or use the Colab GitHub integration if the extension prompts for it
3. do not hardcode tokens in notebook cells or repo files

Success signal:

- `git clone` or `git pull --ff-only` succeeds in the notebook cell
