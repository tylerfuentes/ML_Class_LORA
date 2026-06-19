# Google Colab Browser Quickstart

This repo already includes a Colab-ready notebook and Colab helper scripts.
The only part that cannot be automated from this machine is your Google sign-in.

Use **your own Google account** for this. Colab, Drive, and any checkpoint you
produce will all be tied to whichever account you sign in with — keep it the
same account for every step below (Colab sign-in, Drive mount, `rclone
config` if you back up from a local machine). Do not reuse another
classmate's account or credentials.

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
2. Sign in with your own Google account.
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

## Getting the gold JSONL data onto your own Drive

The gold layer (`train.jsonl`, `val.jsonl`, `test.jsonl`, `train_eval.jsonl`)
is never committed to GitHub (see
[colab-a100-storage-policy.md](/home/nathanaelguitar/ML_Class_LORA/docs/colab-a100-storage-policy.md)),
so it will not exist on your Drive just because you cloned the repo. You need
one of:

1. **Get a shared Drive link from a teammate who already produced the data.**
   Ask whoever ran the WRDS/IBES pipeline to share the
   `ML_Class_LORA/data/gold/` folder (Drive: right-click folder -> Share),
   then use Drive's "Add shortcut to Drive" so it resolves at
   `/content/drive/MyDrive/ML_Class_LORA/data/gold/` after mount. Do not ask
   for or forward raw WRDS/CRSP/Compustat files outside of Drive — those may
   not leave the approved storage layer.
2. **Generate it yourself**, if you have your own WRDS access, following
   [docs/wrds-data-setup.md](/home/nathanaelguitar/ML_Class_LORA/docs/wrds-data-setup.md)
   and `scripts/prepare_ibes_dataset.py` / the WRDS pipeline scripts under
   `scripts/`, then upload the resulting `data/processed/.../jsonl/*.jsonl`
   files to your own Drive at the path above. From a local machine with
   `rclone` configured against your own Google account, that looks like:

   ```bash
   rclone copy data/processed/<your_pipeline>/jsonl/train.jsonl \
     gdrive:ML_Class_LORA/data/gold/
   # repeat for train_eval.jsonl, val.jsonl, test.jsonl
   ```

Verify the data landed correctly before training: cell 6 in the notebook
reads each file under `config.datasets` and prints `exists=True`/`size_mb`
for all four. Don't proceed past that cell until all four show `exists=True`.

## Starting fresh vs. resuming someone else's checkpoint

Most classmates will be **starting a fresh run** — a new `RUN_NAME` in cell
10/12, no `--resume-latest` flag, training from the base HF model. That's
the default and requires nothing extra.

Only use `--resume-latest` if you have an existing checkpoint placed at
`/content/drive/MyDrive/ML_Class_LORA/checkpoints/<run_name>/checkpoint-N/`
under **your own Drive account** — e.g. if a teammate shared their
checkpoint folder with you and you added it as a shortcut at that exact
path, with the matching `RUN_NAME` in cells 10 and 12. Resuming with a
mismatched `RUN_NAME` or missing checkpoint files will silently start a
fresh run instead of resuming, since `--resume-latest` only resumes if it
finds at least one `checkpoint-*` directory there.

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
