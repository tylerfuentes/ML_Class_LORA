# AGENTS

## Project summary

This repo builds a finance/event-reasoning LoRA workflow around `Qwen/Qwen3.6-27B`.
The current verified training path is Hugging Face Transformers + PEFT + bitsandbytes QLoRA.
The required data pipeline engine is PySpark.
FinGPT is the upstream finance task/benchmark reference, imported as a git submodule.
WRDS data is shared through Google Drive and local ignored folders, not through git.

## Local machine context

This repo lives on a DGX Spark workstation with existing team SSH onboarding and relay infrastructure already in use.
The repo already has working QLoRA scaffolding and a repaired CUDA-capable `.venv`.
Use the repo-local workflow below first; do not re-derive infrastructure that is already present on this machine.

Relevant recent local facts:
- Existing SSH onboarding for `ML_Class_LORA` was already implemented and pushed.
- Existing Qwen 3.6 27B QLoRA training infrastructure was already found and migrated into this repo.
- Reverse-SSH relay access for teammates already exists; do not rewrite that workflow unless explicitly asked.
- The main repo `.venv` was previously repaired after an Unsloth install pulled a CPU-only torch build on ARM64.

## Directory map

- `README.md`
  - human-readable project entry point
- `docs/`
  - operational docs, source references, agent workflow, WRDS setup
- `config/`
  - example local path configuration
- `scripts/`
  - data checks, WRDS prep, doctor script, utility scripts
- `scripts/colab/`
  - reusable logic the Colab notebook shells out to (runtime verification,
    sanity gate, training launcher, eval runner, HF push) — keep the
    notebook thin, put logic here instead
- `notebooks/`
  - thin Colab-executable notebooks, outputs always cleared before commit
- `training/`
  - verified QLoRA training scripts
- `data/samples/`
  - tiny committed samples only
- `data/processed/`
  - ignored local processed datasets
- `admin/local/`
  - ignored local WRDS downloads, browser state, machine-specific files
- `external/FinGPT/`
  - canonical upstream finance framework/reference layer

## Core scripts

- `scripts/doctor.py`
  - verify local setup, paths, imports, CUDA visibility, and warnings
- `scripts/check_wrds_data.py`
  - confirm the raw WRDS CSV exists and has the expected IBES columns
- `scripts/summarize_ibes.py`
  - print a compact PySpark audit report for the raw WRDS IBES CSV
- `scripts/prepare_ibes_dataset.py`
  - build PySpark bronze / silver / gold IBES artifacts plus small JSONL baselines
- `scripts/ibes_pipeline.py`
  - shared PySpark bronze / silver / gold IBES helpers used by the command scripts
- `scripts/test_wrds_connection.py`
  - sanity-check direct WRDS Python/Postgres authentication
- `scripts/market_reaction/check_market_data.py`
  - validate the expected CRSP/link/benchmark files and columns without loading full datasets
- `scripts/market_reaction/build_event_panel.py`
  - validate the event-panel join contract and required schemas
- `scripts/market_reaction/compute_event_windows.py`
  - validate requested event-window definitions and expected output columns
- `scripts/market_reaction/score_label_alignment.py`
  - validate the planned label-vs-return evaluation contract
- `training/train_smoke.py`
  - verified one-step QLoRA smoke run
- `training/train_finance_lora.py`
  - main small-baseline adapter training entry point
- `training/validate_adapter.py`
  - load and sanity-check a saved adapter

## Data flow

Google Drive / WRDS download
→ `admin/local/wrds-downloads/`
→ `scripts/check_wrds_data.py`
→ `scripts/summarize_ibes.py`
→ `scripts/prepare_ibes_dataset.py`
→ `data/processed/ibes_lora_baseline/bronze`
→ `data/processed/ibes_lora_baseline/silver`
→ `data/processed/ibes_lora_baseline/gold`
→ `data/processed/ibes_lora_baseline/jsonl`
→ training
→ market-reaction validation and planning

Expected raw input path:

- `admin/local/wrds-downloads/tr_ibes_11289435.csv`

Expected processed outputs:

- `data/processed/ibes_lora_baseline/bronze/ibes_bronze.parquet`
- `data/processed/ibes_lora_baseline/silver/ibes_eps_us_current.parquet`
- `data/processed/ibes_lora_baseline/gold/ibes_revision_events.parquet`
- `data/processed/ibes_lora_baseline/jsonl/baseline_1k/train.jsonl`
- `data/processed/ibes_lora_baseline/jsonl/baseline_1k/eval.jsonl`
- `data/processed/ibes_lora_baseline/jsonl/baseline_1k/holdout.jsonl`

Google Drive mirrors:

- `WRDS/raw/tr_ibes_11289435.csv`
- `WRDS/processed/ibes_lora_baseline_1k.jsonl`
- `WRDS/processed/ibes_lora_baseline_10k.jsonl`
- `WRDS/README.txt`

## WRDS access guidance

- Prefer direct WRDS Python/Postgres access when credentials are verified to work.
- Do not commit WRDS CSVs or derived large datasets; keep them under ignored local paths or Google Drive.
- Commit only small non-proprietary samples, scripts, and docs.

## Training flow

1. Run doctor.
2. Check the raw WRDS CSV.
3. Summarize the IBES CSV with PySpark.
4. Build PySpark bronze / silver / gold IBES artifacts.
5. Run `training/train_smoke.py`.
6. Train a 1k baseline adapter.
7. Validate or evaluate the saved adapter.
8. Scale to larger gold-derived splits only after the first adapter works.

Current project status:

- the `1k` adapter is the best general finance adapter so far
- the `10k` adapter is the best structured IBES JSON specialist so far
- pure IBES scaling is paused
- the next bottleneck is market-reaction data, not more training
- do not start blind pure-IBES scale-up runs such as `50k` just because more rows exist
- if training resumes, it must be a controlled tournament focused on data quality, diversity, mixing, and measured generalization
- every new candidate run must be evaluated against base Qwen, the `1k` adapter, and the `10k` adapter before the next run starts
- do not keep training just to fill time; stop when the public-benchmark/generalization answer is clear
- a WRDS-500k Unsloth run reached global_step=4500 (~71k examples seen,
  "the 71k checkpoint") before being interrupted; it is backed up to Drive,
  reload-verified, and compared against the `1k`/`10k` adapters as of
  2026-06-19 — see `docs/eval-findings.md`. It is the only one of the three
  with no format/parse regression on the WRDS task, but accuracy/F1 are not
  yet measurable on this holdout (label-extractor mismatch, documented in
  eval-findings.md) — **not promoted to Hugging Face**, stays archived in
  Drive until that metric is fixed and a real quality read exists
- do not push any checkpoint to Hugging Face until accuracy/F1 are actually
  measurable and a promotion decision is explicitly made (see
  `colab-a100-storage-policy.md`)

## Market-reaction flow

1. Validate the IBES gold event file shape.
2. Validate local CRSP daily returns.
3. Validate the CRSP/Compustat link table.
4. Validate optional benchmark returns.
5. Build the schema-aware event-panel contract.
6. Build the schema-aware event-window contract.
7. Score label-alignment only after realized-return columns exist.

Expected local market-reaction paths:

- `admin/local/market-reaction/crsp_daily_returns.csv`
- `admin/local/market-reaction/crsp_compustat_link.csv`
- `admin/local/market-reaction/market_benchmark_returns.csv`
- `data/processed/market_reaction/`

Important market-reaction rules:

- market-reaction scripts are schema-aware validators and planning tools first
- they must not invent data or fabricate joined outputs
- prefer the richer local CRSP daily file with identifier columns when it exists
- support a non-CCM fallback join path using `NCUSIP` / `CUSIP` / `TICKER` plus CRSP stock-header history
- treat any non-CCM join as diagnostic until a real CRSP/Compustat link table exists
- real WRDS, CRSP, and Compustat files belong in Google Drive or ignored local folders
- commit only tiny synthetic samples under `data/samples/market_reaction/`
- no alpha or trading claim is allowed until real backtests exist

## FinGPT usage

- `external/FinGPT/` is the canonical upstream reference.
- Inspect real files under `external/FinGPT/fingpt/FinGPT_Benchmark/` before adapting anything.
- Prefer FinGPT-native prompt/data/task structure first.
- Custom glue should stay minimal and final-mile only.

## Unsloth status

- Unsloth is now the active training path for the WRDS-500k run, both
  locally on this DGX box (`training/train_finance_lora_unsloth.py`,
  `scripts/launch_wrds_unsloth_gb10.sh`) and on Colab A100 (see Colab
  migration section below). This supersedes the earlier "not the current
  default path" guidance.
- On this specific local machine's `.venv`, a direct Unsloth install
  previously broke the CUDA stack on ARM64 by pulling a CPU-only torch
  build. The working `.venv` here has since been repaired and is in active
  use for Unsloth training — do not re-break it by reinstalling Unsloth
  from scratch or upgrading torch/bitsandbytes casually.
- This caution is specific to this DGX box's ARM64 `.venv`. It does not
  apply to Colab, which runs its own clean runtime per session.

## Colab / Unsloth migration (for classmates)

The heavy training workflow can run on a Google Colab A100 runtime instead
of (or in addition to) a local DGX/GPU box. GitHub is the source of truth
for code/docs/configs/cleared-output notebooks; Google Drive holds gold
JSONL data, checkpoints, logs, and outputs; a private Hugging Face repo
holds only a selected, verified final adapter.

Start here:

- [docs/google-colab-browser-quickstart.md](/home/nathanaelguitar/ML_Class_LORA/docs/google-colab-browser-quickstart.md)
  — browser-only path, no local setup required, includes how to get gold
  data onto your own Drive and the difference between starting fresh vs.
  resuming a shared checkpoint
- [docs/vscode-colab-workflow.md](/home/nathanaelguitar/ML_Class_LORA/docs/vscode-colab-workflow.md)
  — VS Code + Colab extension path, what's safe to edit/commit
- [docs/colab-auth-checklist.md](/home/nathanaelguitar/ML_Class_LORA/docs/colab-auth-checklist.md)
  — step-by-step auth (Colab, Drive, optional HF)
- [docs/colab-a100-storage-policy.md](/home/nathanaelguitar/ML_Class_LORA/docs/colab-a100-storage-policy.md)
  — what goes in GitHub vs. Drive vs. Hugging Face, and the required
  backup-verify-compare-promote order before anything reaches Hugging Face
- [notebooks/colab_a100_unsloth_qwen_finance.ipynb](/home/nathanaelguitar/ML_Class_LORA/notebooks/colab_a100_unsloth_qwen_finance.ipynb)
  — the actual thin notebook; reusable logic lives in `scripts/colab/`, not
  in notebook cells

Important gotcha: editing the notebook in GitHub/VS Code does not change a
notebook already open in a Colab browser tab. `git pull` inside Colab only
refreshes the standalone files it shells out to (`scripts/colab/*.py`,
`config/*.yaml`); to pick up a change to the notebook's own cells, reopen
the notebook from GitHub or edit the cell directly in the browser.

Never push full checkpoint history or every candidate adapter to Hugging
Face. Only one verified, selected adapter goes there, after comparison
against existing reference adapters — see the promotion-policy section of
`colab-a100-storage-policy.md` and the worked example in
`docs/eval-findings.md` (the "71k checkpoint" entry).

## Hard rules

- Do not commit WRDS raw or processed data.
- Do not commit CRSP data or Compustat data.
- Do not commit browser profiles or cookies.
- Do not commit model weights or checkpoints.
- Do not train on all 2.8M IBES rows directly.
- Do not replace the PySpark data path with a pandas-only path.
- Do not guess FinGPT APIs; inspect the submodule first.
- Do not modify the working `.venv` to test Unsloth.
- Do not start blind new IBES adapter training until the candidate dataset and evaluation plan are explicit.

## First commands

From the repo root:

```bash
python scripts/doctor.py
python scripts/check_wrds_data.py --input admin/local/wrds-downloads/tr_ibes_11289435.csv
python scripts/summarize_ibes.py --input admin/local/wrds-downloads/tr_ibes_11289435.csv
python scripts/prepare_ibes_dataset.py --input admin/local/wrds-downloads/tr_ibes_11289435.csv --out data/processed/ibes_lora_baseline
```

Or:

```bash
make doctor
make check-data
make summarize-ibes
make prepare-ibes-small
```

## Safe commit checklist

- `git status --short`
- `git diff --cached --stat`
- `find . -type f -size +25M -not -path "*/.git/*"`
- confirm nothing under `admin/local/` is staged
- confirm nothing under `data/processed/` is staged
- confirm no weights/checkpoints/outputs are staged

## Read next

- `docs/agent-workflow.md`
- `docs/wrds-data-setup.md`
- `docs/training-setup.md`
- `docs/fingpt-integration.md`
- `docs/wrds-setup.md`
- `docs/google-colab-browser-quickstart.md` (start here for Colab/no-local-setup)
- `docs/vscode-colab-workflow.md`
- `docs/colab-auth-checklist.md`
- `docs/colab-a100-storage-policy.md`
- `docs/adapter-lifecycle.md`
- `docs/eval-findings.md`
