# Agent Workflow

## Start here

1. Read `AGENTS.md`.
2. Run `python scripts/doctor.py`.
3. Confirm the raw WRDS CSV exists locally.
4. Confirm `external/FinGPT/` exists.
5. Do not touch `.venv` unless explicitly asked.

## How to verify data

Check the raw IBES CSV:

```bash
python scripts/check_wrds_data.py --input admin/local/wrds-downloads/tr_ibes_11289435.csv
```

Summarize it:

```bash
python scripts/summarize_ibes.py --input admin/local/wrds-downloads/tr_ibes_11289435.csv
```

Important:

- Raw WRDS data is local-only.
- Shared team storage is Google Drive, not git.
- Audit the data before building any modeling split.

## How to inspect FinGPT

Start with the real submodule files:

- `external/FinGPT/fingpt/FinGPT_Benchmark/utils.py`
- `external/FinGPT/fingpt/FinGPT_Benchmark/data/download.py`
- `external/FinGPT/fingpt/FinGPT_Benchmark/train_lora.py`
- `external/FinGPT/fingpt/FinGPT_Benchmark/benchmarks/`

Do not invent a local FinGPT wrapper before inspecting those files.

## How to prepare bronze / silver / gold data

```bash
python scripts/prepare_ibes_dataset.py \
  --input admin/local/wrds-downloads/tr_ibes_11289435.csv \
  --out data/processed/ibes_lora_baseline
```

This creates:

- `data/processed/ibes_lora_baseline/bronze/ibes_bronze.parquet`
- `data/processed/ibes_lora_baseline/silver/ibes_eps_us_current.parquet`
- `data/processed/ibes_lora_baseline/gold/ibes_revision_events.parquet`
- `data/processed/ibes_lora_baseline/jsonl/baseline_1k/`
- `data/processed/ibes_lora_baseline/reports/ibes_pipeline_report.json`

These files stay out of git.

## How to run smoke training

```bash
python training/train_smoke.py \
  --model-id Qwen/Qwen3.6-27B \
  --max-steps 1 \
  --max-seq-length 1024 \
  --local-files-only
```

## How to run a small baseline

```bash
python training/train_finance_lora.py \
  --model-id Qwen/Qwen3.6-27B \
  --train-file data/processed/ibes_lora_baseline/jsonl/baseline_1k/train.jsonl \
  --eval-file data/processed/ibes_lora_baseline/jsonl/baseline_1k/eval.jsonl \
  --test-file data/processed/ibes_lora_baseline/jsonl/baseline_1k/holdout.jsonl \
  --output-dir outputs/qwen36-27b-ibes-baseline \
  --epochs 1 \
  --max-seq-length 1024 \
  --local-files-only
```

## How to run eval

Keep the holdout split untouched and compare base vs adapter on the same examples before claiming improvement.
Technical training success is not the same as model improvement.
The current adapter is an ordinary instruction/structured-output LoRA, not explicit Qwen thinking-mode training.

If an adapter already exists, validate that it loads:

```bash
python training/validate_adapter.py \
  --base-model Qwen/Qwen3.6-27B \
  --adapter-dir outputs/qwen36-27b-ibes-baseline
```

Then run the evaluation harness:

```bash
python eval/evaluate_base_vs_adapter.py \
  --model-id Qwen/Qwen3.6-27B \
  --adapter-path outputs/qwen36-27b-ibes-baseline \
  --holdout-file data/processed/ibes_lora_baseline/jsonl/baseline_1k/holdout.jsonl \
  --output-dir outputs/evals/qwen36-27b-ibes-baseline-holdout \
  --qwen-thinking-mode both \
  --max-new-tokens 80 \
  --batch-size 1 \
  --local-files-only
```

This compares:

- base Qwen in thinking mode
- adapter Qwen in thinking mode
- base Qwen in non-thinking mode
- adapter Qwen in non-thinking mode

Look at:

- `metrics.json`
- `eval_summary.md`
- `confusion_matrix.csv`
- `regression_examples.md`

Do not claim benchmark or reasoning improvement until those files show exact metric wins over base Qwen.

## What not to touch

- `.venv` for Unsloth experiments
- `admin/local/` browser cookies or profile data
- raw WRDS CSVs
- processed WRDS parquet or JSONL outputs
- large model artifacts
- the full 2.8M-row IBES file as a direct training set

## How to commit safely

Run:

```bash
git status --short
git diff --cached --stat
find . -type f -size +25M -not -path "*/.git/*"
```

Only commit:

- docs
- scripts
- config examples
- tiny non-proprietary samples

Do not commit:

- `admin/local/`
- `data/processed/`
- `outputs/`
- `checkpoints/`
- raw WRDS extracts
