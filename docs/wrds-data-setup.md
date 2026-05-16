# WRDS Data Setup

This repo stores **code only** in GitHub.

WRDS raw data and WRDS-derived processed datasets belong in:

- Google Drive for team sharing
- ignored local folders for actual execution on the DGX

The required medallion pipeline engine is **PySpark**.

## Storage model

GitHub keeps:

- pipeline scripts
- training code
- docs
- configs
- tiny non-proprietary samples

Google Drive keeps:

- `WRDS/raw/tr_ibes_11289435.csv`
- `WRDS/processed/ibes_lora_baseline_1k.jsonl`
- `WRDS/processed/ibes_lora_baseline_10k.jsonl`
- `WRDS/README.txt`

Local execution on the DGX keeps:

- `admin/local/wrds-downloads/tr_ibes_11289435.csv`
- `data/processed/ibes_lora_baseline/...`

Do **not** commit WRDS raw CSVs or WRDS-derived processed datasets.

## First local step

Each teammate who needs the data pipeline locally should place the shared WRDS CSV at:

```text
admin/local/wrds-downloads/tr_ibes_11289435.csv
```

The repo already ignores `admin/local/` and `data/processed/`.

## Commands

From the repo root:

```bash
python scripts/doctor.py
python scripts/check_wrds_data.py --input admin/local/wrds-downloads/tr_ibes_11289435.csv
python scripts/summarize_ibes.py --input admin/local/wrds-downloads/tr_ibes_11289435.csv
python scripts/prepare_ibes_dataset.py --input admin/local/wrds-downloads/tr_ibes_11289435.csv --out data/processed/ibes_lora_baseline
```

Or with the Makefile:

```bash
make doctor
make check-data
make summarize-ibes
make prepare-ibes-small
```

The pipeline code uses PySpark for bronze / silver / gold processing and only materializes small sampled JSONL exports for LoRA/eval.

## Medallion layout

The IBES pipeline writes a lightweight bronze / silver / gold layout under the output directory:

```text
data/processed/ibes_lora_baseline/
  bronze/ibes_bronze.parquet
  silver/ibes_eps_us_current.parquet
  gold/ibes_revision_events.parquet
  jsonl/baseline_1k/train.jsonl
  jsonl/baseline_1k/eval.jsonl
  jsonl/baseline_1k/holdout.jsonl
  reports/ibes_pipeline_report.json
```

If enough gold events exist, the pipeline can also export:

```text
jsonl/baseline_10k/
```

## Pipeline intent

### Bronze

- raw WRDS IBES CSV loaded into typed columns with PySpark
- normalized tickers / OFTIC / CUSIP
- parsed dates and numeric values
- event date derived from revision date, then announcement date, then actual date

### Silver

- EPS rows only
- US-firm rows only
- current rows only, treating blank `CURRFL` as current
- USD reporting rows only, treating blank report currency as USD
- obvious duplicates removed

### Gold

- event-level analyst revision consensus rows
- grouped by company + event date + fiscal horizon
- consensus features
- prior-consensus features
- deterministic direction label
- deterministic magnitude bucket

The first LoRA/eval JSONL exports are intentionally simple and structured. Do not add fancy reasoning fields until the event labels are trustworthy.

## Scale plan

The first `800 / 100 / 100` split is only a validation gate for the first successful LoRA run.

After the first adapter trains and reloads cleanly, the next intended scale step is:

- `10,000` train
- `1,000` eval
- `1,000` holdout

The gold table should stay much larger than either split so additional sources can be joined later without redesigning the pipeline.

## Colab

Colab is optional for:

- EDA
- schema checks
- small local summaries

Colab is **not** the main training path for `Qwen/Qwen3.6-27B`.
Use the DGX for actual training.

## Related docs

- `AGENTS.md`
- `docs/agent-workflow.md`
- `docs/training-setup.md`
- `docs/fingpt-integration.md`
- `docs/wrds-setup.md`
