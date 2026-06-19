# Overnight Run Summary

## Command

```bash
python scripts/run_pipeline.py --config configs/data_pipeline.yaml --start bronze --through train
```

## Raw Data Located

- `ibes_raw_csv`: `2824304` rows from `/home/nathanaelguitar/ML_Class_LORA/admin/local/wrds-downloads/tr_ibes_11289435.csv`
- `crsp_presence_csv`: `83488398` rows from `/home/nathanaelguitar/Downloads/CRSP-1980-2024.csv`
- `crsp_daily_zip`: `20322877` rows from `/home/nathanaelguitar/ML_Class_LORA/admin/local/market-reaction/crsp_daily_returns_with_ids_1995-12-29_2006-01-16.zip`
- `crsp_stock_header_csv`: `38872` rows from `/home/nathanaelguitar/ML_Class_LORA/admin/local/market-reaction/crsp_stock_header_full.csv`
- `market_benchmark_csv`: `2559` rows from `/home/nathanaelguitar/ML_Class_LORA/admin/local/market-reaction/market_benchmark_returns_1995-12-29_2006-01-16.csv`

## Silver And Gold Counts

- IBES silver rows: `943231`
- IBES gold rows: `796423`
- Event-panel matched rows: `558349`
- Event-panel unmatched rows: `23867`
- Training gold rows: `582216`

## JSONL Splits

- train: `523994`
- train_eval: `1000`
- val: `29111`
- test: `29111`
- sample_100: `100`

## Split Contract

- `train`: adapter optimization rows
- `train_eval`: small deterministic subset of `val` used only for fast training-time health checks
- `val`: full validation pool retained locally, not used for repeated trainer validation during long runs
- `test`: full WRDS holdout reserved for final post-train base-vs-adapter measurement
- public benchmarks: separate generalization measurement and never part of Trainer evaluation

## Label Distribution

```json
{
  "sample_100": {
    "negative": 41,
    "neutral": 13,
    "positive": 46
  },
  "test": {
    "negative": 12749,
    "neutral": 4896,
    "positive": 11466
  },
  "train": {
    "negative": 243294,
    "neutral": 55554,
    "positive": 225146
  },
  "train_eval": {
    "negative": 403,
    "neutral": 183,
    "positive": 414
  },
  "val": {
    "negative": 12614,
    "neutral": 5122,
    "positive": 11375
  }
}
```

## Token Stats

```json
{
  "sample_100": {
    "input_tokens": {
      "max": 438,
      "mean": 417.46,
      "min": 343,
      "p50": 422,
      "p95": 429
    },
    "output_tokens": {
      "max": 71,
      "mean": 69.21,
      "min": 67,
      "p50": 69,
      "p95": 70
    }
  },
  "test": {
    "input_tokens": {
      "max": 449,
      "mean": 412.38,
      "min": 319,
      "p50": 421,
      "p95": 435
    },
    "output_tokens": {
      "max": 72,
      "mean": 69.36,
      "min": 67,
      "p50": 69,
      "p95": 71
    }
  },
  "train": {
    "input_tokens": {
      "max": 449,
      "mean": 420.23,
      "min": 320,
      "p50": 423,
      "p95": 432
    },
    "output_tokens": {
      "max": 72,
      "mean": 69.26,
      "min": 67,
      "p50": 69,
      "p95": 70
    }
  },
  "train_eval": {
    "input_tokens": {
      "max": 444,
      "mean": 414.85,
      "min": 323,
      "p50": 422,
      "p95": 433
    },
    "output_tokens": {
      "max": 71,
      "mean": 69.28,
      "min": 67,
      "p50": 69,
      "p95": 71
    }
  },
  "val": {
    "input_tokens": {
      "max": 445,
      "mean": 416.32,
      "min": 320,
      "p50": 422,
      "p95": 433
    },
    "output_tokens": {
      "max": 72,
      "mean": 69.29,
      "min": 67,
      "p50": 69,
      "p95": 71
    }
  }
}
```

## Validation And Training Gates

- validation passed: `True`
- smoke training succeeded: `True`
- overnight training started: `True`

## Artifact Paths

- bronze manifest: `data/processed/wrds_qwen_pipeline/reports/bronze_manifest.json`
- silver report: `data/processed/wrds_qwen_pipeline/reports/silver_report.json`
- gold report: `data/processed/wrds_qwen_pipeline/reports/gold_report.json`
- validation report: `data/processed/wrds_qwen_pipeline/reports/validation_report.json`
- jsonl directory: `data/processed/wrds_qwen_pipeline/jsonl`
- smoke log: `logs/wrds_qwen_pipeline/smoke_20260613T201855Z.log`
- smoke output dir: `outputs/wrds_qwen_pipeline/smoke/20260613T201855Z`
- overnight log: `logs/wrds_qwen_pipeline/overnight_train_20260613T202724Z.log`
- overnight output dir: `outputs/wrds_qwen_pipeline/train/20260613T202724Z`
