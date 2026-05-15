# FinGPT Dataset Export

Use this when you want a small FinGPT task sample in repo JSONL form for smoke tests, prompt inspection, or adapter training.

This export path is intentionally thin:

- it uses `external/FinGPT/fingpt/FinGPT_Benchmark/utils.py`
- it does not redefine FinGPT's dataset schema
- it only writes the selected split to JSONL

## Prerequisite

Make sure the submodule is present:

```bash
git submodule update --init --recursive
```

## Export a tiny sample

```bash
cd /home/nathanaelguitar/ML_Class_LORA
source .venv/bin/activate
python scripts/convert_fingpt_dataset.py \
  --dataset sentiment-cls \
  --split train \
  --max-rows 3 \
  --include-metadata \
  --output data/public/fingpt_sentiment_cls_sample.jsonl
```

## Export a larger local training slice

```bash
cd /home/nathanaelguitar/ML_Class_LORA
source .venv/bin/activate
python scripts/convert_fingpt_dataset.py \
  --dataset sentiment-cls \
  --split train \
  --max-rows 1000 \
  --include-metadata \
  --output data/baseline/fingpt_sentiment_cls_train_1000.jsonl
```

## Use FinGPT's local saved dataset cache instead of Hugging Face

If you have already downloaded datasets through FinGPT's own `data/download.py`, load from that local cache:

```bash
python scripts/convert_fingpt_dataset.py \
  --dataset sentiment-cls \
  --split train \
  --use-local-fingpt-cache \
  --output data/baseline/fingpt_sentiment_cls_train.jsonl
```

## Output format

Each output row stays in the same logical structure:

```json
{"instruction":"...","input":"...","output":"..."}
```

Optional source metadata can be added:

```json
{"instruction":"...","input":"...","output":"...","source_dataset":"FinGPT/fingpt-sentiment-cls","source_split":"train"}
```

## Important note

The export script does not apply a local prompt template.

Prompt rendering is handled later, at training time, through FinGPT's native `get_prompt(...)` inside:

- `training/common.py`

That keeps the stored JSONL lossless while still training Qwen against FinGPT's real instruction format.
