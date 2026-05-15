# FinGPT Integration

This repo now treats `external/FinGPT/` as the upstream finance task/data reference layer.

The target model does not change:

- base model: `Qwen/Qwen3.6-27B`
- training style: adapter-only QLoRA in 4-bit by default

The architecture order is:

1. FinGPT-native data/task structure first
2. Unsloth only where it materially speeds training
3. Custom glue only for the final step that FinGPT does not already provide

## Upstream files we are actually using

These are the primary FinGPT files this repo depends on:

- `external/FinGPT/fingpt/FinGPT_Benchmark/utils.py`
  - native prompt template
  - native dataset loader
- `external/FinGPT/fingpt/FinGPT_Benchmark/data/download.py`
  - canonical public dataset list
- `external/FinGPT/fingpt/FinGPT_Benchmark/train_lora.py`
  - reference LoRA training path used by FinGPT
- `external/FinGPT/fingpt/FinGPT_Benchmark/benchmarks/`
  - benchmark task logic such as `fpb.py`, `fiqa.py`, `headline.py`, `tfns.py`, `nwgi.py`
- `external/FinGPT/fingpt/FinGPT_Sentiment_Analysis_v3/training_int4/train.ipynb`
  - FinGPT's prior 4-bit training precedent

## 1. Which FinGPT datasets/tasks are useful for the first adapter?

Best first adapter dataset:

- `sentiment-cls`
  - loads from `FinGPT/fingpt-sentiment-cls`
  - public
  - large enough to be practical
  - already in `instruction` / `input` / `output`

Best next event-oriented task:

- `headline`
  - loads from `FinGPT/fingpt-headline`
  - closer to event classification than generic sentiment
  - still not a true market-reaction label

Best first benchmark tasks against base Qwen vs LoRA:

- `fpb`
- `fiqa`
- `tfns`
- `nwgi`
- optionally `headline` for a more event-flavored classification check

## 2. What format are they in?

FinGPT benchmark datasets already use the structure we need:

```json
{"instruction":"...","input":"...","output":"..."}
```

FinGPT's native prompt construction in `utils.py` is:

```text
Instruction: {instruction}
Input: {input}
Answer:
```

This repo now preserves that prompt format at training time instead of replacing it with a made-up local template.

## 3. How do we convert them into our JSONL training format?

Only the final storage step is converted.

The export script:

- calls FinGPT's real `load_dataset(...)`
- keeps `instruction`, `input`, and `output` unchanged
- optionally annotates source metadata

Script:

- `scripts/convert_fingpt_dataset.py`

This is intentionally small because the data is already close to the training format we want.

## 4. Can they support event-driven reasoning, sentiment, earnings/news reaction, or only generic finance sentiment?

What FinGPT can support immediately:

- generic finance sentiment
- headline / event classification
- finance QA and benchmark-style evaluation

What FinGPT does not fully replace:

- WRDS-backed event-reaction supervision
- intraday or abnormal-return labels
- analyst revision / surprise logic from IBES

So for this project:

- FinGPT is strong for the first finance adapter and benchmark layer
- WRDS / SEC / TAQ style data is still the better next step for market-reaction grounding

## 5. What benchmark can we run against base Qwen before and after LoRA?

First benchmark set:

- `fpb`
- `fiqa`
- `tfns`
- `nwgi`

If we want one benchmark that is slightly more event-oriented:

- `headline`

Recommended order:

1. Run base Qwen on a small FinGPT benchmark slice.
2. Fine-tune a QLoRA adapter on `sentiment-cls`.
3. Re-run the benchmark slice.
4. Add `headline` next if we want a better event-classification story before moving to WRDS-style labels.

## 6. What files should be copied, referenced, or scripted without vendoring the entire repo?

Reference directly from the submodule:

- `FinGPT_Benchmark/utils.py`
- `FinGPT_Benchmark/data/download.py`
- benchmark task files in `FinGPT_Benchmark/benchmarks/`

Do not copy into the main tree unless there is a narrow reason.

This repo should own only:

- training glue for `Qwen/Qwen3.6-27B`
- export scripts
- small smoke-test samples
- local docs
- optional notebook or training launchers

## Current repo decisions

- `external/FinGPT/` is tracked as a git submodule.
- `training/common.py` now builds prompts through FinGPT's actual `get_prompt(...)`.
- `scripts/convert_fingpt_dataset.py` now exports through FinGPT's actual `load_dataset(...)`.
- The base training path remains the repo-local Hugging Face / PEFT QLoRA workflow because it is verified on this DGX.

## Unsloth status on this machine

Unsloth is still the intended fast-training path, but it is not the default training environment yet.

Reason:

- a direct `pip install unsloth` in this ARM64 DGX environment pulled `torch 2.10.0+cpu`
- that dropped CUDA visibility and broke the previously verified training stack
- the repo `.venv` was then rebuilt back to a working CUDA-capable `torch 2.12.0+cu130`

So today:

- verified path: repo-local HF/PEFT QLoRA
- intended acceleration path: Unsloth in a separate, isolated environment once the ARM64/CUDA packaging issue is solved cleanly
