# Sources

This page collects the main upstream references used by this repo so teammates can see where the project assumptions, task formats, and data workflows came from.

## Core project dependencies

- FinGPT repository:
  - https://github.com/AI4Finance-Foundation/FinGPT
- Unsloth repository:
  - https://github.com/unslothai/unsloth
- Qwen model family on Hugging Face:
  - https://huggingface.co/Qwen

## FinGPT task and benchmark references

- FinGPT benchmark utilities and prompt format:
  - `external/FinGPT/fingpt/FinGPT_Benchmark/utils.py`
- FinGPT benchmark dataset downloader:
  - `external/FinGPT/fingpt/FinGPT_Benchmark/data/download.py`
- FinGPT benchmark training reference:
  - `external/FinGPT/fingpt/FinGPT_Benchmark/train_lora.py`
- FinGPT benchmark task implementations:
  - `external/FinGPT/fingpt/FinGPT_Benchmark/benchmarks/`

Useful public dataset endpoints confirmed during integration:

- `FinGPT/fingpt-sentiment-cls`
  - https://huggingface.co/datasets/FinGPT/fingpt-sentiment-cls
- `FinGPT/fingpt-headline`
  - https://huggingface.co/datasets/FinGPT/fingpt-headline

## Unsloth references

- Unsloth Qwen fine-tuning docs:
  - https://docs.unsloth.ai/models/qwen3-how-to-run-and-fine-tune
- Unsloth chat template docs:
  - https://docs.unsloth.ai/basics/chat-templates
- Unsloth datasets guide:
  - https://docs.unsloth.ai/basics/datasets-guide

## SEC EDGAR references

- SEC developer resources:
  - https://www.sec.gov/about/developer-resources
- SEC EDGAR APIs:
  - https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- SEC company ticker map used by the demo script:
  - https://www.sec.gov/files/company_tickers.json

## Research references

- SenseAI:
  - Kabalisa, B. (2026). *SenseAI: A Human-in-the-Loop Dataset for RLHF-Aligned Financial Sentiment Reasoning*.
  - https://arxiv.org/abs/2604.05135
- iGRPO:
  - Hatamizadeh et al. (2026). *iGRPO: Self-Feedback-Driven LLM Reasoning*.
  - https://arxiv.org/abs/2602.09000
- Search-driven reward optimization for reasoning:
  - Ahmadi et al. (2026). *Enhanced LLM Reasoning by Optimizing Reward Functions with Search-Driven Reinforcement Learning*.
  - https://arxiv.org/abs/2605.02073

## Notes

- When this repo says “FinGPT-native,” it means we prefer the prompt, dataset, and benchmark structure already present in the FinGPT submodule before creating local wrappers.
- When this repo says “verified training path,” it refers to the repo-local Hugging Face / PEFT QLoRA workflow that was actually run successfully on this DGX.
