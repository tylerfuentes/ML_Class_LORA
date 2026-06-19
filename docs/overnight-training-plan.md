# Overnight Training Plan

This plan exists to use a long overnight window productively without defaulting to a blind larger pure-IBES run.

## Goal

Find out whether a better data strategy improves general finance performance without losing the structured IBES skill.

The target is not maximum rows consumed.
The target is measured generalization.

## Current baseline roles

- `1k` adapter: best general finance adapter so far
- `10k` adapter: best narrow IBES structured JSON specialist
- pure `50k` IBES scaling: not started
- no alpha or trading claim until market-reaction data and backtests exist

## Candidate datasets

These are built locally under ignored `outputs/overnight_tournament/candidates/` by:

```bash
./.venv/bin/python scripts/build_overnight_training_candidates.py
```

Candidates:

1. `high_quality_ibes_4k`
2. `balanced_ibes_10k`
3. `mixed_finance_10k`
4. `diverse_ibes_10k`

## Tournament order

Default order:

1. `high_quality_ibes_4k`
2. `balanced_ibes_10k`
3. `mixed_finance_10k`
4. `diverse_ibes_10k`

Rationale:

- start with the cheapest quality-first test
- test concentration and balance before scale-plus-mixing
- only spend the full mixed-data effort if the earlier pure-IBES quality fixes do not answer the question

## Training path

- base model: `Qwen/Qwen3.6-27B`
- framework: Hugging Face Transformers + PEFT + bitsandbytes QLoRA
- mode: non-thinking
- no fake `<think>` traces
- no Unsloth

## Required evaluation after every candidate

- IBES holdout
- FiQA
- Financial PhraseBank
- TFNS
- NWGI

Compare against:

- base Qwen
- `1k` adapter
- `10k` adapter

Metrics:

- accuracy
- macro F1
- parse failure rate
- exact JSON match on IBES
- magnitude bucket accuracy on IBES
- confusion matrix
- representative wins and regressions

## Stop rules

Stop if any of the following happens:

1. a candidate beats the `1k` adapter on at least three public benchmarks while preserving IBES performance
2. the first two candidates both clearly regress public benchmarks versus `1k`
3. GPU/runtime stability becomes unreliable
4. outputs show format collapse, severe label bias, or over-specialization

Do not keep training just to fill time.

## Runner

The local tournament runner is:

```bash
./.venv/bin/python scripts/run_overnight_training_tournament.py
```

Use `--dry-run` first if you only want prerequisite validation.

The runner writes:

- local run artifacts under ignored `outputs/overnight_tournament/runs/`
- a tracked summary report at `docs/overnight-training-report.md`

## Market-reaction work remains separate

Market-reaction data and join logic should continue in parallel, but they must not block the controlled training tournament.

The remaining hard blocker for the eventual backtesting stage is still the real CRSP/Compustat link table.
