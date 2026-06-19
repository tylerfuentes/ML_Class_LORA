# Next Training Experiment

This document captures the post-10k diagnosis and the parked next-training idea.

Current status:

- this is not the active repo milestone
- no `50k` training is approved
- no new training should start until the market-reaction milestone clarifies whether better event labels are actually the bottleneck

## Candidate manifests

These manifests are local-only and live under ignored `outputs/next-training-experiment/manifests/`.

### balanced_ibes_10k

- path: `outputs/next-training-experiment/manifests/candidate_a_balanced_ibes_10k.json`
- rows: `10000`
- description: Balanced by direction first, then constrained by magnitude and concentration caps.

### diverse_ibes_10k

- path: `outputs/next-training-experiment/manifests/candidate_b_diverse_ibes_10k.json`
- rows: `10000`
- description: Maximize company/date/fiscal-period diversity with per-ticker and per-period caps.

### mixed_finance_10k

- path: `outputs/next-training-experiment/manifests/candidate_c_mixed_finance_10k.json`
- rows: `10000`
- description: 8000 diverse IBES rows plus 2000 public finance benchmark-style rows.

### small_high_quality_4k

- path: `outputs/next-training-experiment/manifests/candidate_d_small_high_quality_4k.json`
- rows: `4000`
- description: Higher-consensus, lower-repetition IBES subset with stricter filtering.

## Recommendation

- status: parked until after market-reaction measurement
- recommended experiment: `mixed finance 10k`
- problem addressed: The public regressions are concentrated in neutral-to-positive overcalls, which is more consistent with overspecialization than with missing capacity. A mixed dataset directly targets retention of general finance sentiment boundaries while preserving IBES structure.
- target metric: FIQA/FPB/TFNS macro F1 should recover toward the 1k adapter while keeping IBES exact JSON near the 10k adapter.
- main risk: Adding public examples could slightly reduce narrow IBES formatting gains if the mix is too aggressive.
- evaluation: Compare the new adapter against base Qwen, the 1k adapter, and the 10k adapter on IBES holdout, FIQA, FPB, TFNS, and NWGI using accuracy, macro F1, parse failure, exact JSON, magnitude bucket accuracy, and confusion matrices.
- why better than 50k pure IBES: A 50k pure-IBES run would increase the same specialization pressure without first testing whether diversity or mixed supervision solves the actual problem.
- reason it is parked: the next evidence gap is downstream market reaction, not another pure model-quality number

## CRSP / link data request

- CRSP daily returns
  - required columns: `permno`, `date`, `ret`, `prc`, `shrout`, `vol`
  - date range: at least the full IBES event span already in gold, plus a small buffer for post-event windows
  - expected Google Drive location: `ML_Class_LORA_shared/crsp/crsp_daily_returns/`
  - unlocks: event-window realized return labels and abnormal-return calculations
- CRSP/Compustat link table
  - required columns: `gvkey`, `permno`, `linktype`, `linkprim`, `linkdt`, `linkenddt`
  - date range: full range covering the IBES gold sample
  - expected Google Drive location: `ML_Class_LORA_shared/crsp/crsp_compustat_link/`
  - unlocks: reliable entity joins from IBES into returns and fundamentals
- Market benchmark returns
  - required columns: `date`, benchmark return series such as market index total return
  - date range: same as CRSP daily returns
  - expected Google Drive location: `ML_Class_LORA_shared/crsp/market_benchmarks/`
  - unlocks: market-adjusted or abnormal return labels
