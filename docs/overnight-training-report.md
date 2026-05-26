# Overnight Training Report

This report tracks the controlled overnight tournament for finding a stronger general finance/event adapter without blind pure-IBES scaling.

## Baseline roles

- `1k adapter`: best general finance adapter so far
- `10k adapter`: best narrow IBES structured JSON specialist
- pure `50k` IBES scaling: not started
- no alpha or trading claim until market-reaction data and backtests exist

## Tournament results

| candidate | status | public wins vs 1k | public wins vs 10k | public regressions vs 1k | IBES accuracy | IBES macro F1 | IBES exact JSON |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| high_quality_ibes_4k | completed | 1 | 1 | 3 | 1.0000 | 1.0000 | 0.7350 |
| balanced_ibes_10k | completed | 3 | 4 | 1 | 1.0000 | 1.0000 | 0.9970 |

## Baseline comparison anchors

- 1k IBES exact JSON: `0.9940`
- 10k IBES exact JSON: `0.9990`

## Stop rule

- Stopped after balanced_ibes_10k: it beat the 1k adapter on at least three public benchmarks while preserving IBES performance.

## Metric snapshot

- `high_quality_ibes_4k`
  - IBES holdout: `1.0000` accuracy, `1.0000` macro F1, `0.7350` exact JSON, `0.7350` magnitude bucket accuracy, `0.0000` parse failure
  - public macro F1: `FiQA 0.6851`, `FPB 0.6438`, `TFNS 0.6878`, `NWGI 0.5647`
- `balanced_ibes_10k`
  - IBES holdout: `1.0000` accuracy, `1.0000` macro F1, `0.9970` exact JSON, `0.9970` magnitude bucket accuracy, `0.0000` parse failure
  - public macro F1: `FiQA 0.6721`, `FPB 0.7858`, `TFNS 0.7465`, `NWGI 0.6487`
- baseline anchors
  - `1k` public macro F1: `FiQA 0.6759`, `FPB 0.7553`, `TFNS 0.7419`, `NWGI 0.6023`
  - `10k` public macro F1: `FiQA 0.6597`, `FPB 0.7125`, `TFNS 0.6970`, `NWGI 0.6191`

## Decision summary

- best model for structured IBES: `10k adapter` remains the strongest narrow IBES JSON specialist overall; `balanced_ibes_10k` is close and clearly preserves the structured task
- best model for public finance generalization: `balanced_ibes_10k` from this tournament
- best candidate for the next event-reasoning stage: `balanced_ibes_10k`, because it preserved IBES exact-JSON fidelity while improving on the `1k` adapter across `FPB`, `TFNS`, and `NWGI`
- more pure IBES scaling justified: no blind scale-up from this result alone; the useful signal was dataset balancing and concentration control, not simply “more rows”
- mixed-finance training justified: still yes as a follow-up experiment, but not because the tournament failed; it is now a targeted next step to see whether `FiQA` can be recovered without giving up the gains on `FPB`, `TFNS`, `NWGI`, and IBES fidelity
- missing data for market-reaction/backtesting: formal CRSP/Compustat link table is still missing

## Interpretation

- `high_quality_ibes_4k` showed that cleaner smaller data alone was not enough. It preserved direction classification, but exact JSON and magnitude fidelity collapsed relative to the established `1k` and `10k` adapters.
- `balanced_ibes_10k` is the first candidate that materially improved general finance performance without sacrificing the structured IBES task. It beat the `1k` adapter on three public benchmarks and the `10k` adapter on all four public benchmarks while keeping IBES holdout performance at `1.0000 / 1.0000`.
- The remaining weakness is `FiQA`, where `balanced_ibes_10k` regressed slightly versus the `1k` adapter. That makes the next training question a targeted data-mix/generalization question rather than a pure scaling question.
