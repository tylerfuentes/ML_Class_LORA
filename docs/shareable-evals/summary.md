# Shareable Evaluation Summary

This folder is the git-safe view of the recent evaluation work.

- Included: aggregate metrics, confusion counts, comparison deltas, and links back to the ignored local metrics files.
- Excluded: WRDS raw data, WRDS-derived predictions, raw holdout examples, adapters, checkpoints, caches, and logs.

## WRDS holdout

### WRDS baseline 1k holdout

- base accuracy: `0.4200`
- adapter accuracy: `1.0000`
- base macro F1: `0.5183`
- adapter macro F1: `1.0000`
- base parse failure rate: `0.5600`
- adapter parse failure rate: `0.0000`
- adapter exact JSON match: `0.9700`
- adapter magnitude bucket accuracy: `0.9700`
- source metrics file: `outputs/evals/qwen36-27b-ibes-baseline-holdout/metrics.json`

### WRDS 10k holdout with 10k adapter

- base accuracy: `0.4490`
- adapter accuracy: `1.0000`
- base macro F1: `0.5405`
- adapter macro F1: `1.0000`
- base parse failure rate: `0.5340`
- adapter parse failure rate: `0.0000`
- adapter exact JSON match: `0.9990`
- adapter magnitude bucket accuracy: `0.9990`
- source metrics file: `outputs/evals/qwen36-27b-ibes-10k-holdout/metrics.json`

### WRDS 10k holdout with prior 1k adapter

- base accuracy: `0.4490`
- adapter accuracy: `1.0000`
- base macro F1: `0.5405`
- adapter macro F1: `1.0000`
- base parse failure rate: `0.5340`
- adapter parse failure rate: `0.0000`
- adapter exact JSON match: `0.9940`
- adapter magnitude bucket accuracy: `0.9940`
- source metrics file: `outputs/evals/qwen36-27b-ibes-1k-on-10k-holdout/metrics.json`

### WRDS 10k minus 1k

- accuracy delta: `+0.0000`
- macro F1 delta: `+0.0000`
- parse failure delta: `+0.0000`
- exact JSON match delta: `+0.0050`
- magnitude bucket delta: `+0.0050`

## Public benchmarks

### FIQA

- 1k adapter accuracy: `0.8125`
- 10k adapter accuracy: `0.7969`
- 1k adapter macro F1: `0.6759`
- 10k adapter macro F1: `0.6597`
- 10k minus 1k accuracy delta: `-0.0156`
- 10k minus 1k macro F1 delta: `-0.0162`
- source metrics files: `outputs/evals/qwen36-27b-fiqa-nonthinking/metrics.json`, `outputs/evals/qwen36-27b-fiqa-10k-nonthinking/metrics.json`

### FPB

- 1k adapter accuracy: `0.7188`
- 10k adapter accuracy: `0.6406`
- 1k adapter macro F1: `0.7553`
- 10k adapter macro F1: `0.7125`
- 10k minus 1k accuracy delta: `-0.0781`
- 10k minus 1k macro F1 delta: `-0.0429`
- source metrics files: `outputs/evals/qwen36-27b-fpb-nonthinking/metrics.json`, `outputs/evals/qwen36-27b-fpb-10k-nonthinking/metrics.json`

### TFNS

- 1k adapter accuracy: `0.7500`
- 10k adapter accuracy: `0.6953`
- 1k adapter macro F1: `0.7419`
- 10k adapter macro F1: `0.6970`
- 10k minus 1k accuracy delta: `-0.0547`
- 10k minus 1k macro F1 delta: `-0.0449`
- source metrics files: `outputs/evals/qwen36-27b-tfns-1k-nonthinking/metrics.json`, `outputs/evals/qwen36-27b-tfns-10k-nonthinking/metrics.json`

### NWGI

- 1k adapter accuracy: `0.6094`
- 10k adapter accuracy: `0.6250`
- 1k adapter macro F1: `0.6023`
- 10k adapter macro F1: `0.6191`
- 10k minus 1k accuracy delta: `+0.0156`
- 10k minus 1k macro F1 delta: `+0.0169`
- source metrics files: `outputs/evals/qwen36-27b-nwgi-1k-nonthinking/metrics.json`, `outputs/evals/qwen36-27b-nwgi-10k-nonthinking/metrics.json`

## Recommendation

- Keep the non-thinking structured IBES path.
- Do not start 50k yet.
- Use CRSP daily returns plus the CRSP/Compustat link table as the next unblocker for market-reaction work.

