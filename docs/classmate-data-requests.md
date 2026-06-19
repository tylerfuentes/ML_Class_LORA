# Classmate Data Requests

These are the exact requests to send for the next market-reaction milestone.

## Short request

Please upload the following WRDS exports into the shared `ML_Class_LORA_shared` drive so we can build the market-reaction panel for the next milestone:

1. `CRSP daily returns`
Required columns: `permno`, `date`, `ret`, `prc`, `shrout`, `vol`
Target folder: `ML_Class_LORA_shared/crsp/crsp_daily_returns/`

2. `CRSP/Compustat link table`
Required columns: `gvkey`, `permno`, `linktype`, `linkprim`, `linkdt`, `linkenddt`
Target folder: `ML_Class_LORA_shared/crsp/crsp_compustat_link/`

3. `Market benchmark returns`, if available
Required columns: `date` plus the benchmark return field
Target folder: `ML_Class_LORA_shared/crsp/market_benchmarks/`

If you already have Compustat fundamentals extracted too, upload those separately under `ML_Class_LORA_shared/compustat/fundamentals/`, but that is optional for the first join pass.

## Longer request with context

We finished the completed `10k` IBES adapter evaluation. The result is that the `10k` adapter is the best narrow structured IBES JSON specialist, while the earlier `1k` adapter is still the best general finance adapter on public benchmarks. Because of that, the next milestone is not more training. The next milestone is testing whether these event labels line up with realized market reactions.

To unblock that, please export and upload:

1. `CRSP daily returns`
Columns: `permno`, `date`, `ret`, `prc`, `shrout`, `vol`
Purpose: event-window returns such as `t+0 to t+1`, `t+0 to t+3`, and `t+0 to t+5`
Folder: `ML_Class_LORA_shared/crsp/crsp_daily_returns/`

2. `CRSP/Compustat link table`
Columns: `gvkey`, `permno`, `linktype`, `linkprim`, `linkdt`, `linkenddt`
Purpose: robust identifier joins from firm/event records into CRSP
Folder: `ML_Class_LORA_shared/crsp/crsp_compustat_link/`

3. `Market benchmark returns`, if available
Columns: `date` and a benchmark return series
Purpose: market-adjusted or abnormal-return variants
Folder: `ML_Class_LORA_shared/crsp/market_benchmarks/`

4. `Compustat fundamentals`, optional
Suggested fields: core identifiers plus a small fundamentals slice
Purpose: later firm-context enrichment after the returns join works
Folder: `ML_Class_LORA_shared/compustat/fundamentals/`

Important constraint: we do not need anyone to build labels manually. We only need the raw exports in shared storage so the repo pipeline can build the joins and event-window measurements locally.
