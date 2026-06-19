# Market-Reaction Milestone

This document defines the next milestone after the completed `10k` IBES evaluation.

The goal is not to claim alpha. The goal is to measure whether our event labels are useful for explaining realized market reactions.

## Expected input datasets

### 1. IBES gold events

Purpose:

- the event source table for the market-reaction panel
- carries the event timestamp plus gold labels and firm identifiers

Expected local path:

- preferred working path: `data/processed/ibes_lora_baseline/gold/ibes_revision_events.csv`
- if the current local artifact is parquet, export a narrow CSV/TSV/JSONL view first or run the validation in a parquet-capable environment

Expected columns:

- required: `event_id`, `event_date`, `label_direction`, `label_magnitude`
- required identifier set: at least one of `gvkey`, `cusip`, `ticker`
- recommended extras: source headline/text, fiscal period, company name

### 2. CRSP daily returns

Purpose:

- realized event-window return measurement

Expected local path:

- preferred: `admin/local/market-reaction/crsp_daily_returns_with_ids_1995-12-29_2006-01-16.zip`
- accepted fallback: `admin/local/market-reaction/crsp_daily_returns.csv`

Expected columns:

- `permno`
- `date`
- `ret`
- `prc`
- `shrout`
- `vol`

Preferred extra columns for fallback identifier mapping:

- `cusip`
- `ncusip`
- `ticker`
- `comnam`

### 3. CRSP/Compustat link table

Purpose:

- map event identifiers into CRSP `permno`

Expected local path:

- `admin/local/market-reaction/crsp_compustat_link.csv`

Expected columns:

- `gvkey`
- `permno`
- `linktype`
- `linkprim`
- `linkdt`
- `linkenddt`

Current status:

- still blocked in the current Cornell WRDS entitlement because the `crsp_m_ccm` query form is not available
- the pipeline must therefore support a lower-confidence non-CCM fallback path until a real link table arrives

### 4. CRSP stock header / name history

Purpose:

- fallback identifier mapping and audit trail when the formal CCM link table is unavailable

Expected local path:

- `admin/local/market-reaction/crsp_stock_header_full.csv`

Expected columns:

- `permno`
- `permco`
- `cusip`
- `htick`
- `hcomnam`
- `begdat`
- `enddat`

### 5. Market benchmark returns

Purpose:

- optional market-adjusted and abnormal-return measurement

Expected local path:

- `admin/local/market-reaction/market_benchmark_returns_1995-12-29_2006-01-16.csv`

Expected columns:

- `date`
- `benchmark_return`

Accepted aliases in the schema tools include common names such as `market_return`, `vwretd`, `ewretd`, and `sprtrn`.

## Join keys

Join priority:

1. `gvkey -> CRSP/Compustat link -> permno`
2. `cusip` or `cusip8` against CRSP `cusip` or `ncusip` from the enriched daily file
3. `ticker` against CRSP `ticker` or stock-header `htick`, with date-range checks
4. company-name matching only as an audit aid, not as a primary production join

Date logic:

- the event date must fall within the link interval `[linkdt, linkenddt]`
- if multiple link rows match, the eventual implementation must make the tie-break explicit and auditable
- for non-CCM fallback joins, date-aware filters should use stock-header `begdat` / `enddat` where available

Join audit fields to carry forward:

- `join_key_type`
- `join_method`
- `join_confidence`
- `join_status`

## Event-window definitions

Default windows:

- `0:1`
- `0:3`
- `0:5`
- `-1:1`

Interpretation:

- `0:1` means from event day close or same-day anchor through the next trading day
- `0:3` means event day through the third trading day after the event
- `0:5` means event day through the fifth trading day after the event
- `-1:1` means one trading day before through one trading day after the event

Planned output column naming:

- raw return: `raw_return_w_p0_p1`
- benchmark return: `benchmark_return_w_p0_p1`
- abnormal return: `abnormal_return_w_p0_p1`

The same naming pattern extends to the other windows, for example `raw_return_w_m1_p1`.

## Questions to answer

1. Do raw IBES labels correlate with realized event-window returns?
2. Do `1k` adapter predictions correlate with realized event-window returns?
3. Do `10k` adapter predictions correlate with realized event-window returns?
4. Does slightly better JSON and magnitude fidelity from the `10k` adapter matter downstream?
5. Which event windows are stable enough to present without overstating the result?

## Planned outputs

### Event panel

Expected output path:

- `data/processed/market_reaction/event_panel.csv`

Expected columns:

- `event_id`
- `event_date`
- `gvkey`
- `cusip`
- `ticker`
- `permno`
- `label_direction`
- `label_magnitude`
- `join_key_type`
- `join_method`
- `join_confidence`
- `join_status`

### Event-window panel

Expected output path:

- `data/processed/market_reaction/event_windows.csv`

Expected columns:

- all event-panel columns
- one raw return column per requested window
- one benchmark return column per requested window when benchmark data exists
- one abnormal return column per requested window when benchmark data exists
- any carried-through model label columns needed for downstream scoring

### Alignment report

Expected output path:

- `outputs/market_reaction/alignment_report/`

Expected contents:

- label-by-return summary tables
- hit-rate summaries
- positive-minus-negative return spread summaries
- market-adjusted return summaries when benchmark data exists
- toy strategy diagnostics with explicit caveats

### Phase 1: data join

- normalize the event table to one row per event
- map firm identifiers onto `permno`
- attach daily returns and benchmark returns

### Phase 2: label construction

- compute raw cumulative returns for selected windows
- compute benchmark-adjusted returns if the benchmark series is available
- keep both continuous outcomes and simple bucketed outcomes

### Phase 3: model comparison

- score the same event rows with raw IBES labels, `1k` adapter outputs, and `10k` adapter outputs
- compare directional agreement, magnitude agreement, and correlation with realized returns

### Phase 4: writeup

- document where the labels help
- document where they do not help
- avoid any unsupported trading claim

## Known limitations

- without CRSP data, the pipeline can only validate schemas and paths
- without the real CCM link table, joins into `permno` are fallback-only and less reliable
- benchmark-adjusted returns are unavailable until benchmark data exists locally
- current scripts are contract validators, not final research-grade backtests
- no alpha or trading claim is justified from sentiment classification metrics alone

## Guardrails

- do not fabricate missing market data
- do not report alpha
- do not report trading readiness
- do not report causality from small classification gains alone
- do not start `50k` training before this milestone clarifies the real bottleneck

## Success criteria

- the repo can build a reproducible joined event panel once the missing datasets arrive
- the comparison between raw IBES, `1k`, and `10k` outputs is measurable with clear metrics
- the next decision after this milestone is evidence-based rather than driven by more scale for its own sake
