# Market-Reaction Validation Tools

These scripts are schema-aware validation and planning tools for the next repo milestone.

They do not:

- fabricate WRDS or CRSP data
- write fake joined outputs
- claim alpha
- start any new training

They do:

- validate expected files and columns
- print sample rows without loading full datasets
- define the planned join and event-window contracts
- fail clearly when required inputs are missing

## Scripts

- `check_market_data.py`
  - validate CRSP daily returns, CRSP/Compustat link, and optional benchmark files
- `build_event_panel.py`
  - validate the IBES gold event file plus market-data inputs and print the join plan
- `compute_event_windows.py`
  - validate the event-panel and return-window contract
- `score_label_alignment.py`
  - validate the final evaluation contract between labels and realized returns

## Synthetic smoke commands

```bash
python3 scripts/market_reaction/check_market_data.py \
  --crsp-daily-returns data/samples/market_reaction/crsp_daily_returns_sample.csv \
  --crsp-compustat-link data/samples/market_reaction/crsp_compustat_link_sample.csv \
  --market-benchmark-returns data/samples/market_reaction/market_benchmark_returns_sample.csv

python3 scripts/market_reaction/build_event_panel.py \
  --ibes-gold-events data/samples/market_reaction/ibes_gold_events_sample.csv \
  --crsp-daily-returns data/samples/market_reaction/crsp_daily_returns_sample.csv \
  --crsp-compustat-link data/samples/market_reaction/crsp_compustat_link_sample.csv \
  --market-benchmark-returns data/samples/market_reaction/market_benchmark_returns_sample.csv \
  --output-path data/samples/market_reaction/planned_event_panel.csv

python3 scripts/market_reaction/compute_event_windows.py \
  --event-panel data/samples/market_reaction/event_panel_sample.csv \
  --crsp-daily data/samples/market_reaction/crsp_daily_returns_sample.csv \
  --benchmark-returns data/samples/market_reaction/market_benchmark_returns_sample.csv \
  --windows 0:1 0:3 0:5 -1:1 \
  --output-file data/samples/market_reaction/planned_event_windows.csv

python3 scripts/market_reaction/score_label_alignment.py \
  --event-window-file data/samples/market_reaction/event_windows_sample.csv \
  --label-columns gold_direction_label adapter_1k_direction_label adapter_10k_direction_label \
  --windows 0:1 0:3 0:5 -1:1 \
  --output-dir data/samples/market_reaction/planned_alignment_report
```
