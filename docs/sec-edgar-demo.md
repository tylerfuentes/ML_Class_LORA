# SEC EDGAR Demo

This repo includes a small script for pulling public SEC EDGAR filing data into JSONL so the team can inspect real event text instead of only synthetic or screenshot-backed examples.

## Script

`scripts/fetch_sec_edgar_filings.py`

What it does:

- resolves tickers to CIKs using the SEC company ticker map
- fetches recent company submissions from `data.sec.gov`
- filters by form type, defaulting to `8-K`
- downloads the filing document from the SEC archives
- strips HTML to plain text
- writes one JSONL row per filing

## SEC access notes

Use a real `User-Agent` string with identifying contact information. The SEC developer documentation and API docs say automated access must comply with SEC fair-access guidance.

Official references:

- https://www.sec.gov/about/developer-resources
- https://www.sec.gov/search-filings/edgar-application-programming-interfaces

## Example

```bash
cd /home/nathanaelguitar/ML_Class_LORA
python scripts/fetch_sec_edgar_filings.py \
  --tickers MSFT NVDA \
  --form 8-K \
  --limit-per-ticker 1 \
  --output data/public/sec_recent_8k_demo.jsonl \
  --user-agent "ML_Class_LORA class project contact: nathanaelguitar"
```

## Output schema

Each JSONL row includes:

- `ticker`
- `company`
- `cik`
- `form`
- `filing_date`
- `acceptance_datetime`
- `accession_number`
- `primary_document`
- `primary_doc_description`
- `filing_url`
- `text_excerpt`
- `text_truncated`
- `text_length_chars`

This is a data-ingest step, not yet a final Qwen training row. The next step would be converting these filings into reasoning-oriented `instruction` / `input` / `output` JSONL examples once the team decides the annotation format.
