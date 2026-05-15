#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no}/{primary_doc}"


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data and not data.isspace():
            self.parts.append(data)

    def text(self) -> str:
        raw = " ".join(self.parts)
        return re.sub(r"\s+", " ", raw).strip()


def http_get_json(url: str, user_agent: str) -> Any:
    request = Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    with urlopen(request, timeout=30) as response:
        payload = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            payload = gzip.decompress(payload)
        return json.loads(payload.decode("utf-8"))


def http_get_text(url: str, user_agent: str) -> str:
    request = Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    with urlopen(request, timeout=30) as response:
        payload = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            payload = gzip.decompress(payload)
        charset = response.headers.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")


def build_ticker_map(user_agent: str) -> dict[str, dict[str, str]]:
    payload = http_get_json(SEC_TICKER_MAP_URL, user_agent)
    mapping: dict[str, dict[str, str]] = {}
    for _, entry in payload.items():
        ticker = entry["ticker"].upper()
        mapping[ticker] = {
            "cik": str(entry["cik_str"]).zfill(10),
            "title": entry["title"],
        }
    return mapping


def filing_rows(submissions: dict[str, Any], form: str) -> list[dict[str, str]]:
    recent = submissions["filings"]["recent"]
    rows = []
    fields = [
        "accessionNumber",
        "filingDate",
        "acceptanceDateTime",
        "form",
        "primaryDocument",
        "primaryDocDescription",
    ]
    for idx, filing_form in enumerate(recent["form"]):
        if filing_form != form:
            continue
        row = {field: recent[field][idx] for field in fields}
        rows.append(row)
    return rows


def html_to_text(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    text = parser.text()
    markers = [
        "UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
        "FORM 8-K",
        "FORM 10-K",
        "FORM 10-Q",
    ]
    for marker in markers:
        idx = text.find(marker)
        if idx != -1:
            return text[idx:]
    return text


def fetch_filings(
    ticker: str,
    ticker_meta: dict[str, str],
    form: str,
    limit: int,
    user_agent: str,
    pause_seconds: float,
    max_chars: int,
) -> list[dict[str, Any]]:
    cik = ticker_meta["cik"]
    submissions = http_get_json(SEC_SUBMISSIONS_URL.format(cik=cik), user_agent)
    rows = filing_rows(submissions, form)[:limit]
    records: list[dict[str, Any]] = []

    for row in rows:
        accession_no = row["accessionNumber"].replace("-", "")
        filing_url = SEC_ARCHIVES_URL.format(
            cik_int=str(int(cik)),
            accession_no=accession_no,
            primary_doc=row["primaryDocument"],
        )
        html = http_get_text(filing_url, user_agent)
        text = html_to_text(html)
        records.append(
            {
                "source": "SEC EDGAR",
                "ticker": ticker,
                "company": ticker_meta["title"],
                "cik": cik,
                "form": row["form"],
                "filing_date": row["filingDate"],
                "acceptance_datetime": row["acceptanceDateTime"],
                "accession_number": row["accessionNumber"],
                "primary_document": row["primaryDocument"],
                "primary_doc_description": row["primaryDocDescription"],
                "filing_url": filing_url,
                "text_excerpt": text[:max_chars],
                "text_truncated": len(text) > max_chars,
                "text_length_chars": len(text),
            }
        )
        time.sleep(pause_seconds)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", required=True, help="One or more US equity tickers.")
    parser.add_argument("--form", default="8-K", help="SEC form to fetch, default: 8-K")
    parser.add_argument("--limit-per-ticker", type=int, default=1, help="Number of filings per ticker.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--pause-seconds", type=float, default=0.2, help="Pause between filing fetches.")
    parser.add_argument("--max-chars", type=int, default=4000, help="Max filing text chars to keep per row.")
    parser.add_argument(
        "--user-agent",
        default=os.environ.get("SEC_USER_AGENT"),
        help="SEC-compliant User-Agent. Prefer name + contact email. Can also be set via SEC_USER_AGENT.",
    )
    args = parser.parse_args()

    if not args.user_agent:
        raise SystemExit(
            "Missing SEC User-Agent. Pass --user-agent 'Name email@example.com' "
            "or set SEC_USER_AGENT."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ticker_map = build_ticker_map(args.user_agent)
    all_records: list[dict[str, Any]] = []

    for raw_ticker in args.tickers:
        ticker = raw_ticker.upper()
        if ticker not in ticker_map:
            raise SystemExit(f"Ticker not found in SEC company map: {ticker}")
        try:
            records = fetch_filings(
                ticker=ticker,
                ticker_meta=ticker_map[ticker],
                form=args.form,
                limit=args.limit_per_ticker,
                user_agent=args.user_agent,
                pause_seconds=args.pause_seconds,
                max_chars=args.max_chars,
            )
        except (HTTPError, URLError) as exc:
            raise SystemExit(f"Failed to fetch SEC data for {ticker}: {exc}") from exc
        all_records.extend(records)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Wrote {len(all_records)} records to {output_path}")


if __name__ == "__main__":
    main()
