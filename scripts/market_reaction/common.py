#!/usr/bin/env python3
"""Shared schema and validation helpers for market-reaction planning tools."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_IBES_GOLD_EVENTS = Path("data/processed/ibes_lora_baseline/gold/ibes_revision_events.csv")
DEFAULT_CRSP_DAILY_RETURNS = Path("admin/local/market-reaction/crsp_daily_returns.csv")
DEFAULT_CRSP_LINK = Path("admin/local/market-reaction/crsp_compustat_link.csv")
DEFAULT_BENCHMARK_RETURNS = Path("admin/local/market-reaction/market_benchmark_returns.csv")
DEFAULT_EVENT_PANEL = Path("data/processed/market_reaction/event_panel.csv")
DEFAULT_EVENT_WINDOWS = Path("data/processed/market_reaction/event_windows.csv")
DEFAULT_ALIGNMENT_REPORT = Path("outputs/market_reaction/alignment_report")

WINDOW_DEFAULTS = ["0:1", "0:3", "0:5", "-1:1"]

SUPPORTED_TABULAR_EXTENSIONS = {".csv", ".tsv", ".jsonl", ".ndjson"}

IBES_REQUIRED_ALIASES = {
    "event_id": ("event_id", "id", "example_id"),
    "event_date": ("event_date", "announcement_date", "date", "anndats"),
    "label_direction": ("label_direction", "direction", "sentiment_label", "label"),
    "label_magnitude": ("label_magnitude", "magnitude", "magnitude_bucket"),
}

IBES_IDENTIFIER_ALIASES = {
    "gvkey": ("gvkey",),
    "cusip": ("cusip", "cusip9", "cusip8"),
    "ticker": ("ticker", "ticker_symbol", "oftic"),
}

CRSP_DAILY_ALIASES = {
    "permno": ("permno",),
    "date": ("date", "trading_date"),
    "ret": ("ret", "return"),
    "prc": ("prc", "price"),
    "shrout": ("shrout", "shares_outstanding"),
    "vol": ("vol", "volume"),
}

CRSP_LINK_ALIASES = {
    "gvkey": ("gvkey",),
    "permno": ("permno",),
    "linktype": ("linktype",),
    "linkprim": ("linkprim",),
    "linkdt": ("linkdt", "link_start_date"),
    "linkenddt": ("linkenddt", "link_end_date"),
}

BENCHMARK_ALIASES = {
    "date": ("date", "trading_date"),
    "benchmark_return": ("benchmark_return", "market_return", "vwretd", "ewretd", "sprtrn"),
}

EVENT_PANEL_REQUIRED_ALIASES = {
    "event_id": ("event_id",),
    "event_date": ("event_date",),
    "permno": ("permno",),
    "label_direction": ("label_direction", "gold_direction_label"),
    "label_magnitude": ("label_magnitude", "gold_magnitude_label"),
    "join_key_type": ("join_key_type",),
    "join_status": ("join_status",),
}

EVENT_WINDOWS_REQUIRED_BASE_ALIASES = {
    "event_id": ("event_id",),
    "event_date": ("event_date",),
    "permno": ("permno",),
}

DEFAULT_LABEL_COLUMNS = [
    "gold_direction_label",
    "adapter_1k_direction_label",
    "adapter_10k_direction_label",
]


@dataclass
class InspectionResult:
    path: Path
    exists: bool
    file_format: str | None
    size_bytes: int | None
    columns: list[str]
    sample_rows: list[dict[str, str]]
    notes: list[str]


class ValidationIssue(Exception):
    """Raised for clear validation failures."""


def status_prefix(level: str) -> str:
    return f"[{level}]"


def format_bytes(size: int | None) -> str:
    if size is None:
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    return f"{value:.1f} {unit}"


def detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".jsonl", ".ndjson"}:
        return suffix.lstrip(".")
    if suffix == ".parquet":
        raise ValidationIssue(
            f"Unsupported file format for stdlib-only schema validation: {path}. "
            "Export a CSV/TSV/JSONL header view or run in an environment with parquet tooling."
        )
    raise ValidationIssue(
        f"Unsupported file format for schema validation: {path}. "
        f"Supported extensions: {', '.join(sorted(SUPPORTED_TABULAR_EXTENSIONS))}."
    )


def sniff_delimiter(sample: str, fallback: str = ",") -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|")
        return dialect.delimiter
    except csv.Error:
        return fallback


def inspect_tabular_file(path: Path, sample_rows: int = 3) -> InspectionResult:
    if not path.exists():
        return InspectionResult(path, False, None, None, [], [], [f"missing file: {path}"])

    file_format = detect_format(path)
    size_bytes = path.stat().st_size
    notes: list[str] = []

    if file_format in {"csv", "tsv"}:
        with path.open("r", encoding="utf-8", newline="") as handle:
            preface = handle.read(8192)
            handle.seek(0)
            delimiter = "\t" if file_format == "tsv" else sniff_delimiter(preface)
            reader = csv.DictReader(handle, delimiter=delimiter)
            columns = reader.fieldnames or []
            rows: list[dict[str, str]] = []
            for _, row in zip(range(sample_rows), reader):
                rows.append({key: value for key, value in row.items() if key is not None})
        notes.append(f"delimiter={repr(delimiter)}")
        return InspectionResult(path, True, file_format, size_bytes, columns, rows, notes)

    rows = []
    columns: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValidationIssue(f"Expected JSON object lines in {path} but found: {type(payload).__name__}")
            if not columns:
                columns = list(payload.keys())
            rows.append({str(key): str(value) for key, value in payload.items()})
            if len(rows) >= sample_rows:
                break
        if not columns:
            raise ValidationIssue(f"No JSON objects found in {path}")
    return InspectionResult(path, True, file_format, size_bytes, columns, rows, notes)


def resolve_aliases(columns: Iterable[str], alias_map: dict[str, tuple[str, ...]]) -> tuple[dict[str, str], list[str]]:
    column_lookup = {column.lower(): column for column in columns}
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for canonical, aliases in alias_map.items():
        match = next((column_lookup[alias.lower()] for alias in aliases if alias.lower() in column_lookup), None)
        if match is None:
            missing.append(canonical)
        else:
            resolved[canonical] = match
    return resolved, missing


def require_columns(columns: Iterable[str], alias_map: dict[str, tuple[str, ...]], label: str) -> dict[str, str]:
    resolved, missing = resolve_aliases(columns, alias_map)
    if missing:
        missing_text = ", ".join(missing)
        raise ValidationIssue(f"{label} is missing required columns: {missing_text}")
    return resolved


def parse_window_spec(spec: str) -> tuple[int, int]:
    if ":" not in spec:
        raise ValidationIssue(f"Invalid window spec '{spec}'. Expected format start:end, for example 0:1.")
    start_text, end_text = spec.split(":", 1)
    try:
        start = int(start_text)
        end = int(end_text)
    except ValueError as exc:
        raise ValidationIssue(f"Invalid window spec '{spec}'. Window bounds must be integers.") from exc
    if start > end:
        raise ValidationIssue(f"Invalid window spec '{spec}'. Start must be less than or equal to end.")
    return start, end


def window_slug(spec: str) -> str:
    start, end = parse_window_spec(spec)
    return f"w_{signed_token(start)}_{signed_token(end)}"


def signed_token(value: int) -> str:
    if value < 0:
        return f"m{abs(value)}"
    return f"p{value}"


def raw_return_column(spec: str) -> str:
    return f"raw_return_{window_slug(spec)}"


def benchmark_return_column(spec: str) -> str:
    return f"benchmark_return_{window_slug(spec)}"


def abnormal_return_column(spec: str) -> str:
    return f"abnormal_return_{window_slug(spec)}"


def hit_rate_column(spec: str) -> str:
    return f"direction_hit_{window_slug(spec)}"


def print_inspection(label: str, result: InspectionResult) -> None:
    print(f"{status_prefix('INFO')} {label}: {result.path}")
    if not result.exists:
        print(f"{status_prefix('FAIL')} file does not exist")
        return
    print(f"{status_prefix('INFO')} format={result.file_format} size={format_bytes(result.size_bytes)}")
    if result.notes:
        print(f"{status_prefix('INFO')} notes={'; '.join(result.notes)}")
    print(f"{status_prefix('INFO')} columns={', '.join(result.columns)}")
    if result.sample_rows:
        print(f"{status_prefix('INFO')} sample_rows={json.dumps(result.sample_rows, indent=2)}")

