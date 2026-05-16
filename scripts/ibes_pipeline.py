#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

EXPECTED_IBES_COLUMNS = [
    "TICKER",
    "CUSIP",
    "OFTIC",
    "CNAME",
    "ACTDATS",
    "ESTIMATOR",
    "ANALYS",
    "CURRFL",
    "PDF",
    "FPI",
    "MEASURE",
    "VALUE",
    "CURR",
    "USFIRM",
    "FPEDATS",
    "ACTTIMS",
    "REVDATS",
    "REVTIMS",
    "ANNDATS",
    "ANNTIMS",
    "report_curr",
]

DEFAULT_RAW_IBES = Path("admin/local/wrds-downloads/tr_ibes_11289435.csv")


@dataclass(frozen=True)
class SplitSpec:
    name: str
    train: int
    eval: int
    holdout: int

    @property
    def total(self) -> int:
        return self.train + self.eval + self.holdout


BASELINE_1K = SplitSpec(name="baseline_1k", train=800, eval=100, holdout=100)
BASELINE_10K = SplitSpec(name="baseline_10k", train=10_000, eval=1_000, holdout=1_000)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def load_csv_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def load_raw_ibes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype="string",
        keep_default_na=False,
        na_filter=False,
        low_memory=False,
    )
    missing = [column for column in EXPECTED_IBES_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected IBES columns: {missing}")
    return df[EXPECTED_IBES_COLUMNS].copy()


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA})


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(clean_text(series), errors="coerce")


def clean_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(clean_text(series), errors="coerce").dt.normalize()


def first_valid(series: pd.Series):
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else pd.NA


def build_bronze(df: pd.DataFrame) -> pd.DataFrame:
    bronze = df.copy()

    for column in [
        "TICKER",
        "CUSIP",
        "OFTIC",
        "CNAME",
        "ESTIMATOR",
        "ANALYS",
        "CURRFL",
        "PDF",
        "FPI",
        "MEASURE",
        "CURR",
        "USFIRM",
        "FPEDATS",
        "ACTTIMS",
        "REVDATS",
        "REVTIMS",
        "ANNDATS",
        "ANNTIMS",
        "report_curr",
    ]:
        bronze[column] = clean_text(bronze[column])

    bronze["VALUE"] = clean_numeric(bronze["VALUE"])
    bronze["ACTDATS"] = clean_date(bronze["ACTDATS"])
    bronze["REVDATS"] = clean_date(bronze["REVDATS"])
    bronze["ANNDATS"] = clean_date(bronze["ANNDATS"])
    bronze["FPEDATS"] = clean_date(bronze["FPEDATS"])

    bronze["ticker_norm"] = bronze["TICKER"].str.upper()
    bronze["oftic_norm"] = bronze["OFTIC"].str.upper()
    bronze["cusip8"] = bronze["CUSIP"].str.upper().str.slice(0, 8)
    bronze["company_name_norm"] = bronze["CNAME"].str.upper()
    bronze["measure_norm"] = bronze["MEASURE"].str.upper()
    bronze["currfl_norm"] = bronze["CURRFL"].str.upper()
    bronze["pdf_norm"] = bronze["PDF"].str.upper()
    bronze["fpi_norm"] = bronze["FPI"].str.upper()
    bronze["report_currency_norm"] = bronze["report_curr"].str.upper()
    bronze["estimate_currency_norm"] = bronze["CURR"].str.upper()
    bronze["usfirm_flag"] = clean_numeric(bronze["USFIRM"]).astype("Int64")
    bronze["event_date"] = bronze["REVDATS"].fillna(bronze["ANNDATS"]).fillna(bronze["ACTDATS"])
    bronze["event_time"] = bronze["REVTIMS"].fillna(bronze["ANNTIMS"]).fillna(bronze["ACTTIMS"])
    bronze["company_key"] = bronze["oftic_norm"].fillna(bronze["ticker_norm"]).fillna(bronze["cusip8"])
    return bronze


def build_silver(bronze: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    filters = {
        "rows_in": int(len(bronze)),
        "eps_rows": int(bronze["measure_norm"].eq("EPS").sum()),
        "us_firm_rows": int(bronze["usfirm_flag"].eq(1).sum()),
        "current_rows_assuming_blank_is_current": int(
            bronze["currfl_norm"].fillna("C").eq("C").sum()
        ),
        "usd_rows_assuming_blank_is_usd": int(
            bronze["report_currency_norm"].fillna("USD").eq("USD").sum()
        ),
        "rows_with_event_date": int(bronze["event_date"].notna().sum()),
        "rows_with_numeric_value": int(bronze["VALUE"].notna().sum()),
    }

    mask = (
        bronze["measure_norm"].eq("EPS")
        & bronze["usfirm_flag"].eq(1)
        & bronze["currfl_norm"].fillna("C").eq("C")
        & bronze["report_currency_norm"].fillna("USD").eq("USD")
        & bronze["event_date"].notna()
        & bronze["VALUE"].notna()
        & bronze["company_key"].notna()
    )
    silver = bronze.loc[mask].copy()

    duplicate_subset = [
        "company_key",
        "ticker_norm",
        "oftic_norm",
        "cusip8",
        "ANALYS",
        "ESTIMATOR",
        "FPI",
        "FPEDATS",
        "event_date",
        "VALUE",
    ]
    duplicate_count = int(silver.duplicated(subset=duplicate_subset).sum())
    silver = silver.drop_duplicates(subset=duplicate_subset, keep="first").copy()

    silver = silver.rename(
        columns={
            "ANALYS": "analyst_id",
            "ESTIMATOR": "estimator_id",
            "VALUE": "estimate_value",
            "ACTDATS": "actual_date",
            "REVDATS": "revision_date",
            "ANNDATS": "announce_date",
            "FPEDATS": "fiscal_period_end_date",
            "ACTTIMS": "actual_time",
            "REVTIMS": "revision_time",
            "ANNTIMS": "announce_time",
        }
    )

    silver["analyst_id"] = clean_text(silver["analyst_id"])
    silver["estimator_id"] = clean_text(silver["estimator_id"])

    stats = {
        **filters,
        "rows_after_filters": int(len(silver)),
        "duplicate_rows_removed": duplicate_count,
    }
    return silver, stats


def assign_direction_label(delta: float | None, neutral_abs_delta: float) -> str:
    if delta is None or pd.isna(delta):
        return "neutral"
    if abs(float(delta)) < neutral_abs_delta:
        return "neutral"
    return "positive" if float(delta) > 0 else "negative"


def assign_magnitude_bucket(
    delta: float | None, pct_change: float | None, neutral_abs_delta: float
) -> str:
    if delta is None or pd.isna(delta):
        return "unknown"
    abs_delta = abs(float(delta))
    if abs_delta < neutral_abs_delta:
        return "flat"
    if pct_change is not None and not pd.isna(pct_change):
        pct_value = abs(float(pct_change))
        if pct_value < 0.05:
            return "small"
        if pct_value < 0.15:
            return "medium"
        return "large"
    if abs_delta < 0.05:
        return "small"
    if abs_delta < 0.25:
        return "medium"
    return "large"


def build_gold(silver: pd.DataFrame, neutral_abs_delta: float = 0.005) -> tuple[pd.DataFrame, dict]:
    group_cols = [
        "company_key",
        "ticker_norm",
        "oftic_norm",
        "cusip8",
        "company_name_norm",
        "event_date",
        "fpi_norm",
        "fiscal_period_end_date",
        "measure_norm",
        "report_currency_norm",
    ]

    gold = (
        silver.groupby(group_cols, dropna=False)
        .agg(
            estimate_count=("estimate_value", "size"),
            analyst_count=("analyst_id", lambda s: int(s.dropna().nunique())),
            estimator_count=("estimator_id", lambda s: int(s.dropna().nunique())),
            consensus_mean=("estimate_value", "mean"),
            consensus_median=("estimate_value", "median"),
            consensus_std=("estimate_value", "std"),
            consensus_min=("estimate_value", "min"),
            consensus_max=("estimate_value", "max"),
            actual_date=("actual_date", "min"),
            announce_date=("announce_date", "min"),
            revision_date=("revision_date", "min"),
            event_time=("event_time", first_valid),
        )
        .reset_index()
    )

    gold = gold.sort_values(
        by=["company_key", "fpi_norm", "fiscal_period_end_date", "event_date"],
        kind="mergesort",
    ).reset_index(drop=True)
    history_key = (
        gold["company_key"].fillna("UNK")
        + "|"
        + gold["fpi_norm"].fillna("UNK")
        + "|"
        + gold["fiscal_period_end_date"].astype("string").fillna("UNK")
    )
    gold["history_key"] = history_key
    gold["prior_consensus_median"] = gold.groupby("history_key")["consensus_median"].shift(1)
    gold["prior_consensus_mean"] = gold.groupby("history_key")["consensus_mean"].shift(1)
    gold["consensus_delta"] = gold["consensus_median"] - gold["prior_consensus_median"]
    gold["consensus_pct_change"] = gold["consensus_delta"] / gold["prior_consensus_median"].abs()
    gold.loc[gold["prior_consensus_median"].abs() < 1e-9, "consensus_pct_change"] = pd.NA
    gold["direction_label"] = gold["consensus_delta"].apply(
        lambda value: assign_direction_label(value, neutral_abs_delta)
    )
    gold["magnitude_bucket"] = [
        assign_magnitude_bucket(delta, pct_change, neutral_abs_delta)
        for delta, pct_change in zip(gold["consensus_delta"], gold["consensus_pct_change"])
    ]
    gold["event_type"] = "ibes_analyst_revision_consensus"
    gold["event_id"] = (
        gold["company_key"].fillna("UNK")
        + "|"
        + gold["event_date"].astype("string").fillna("UNK")
        + "|"
        + gold["fpi_norm"].fillna("UNK")
        + "|"
        + gold["fiscal_period_end_date"].astype("string").fillna("UNK")
    )
    gold = gold.drop(columns=["history_key"])

    stats = {
        "rows_in": int(len(silver)),
        "event_rows_out": int(len(gold)),
        "rows_with_prior_consensus": int(gold["prior_consensus_median"].notna().sum()),
        "positive_labels": int(gold["direction_label"].eq("positive").sum()),
        "negative_labels": int(gold["direction_label"].eq("negative").sum()),
        "neutral_labels": int(gold["direction_label"].eq("neutral").sum()),
    }
    return gold, stats


def _value_or_none(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, float):
        return round(value, 6)
    return value


def build_lora_row(row: pd.Series) -> dict:
    structured_input = {
        "task": "classify analyst revision event",
        "company": {
            "company_key": _value_or_none(row["company_key"]),
            "ticker": _value_or_none(row["ticker_norm"]),
            "oftic": _value_or_none(row["oftic_norm"]),
            "cusip8": _value_or_none(row["cusip8"]),
            "company_name": _value_or_none(row["company_name_norm"]),
        },
        "event": {
            "event_id": _value_or_none(row["event_id"]),
            "event_type": _value_or_none(row["event_type"]),
            "event_date": _value_or_none(row["event_date"]),
            "event_time": _value_or_none(row["event_time"]),
            "fiscal_period_end_date": _value_or_none(row["fiscal_period_end_date"]),
            "fpi": _value_or_none(row["fpi_norm"]),
            "currency": _value_or_none(row["report_currency_norm"]),
        },
        "features": {
            "estimate_count": _value_or_none(row["estimate_count"]),
            "analyst_count": _value_or_none(row["analyst_count"]),
            "estimator_count": _value_or_none(row["estimator_count"]),
            "consensus_mean": _value_or_none(row["consensus_mean"]),
            "consensus_median": _value_or_none(row["consensus_median"]),
            "consensus_std": _value_or_none(row["consensus_std"]),
            "consensus_min": _value_or_none(row["consensus_min"]),
            "consensus_max": _value_or_none(row["consensus_max"]),
            "prior_consensus_mean": _value_or_none(row["prior_consensus_mean"]),
            "prior_consensus_median": _value_or_none(row["prior_consensus_median"]),
            "consensus_delta": _value_or_none(row["consensus_delta"]),
            "consensus_pct_change": _value_or_none(row["consensus_pct_change"]),
        },
    }
    structured_output = {
        "direction_label": _value_or_none(row["direction_label"]),
        "magnitude_bucket": _value_or_none(row["magnitude_bucket"]),
        "event_type": _value_or_none(row["event_type"]),
    }
    return {
        "instruction": (
            "Given the structured analyst revision event, classify the consensus revision "
            "direction and magnitude bucket."
        ),
        "input": json.dumps(structured_input, sort_keys=True),
        "output": json.dumps(structured_output, sort_keys=True),
        "event_id": _value_or_none(row["event_id"]),
        "source": "wrds_tr_ibes",
    }


def write_jsonl(rows: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def sample_gold_splits(gold: pd.DataFrame, split: SplitSpec, seed: int) -> dict[str, pd.DataFrame]:
    if len(gold) < split.total:
        raise ValueError(
            f"Need at least {split.total} gold events for {split.name}, but only found {len(gold)}."
        )
    sample = gold.sample(n=split.total, random_state=seed).reset_index(drop=True)
    return {
        "train": sample.iloc[: split.train].copy(),
        "eval": sample.iloc[split.train : split.train + split.eval].copy(),
        "holdout": sample.iloc[split.train + split.eval :].copy(),
    }


def key_missingness(df: pd.DataFrame, columns: Iterable[str]) -> dict[str, float]:
    rows = max(len(df), 1)
    return {
        column: round(float(df[column].isna().sum()) / rows, 6)
        for column in columns
        if column in df.columns
    }


def dataset_date_range(df: pd.DataFrame, column: str) -> dict[str, str | None]:
    if column not in df.columns:
        return {"min": None, "max": None}
    non_null = df[column].dropna()
    if non_null.empty:
        return {"min": None, "max": None}
    return {
        "min": non_null.min().date().isoformat(),
        "max": non_null.max().date().isoformat(),
    }
