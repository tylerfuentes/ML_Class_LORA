#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window

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


def spark_master() -> str:
    return "local[*]"


def create_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master(spark_master())
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )


def java_version() -> str | None:
    try:
        proc = subprocess.run(
            ["java", "-version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    output = (proc.stderr or proc.stdout).splitlines()
    return output[0] if output else None


def clean_text_column(column_name: str):
    return F.when(F.trim(F.col(column_name)) == "", F.lit(None)).otherwise(F.trim(F.col(column_name)))


def load_raw_ibes_spark(spark: SparkSession, path: Path) -> DataFrame:
    df = (
        spark.read.option("header", True)
        .option("multiLine", False)
        .option("mode", "FAILFAST")
        .csv(str(path))
    )
    missing = [column for column in EXPECTED_IBES_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected IBES columns: {missing}")
    return df.select(*EXPECTED_IBES_COLUMNS)


def build_bronze(df: DataFrame) -> DataFrame:
    bronze = df
    text_columns = [
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
    ]
    for column in text_columns:
        bronze = bronze.withColumn(column, clean_text_column(column))

    bronze = (
        bronze.withColumn("VALUE", F.col("VALUE").cast("double"))
        .withColumn("ACTDATS", F.to_date("ACTDATS"))
        .withColumn("REVDATS", F.to_date("REVDATS"))
        .withColumn("ANNDATS", F.to_date("ANNDATS"))
        .withColumn("FPEDATS", F.to_date("FPEDATS"))
        .withColumn("ticker_norm", F.upper(F.col("TICKER")))
        .withColumn("oftic_norm", F.upper(F.col("OFTIC")))
        .withColumn("cusip8", F.upper(F.substring(F.col("CUSIP"), 1, 8)))
        .withColumn("company_name_norm", F.upper(F.col("CNAME")))
        .withColumn("measure_norm", F.upper(F.col("MEASURE")))
        .withColumn("currfl_norm", F.upper(F.col("CURRFL")))
        .withColumn("pdf_norm", F.upper(F.col("PDF")))
        .withColumn("fpi_norm", F.upper(F.col("FPI")))
        .withColumn("report_currency_norm", F.upper(F.col("report_curr")))
        .withColumn("estimate_currency_norm", F.upper(F.col("CURR")))
        .withColumn("usfirm_flag", F.col("USFIRM").cast("int"))
        .withColumn("event_date", F.coalesce(F.col("REVDATS"), F.col("ANNDATS"), F.col("ACTDATS")))
        .withColumn("event_time", F.coalesce(F.col("REVTIMS"), F.col("ANNTIMS"), F.col("ACTTIMS")))
        .withColumn("company_key", F.coalesce(F.col("oftic_norm"), F.col("ticker_norm"), F.col("cusip8")))
    )
    return bronze


def build_silver(bronze: DataFrame) -> tuple[DataFrame, dict]:
    rows_in = bronze.count()
    eps_rows = bronze.filter(F.col("measure_norm") == "EPS").count()
    us_firm_rows = bronze.filter(F.col("usfirm_flag") == 1).count()
    current_rows = bronze.filter(F.coalesce(F.col("currfl_norm"), F.lit("C")) == "C").count()
    usd_rows = bronze.filter(F.coalesce(F.col("report_currency_norm"), F.lit("USD")) == "USD").count()
    rows_with_event_date = bronze.filter(F.col("event_date").isNotNull()).count()
    rows_with_numeric_value = bronze.filter(F.col("VALUE").isNotNull()).count()

    silver = bronze.filter(
        (F.col("measure_norm") == "EPS")
        & (F.col("usfirm_flag") == 1)
        & (F.coalesce(F.col("currfl_norm"), F.lit("C")) == "C")
        & (F.coalesce(F.col("report_currency_norm"), F.lit("USD")) == "USD")
        & F.col("event_date").isNotNull()
        & F.col("VALUE").isNotNull()
        & F.col("company_key").isNotNull()
    )

    before_dedup = silver.count()
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
    silver = silver.dropDuplicates(duplicate_subset)
    after_dedup = silver.count()

    silver = silver.withColumnRenamed("ANALYS", "analyst_id")
    silver = silver.withColumnRenamed("ESTIMATOR", "estimator_id")
    silver = silver.withColumnRenamed("VALUE", "estimate_value")
    silver = silver.withColumnRenamed("ACTDATS", "actual_date")
    silver = silver.withColumnRenamed("REVDATS", "revision_date")
    silver = silver.withColumnRenamed("ANNDATS", "announce_date")
    silver = silver.withColumnRenamed("FPEDATS", "fiscal_period_end_date")
    silver = silver.withColumnRenamed("ACTTIMS", "actual_time")
    silver = silver.withColumnRenamed("REVTIMS", "revision_time")
    silver = silver.withColumnRenamed("ANNTIMS", "announce_time")

    stats = {
        "rows_in": int(rows_in),
        "eps_rows": int(eps_rows),
        "us_firm_rows": int(us_firm_rows),
        "current_rows_assuming_blank_is_current": int(current_rows),
        "usd_rows_assuming_blank_is_usd": int(usd_rows),
        "rows_with_event_date": int(rows_with_event_date),
        "rows_with_numeric_value": int(rows_with_numeric_value),
        "rows_after_filters": int(after_dedup),
        "duplicate_rows_removed": int(before_dedup - after_dedup),
    }
    return silver, stats


def build_gold(silver: DataFrame, neutral_abs_delta: float = 0.005) -> tuple[DataFrame, dict]:
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
        silver.groupBy(*group_cols)
        .agg(
            F.count("*").alias("estimate_count"),
            F.countDistinct("analyst_id").alias("analyst_count"),
            F.countDistinct("estimator_id").alias("estimator_count"),
            F.avg("estimate_value").alias("consensus_mean"),
            F.expr("percentile_approx(estimate_value, 0.5)").alias("consensus_median"),
            F.stddev_samp("estimate_value").alias("consensus_std"),
            F.min("estimate_value").alias("consensus_min"),
            F.max("estimate_value").alias("consensus_max"),
            F.min("actual_date").alias("actual_date"),
            F.min("announce_date").alias("announce_date"),
            F.min("revision_date").alias("revision_date"),
            F.first("event_time", ignorenulls=True).alias("event_time"),
        )
    )

    gold = gold.withColumn(
        "history_key",
        F.concat_ws(
            "|",
            F.coalesce(F.col("company_key"), F.lit("UNK")),
            F.coalesce(F.col("fpi_norm"), F.lit("UNK")),
            F.coalesce(F.col("fiscal_period_end_date").cast("string"), F.lit("UNK")),
        ),
    )

    order_window = Window.partitionBy("history_key").orderBy(
        F.col("event_date").asc(),
        F.coalesce(F.col("event_time"), F.lit("00:00:00")).asc(),
        F.coalesce(F.col("ticker_norm"), F.lit("")),
    )
    gold = (
        gold.withColumn("prior_consensus_median", F.lag("consensus_median").over(order_window))
        .withColumn("prior_consensus_mean", F.lag("consensus_mean").over(order_window))
        .withColumn("consensus_delta", F.col("consensus_median") - F.col("prior_consensus_median"))
        .withColumn(
            "consensus_pct_change",
            F.when(
                F.col("prior_consensus_median").isNull()
                | (F.abs(F.col("prior_consensus_median")) < F.lit(1e-9)),
                F.lit(None).cast("double"),
            ).otherwise(F.col("consensus_delta") / F.abs(F.col("prior_consensus_median"))),
        )
        .withColumn(
            "direction_label",
            F.when(F.col("consensus_delta").isNull(), F.lit("neutral"))
            .when(F.abs(F.col("consensus_delta")) < F.lit(neutral_abs_delta), F.lit("neutral"))
            .when(F.col("consensus_delta") > 0, F.lit("positive"))
            .otherwise(F.lit("negative")),
        )
        .withColumn(
            "magnitude_bucket",
            F.when(F.col("consensus_delta").isNull(), F.lit("unknown"))
            .when(F.abs(F.col("consensus_delta")) < F.lit(neutral_abs_delta), F.lit("flat"))
            .when(F.col("consensus_pct_change").isNotNull() & (F.abs(F.col("consensus_pct_change")) < 0.05), F.lit("small"))
            .when(F.col("consensus_pct_change").isNotNull() & (F.abs(F.col("consensus_pct_change")) < 0.15), F.lit("medium"))
            .when(F.col("consensus_pct_change").isNotNull(), F.lit("large"))
            .when(F.abs(F.col("consensus_delta")) < 0.05, F.lit("small"))
            .when(F.abs(F.col("consensus_delta")) < 0.25, F.lit("medium"))
            .otherwise(F.lit("large"))
        )
        .withColumn("event_type", F.lit("ibes_analyst_revision_consensus"))
        .withColumn(
            "event_id",
            F.concat_ws(
                "|",
                F.coalesce(F.col("company_key"), F.lit("UNK")),
                F.coalesce(F.col("event_date").cast("string"), F.lit("UNK")),
                F.coalesce(F.col("fpi_norm"), F.lit("UNK")),
                F.coalesce(F.col("fiscal_period_end_date").cast("string"), F.lit("UNK")),
            ),
        )
        .drop("history_key")
    )

    event_rows_out = gold.count()
    rows_with_prior_consensus = gold.filter(F.col("prior_consensus_median").isNotNull()).count()
    positive_labels = gold.filter(F.col("direction_label") == "positive").count()
    negative_labels = gold.filter(F.col("direction_label") == "negative").count()
    neutral_labels = gold.filter(F.col("direction_label") == "neutral").count()

    stats = {
        "rows_in": int(silver.count()),
        "event_rows_out": int(event_rows_out),
        "rows_with_prior_consensus": int(rows_with_prior_consensus),
        "positive_labels": int(positive_labels),
        "negative_labels": int(negative_labels),
        "neutral_labels": int(neutral_labels),
    }
    return gold, stats


def _value_or_none(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
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


def sample_gold_splits(gold: DataFrame, split: SplitSpec, seed: int) -> dict[str, pd.DataFrame]:
    total_rows = gold.count()
    if total_rows < split.total:
        raise ValueError(
            f"Need at least {split.total} gold events for {split.name}, but only found {total_rows}."
        )
    sample_pdf = gold.orderBy(F.rand(seed)).limit(split.total).toPandas().reset_index(drop=True)
    return {
        "train": sample_pdf.iloc[: split.train].copy(),
        "eval": sample_pdf.iloc[split.train : split.train + split.eval].copy(),
        "holdout": sample_pdf.iloc[split.train + split.eval :].copy(),
    }


def key_missingness(df: DataFrame, columns: Iterable[str]) -> dict[str, float]:
    row_count = df.count()
    if row_count == 0:
        return {column: 0.0 for column in columns if column in df.columns}
    metrics = df.agg(
        *[
            (F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)) / F.lit(row_count)).alias(column)
            for column in columns
            if column in df.columns
        ]
    ).collect()[0].asDict()
    return {column: round(float(value or 0.0), 6) for column, value in metrics.items()}


def dataset_date_range(df: DataFrame, column: str) -> dict[str, str | None]:
    if column not in df.columns:
        return {"min": None, "max": None}
    row = df.agg(F.min(column).alias("min"), F.max(column).alias("max")).collect()[0]
    return {
        "min": row["min"].isoformat() if row["min"] is not None else None,
        "max": row["max"].isoformat() if row["max"] is not None else None,
    }
