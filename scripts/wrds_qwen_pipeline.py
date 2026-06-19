#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import yaml
from pyspark.sql import DataFrame, SparkSession, Window, functions as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ibes_pipeline import build_bronze, build_gold, build_silver, load_raw_ibes_spark


DEFAULT_REQUIRED_OUTPUT_FIELDS = [
    "event_type",
    "direction",
    "confidence",
    "key_driver",
    "investment_signal",
    "reasoning_summary",
]


@dataclass
class PipelineContext:
    config: dict[str, Any]
    repo_root: Path
    processed_root: Path
    logs_root: Path
    docs_summary_path: Path
    logger: logging.Logger


def repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def stage_order() -> list[str]:
    return ["bronze", "silver", "gold", "jsonl", "validate", "smoke", "train"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the end-to-end WRDS -> Qwen data and training pipeline.")
    parser.add_argument("--config", default="configs/data_pipeline.yaml", help="Pipeline YAML config.")
    parser.add_argument("--start", default="bronze", choices=stage_order(), help="First stage to execute.")
    parser.add_argument("--through", default="train", choices=stage_order(), help="Last stage to execute.")
    parser.add_argument("--force", action="store_true", help="Recompute stage outputs even if they already exist.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_logger(logs_root: Path) -> logging.Logger:
    logs_root.mkdir(parents=True, exist_ok=True)
    log_path = logs_root / f"pipeline_{timestamp()}.log"
    logger = logging.getLogger(f"wrds_qwen_pipeline_{int(time.time())}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.info("pipeline log: %s", repo_rel(log_path))
    return logger


def create_context(config: dict[str, Any]) -> PipelineContext:
    processed_root = resolve_path(config["data"]["processed_root"])
    logs_root = resolve_path(config["data"]["logs_root"])
    docs_summary_path = resolve_path(config["data"]["docs_summary_path"])
    logger = ensure_logger(logs_root)
    return PipelineContext(
        config=config,
        repo_root=REPO_ROOT,
        processed_root=processed_root,
        logs_root=logs_root,
        docs_summary_path=docs_summary_path,
        logger=logger,
    )


def create_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )


def processed_dirs(ctx: PipelineContext) -> dict[str, Path]:
    root = ctx.processed_root
    return {
        "root": root,
        "raw_extracts": root / "raw_extracts",
        "bronze": root / "bronze",
        "silver": root / "silver",
        "gold": root / "gold",
        "jsonl": root / "jsonl",
        "reports": root / "reports",
        "state": root / "state",
    }


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def ensure_parent_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def stage_enabled(start: str, through: str, stage: str) -> bool:
    order = stage_order()
    return order.index(start) <= order.index(stage) <= order.index(through)


def raw_paths(ctx: PipelineContext) -> dict[str, Path]:
    data_cfg = ctx.config["data"]
    return {
        "ibes_raw_csv": resolve_path(data_cfg["raw_ibes_csv"]),
        "crsp_presence_csv": resolve_path(data_cfg["raw_crsp_presence_csv"]),
        "crsp_daily_zip": resolve_path(data_cfg["raw_crsp_daily_zip"]),
        "crsp_stock_header_csv": resolve_path(data_cfg["raw_crsp_stock_header_csv"]),
        "market_benchmark_csv": resolve_path(data_cfg["raw_market_benchmark_csv"]),
    }


def extracted_crsp_daily_csv(ctx: PipelineContext) -> Path:
    dirs = processed_dirs(ctx)
    raw_zip = raw_paths(ctx)["crsp_daily_zip"]
    stem = raw_zip.stem
    return dirs["raw_extracts"] / f"{stem}.csv"


def ensure_extracted_crsp_daily(ctx: PipelineContext, force: bool) -> Path:
    source_zip = raw_paths(ctx)["crsp_daily_zip"]
    target_csv = extracted_crsp_daily_csv(ctx)
    ensure_dirs([target_csv.parent])
    if target_csv.exists() and not force:
        ctx.logger.info("reusing extracted CRSP daily csv: %s", repo_rel(target_csv))
        return target_csv
    if target_csv.exists():
        target_csv.unlink()
    with zipfile.ZipFile(source_zip) as archive:
        member = next(name for name in archive.namelist() if not name.endswith("/"))
        ctx.logger.info("extracting %s from %s", member, repo_rel(source_zip))
        with archive.open(member) as src, target_csv.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return target_csv


def detect_dataset_type(name: str) -> str:
    mapping = {
        "ibes_raw_csv": "wrds_ibes_analyst_estimates",
        "crsp_presence_csv": "crsp_presence_dates",
        "crsp_daily_zip": "crsp_daily_returns_with_identifiers",
        "crsp_stock_header_csv": "crsp_stock_header_history",
        "market_benchmark_csv": "crsp_market_benchmark_returns",
    }
    return mapping.get(name, "unknown")


def manifest_date_columns(name: str) -> list[str]:
    mapping = {
        "ibes_raw_csv": ["ACTDATS", "REVDATS", "ANNDATS", "FPEDATS"],
        "crsp_presence_csv": ["date"],
        "crsp_daily_zip": ["date"],
        "crsp_stock_header_csv": ["BEGDAT", "ENDDAT"],
        "market_benchmark_csv": ["DATE"],
    }
    return mapping.get(name, [])


def build_manifest_entry(
    spark: SparkSession,
    dataset_name: str,
    source_path: Path,
    read_path: Path,
) -> tuple[dict[str, Any], DataFrame]:
    df = spark.read.option("header", True).option("mode", "DROPMALFORMED").csv(str(read_path))
    row_count = df.count()
    columns = df.columns
    null_agg = [
        (F.sum(F.when(F.col(column).isNull() | (F.trim(F.col(column)) == ""), 1).otherwise(0)) / F.lit(row_count)).alias(column)
        for column in columns
    ]
    null_row = df.agg(*null_agg).collect()[0].asDict() if columns else {}

    date_ranges: dict[str, dict[str, str | None]] = {}
    for date_col in manifest_date_columns(dataset_name):
        if date_col in columns:
            parsed = F.to_date(F.col(date_col))
            row = df.select(parsed.alias(date_col)).agg(F.min(date_col).alias("min"), F.max(date_col).alias("max")).collect()[0]
            date_ranges[date_col] = {
                "min": row["min"].isoformat() if row["min"] is not None else None,
                "max": row["max"].isoformat() if row["max"] is not None else None,
            }

    entry = {
        "dataset_name": dataset_name,
        "source_path": str(source_path),
        "read_path": str(read_path),
        "size_bytes": source_path.stat().st_size,
        "column_count": len(columns),
        "columns": columns,
        "row_count": int(row_count),
        "date_ranges": date_ranges,
        "null_rates": {column: round(float(value or 0.0), 6) for column, value in null_row.items()},
        "inferred_dataset_type": detect_dataset_type(dataset_name),
    }
    return entry, df


def build_crsp_presence_bronze(df: DataFrame) -> DataFrame:
    return df.select(
        F.col("PERMNO").cast("int").alias("permno"),
        F.to_date("date").alias("trade_date"),
    )


def build_crsp_daily_bronze(df: DataFrame) -> DataFrame:
    return df.select(
        F.col("PERMNO").cast("int").alias("permno"),
        F.to_date("date").alias("trade_date"),
        F.upper(F.substring(F.col("CUSIP"), 1, 8)).alias("cusip8"),
        F.upper(F.substring(F.col("NCUSIP"), 1, 8)).alias("ncusip8"),
        F.upper(F.trim(F.col("TICKER"))).alias("ticker_norm"),
        F.upper(F.trim(F.col("COMNAM"))).alias("company_name_norm"),
        F.abs(F.col("PRC").cast("double")).alias("price"),
        F.col("VOL").cast("double").alias("volume"),
        F.col("RET").cast("double").alias("ret"),
        F.col("SHROUT").cast("double").alias("shrout"),
    )


def build_stock_header_bronze(df: DataFrame) -> DataFrame:
    return df.select(
        F.col("PERMNO").cast("int").alias("permno"),
        F.col("PERMCO").cast("int").alias("permco"),
        F.upper(F.coalesce(F.trim(F.col("HTSYMBOL")), F.trim(F.col("HTICK")))).alias("header_ticker_norm"),
        F.upper(F.trim(F.col("HCOMNAM"))).alias("header_name_norm"),
        F.upper(F.substring(F.col("CUSIP"), 1, 8)).alias("cusip8"),
        F.to_date("BEGDAT").alias("beg_date"),
        F.to_date("ENDDAT").alias("end_date"),
        F.col("HNAICS").alias("hnaics"),
        F.col("HSICCD").alias("hsiccd"),
        F.col("HPRIMEXC").alias("hprimexc"),
    )


def build_benchmark_bronze(df: DataFrame, benchmark_column: str) -> DataFrame:
    return df.select(
        F.to_date("DATE").alias("trade_date"),
        F.col(benchmark_column).cast("double").alias("benchmark_return"),
    )


def dedup_stats(before: int, after: int) -> dict[str, int]:
    return {
        "rows_before_dedup": int(before),
        "rows_after_dedup": int(after),
        "exact_duplicates_removed": int(before - after),
    }


def bronze_paths(ctx: PipelineContext) -> dict[str, Path]:
    dirs = processed_dirs(ctx)
    return {
        "manifest": dirs["reports"] / "bronze_manifest.json",
        "ibes": dirs["bronze"] / "ibes_bronze.parquet",
        "crsp_presence": dirs["bronze"] / "crsp_presence_bronze.parquet",
        "crsp_daily": dirs["bronze"] / "crsp_daily_bronze.parquet",
        "stock_header": dirs["bronze"] / "crsp_stock_header_bronze.parquet",
        "benchmark": dirs["bronze"] / "market_benchmark_bronze.parquet",
        "report": dirs["reports"] / "bronze_report.json",
    }


def silver_paths(ctx: PipelineContext) -> dict[str, Path]:
    dirs = processed_dirs(ctx)
    return {
        "ibes": dirs["silver"] / "ibes_eps_us_current.parquet",
        "crsp_presence": dirs["silver"] / "crsp_presence_silver.parquet",
        "crsp_daily": dirs["silver"] / "crsp_daily_returns_silver.parquet",
        "stock_header": dirs["silver"] / "crsp_stock_header_silver.parquet",
        "benchmark": dirs["silver"] / "market_benchmark_silver.parquet",
        "report": dirs["reports"] / "silver_report.json",
    }


def gold_paths(ctx: PipelineContext) -> dict[str, Path]:
    dirs = processed_dirs(ctx)
    return {
        "ibes_gold": dirs["gold"] / "ibes_revision_events.parquet",
        "event_panel": dirs["gold"] / "event_panel.parquet",
        "event_windows": dirs["gold"] / "event_windows.parquet",
        "training_gold": dirs["gold"] / "wrds_signal_gold.parquet",
        "report": dirs["reports"] / "gold_report.json",
    }


def jsonl_paths(ctx: PipelineContext) -> dict[str, Path]:
    dirs = processed_dirs(ctx)
    return {
        "train": dirs["jsonl"] / "train.jsonl",
        "train_eval": dirs["jsonl"] / "train_eval.jsonl",
        "val": dirs["jsonl"] / "val.jsonl",
        "test": dirs["jsonl"] / "test.jsonl",
        "sample_100": dirs["jsonl"] / "sample_100.jsonl",
        "report": dirs["reports"] / "jsonl_report.json",
    }


def validation_paths(ctx: PipelineContext) -> dict[str, Path]:
    dirs = processed_dirs(ctx)
    return {
        "json": dirs["reports"] / "validation_report.json",
        "md": dirs["reports"] / "validation_report.md",
    }


def run_bronze_stage(ctx: PipelineContext, spark: SparkSession, force: bool) -> dict[str, Any]:
    paths = bronze_paths(ctx)
    ensure_parent_dirs(paths.values())
    if paths["report"].exists() and not force:
        ctx.logger.info("reusing bronze report: %s", repo_rel(paths["report"]))
        return json.loads(paths["report"].read_text())

    crsp_daily_csv = ensure_extracted_crsp_daily(ctx, force=force)
    raw = raw_paths(ctx)
    manifest: list[dict[str, Any]] = []

    ibes_entry, ibes_raw_df = build_manifest_entry(spark, "ibes_raw_csv", raw["ibes_raw_csv"], raw["ibes_raw_csv"])
    manifest.append(ibes_entry)
    ibes_bronze = build_bronze(load_raw_ibes_spark(spark, raw["ibes_raw_csv"]))
    ibes_bronze.write.mode("overwrite").parquet(str(paths["ibes"]))

    presence_entry, presence_raw_df = build_manifest_entry(spark, "crsp_presence_csv", raw["crsp_presence_csv"], raw["crsp_presence_csv"])
    manifest.append(presence_entry)
    crsp_presence_bronze = build_crsp_presence_bronze(presence_raw_df)
    crsp_presence_bronze.write.mode("overwrite").parquet(str(paths["crsp_presence"]))

    daily_entry, daily_raw_df = build_manifest_entry(spark, "crsp_daily_zip", raw["crsp_daily_zip"], crsp_daily_csv)
    manifest.append(daily_entry)
    crsp_daily_bronze = build_crsp_daily_bronze(daily_raw_df)
    crsp_daily_bronze.write.mode("overwrite").parquet(str(paths["crsp_daily"]))

    stock_entry, stock_raw_df = build_manifest_entry(spark, "crsp_stock_header_csv", raw["crsp_stock_header_csv"], raw["crsp_stock_header_csv"])
    manifest.append(stock_entry)
    stock_bronze = build_stock_header_bronze(stock_raw_df)
    stock_bronze.write.mode("overwrite").parquet(str(paths["stock_header"]))

    benchmark_column = ctx.config["pipeline"]["benchmark_column"]
    bench_entry, bench_raw_df = build_manifest_entry(spark, "market_benchmark_csv", raw["market_benchmark_csv"], raw["market_benchmark_csv"])
    manifest.append(bench_entry)
    benchmark_bronze = build_benchmark_bronze(bench_raw_df, benchmark_column)
    benchmark_bronze.write.mode("overwrite").parquet(str(paths["benchmark"]))

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "manifest_entries": manifest,
        "artifacts": {key: str(value) for key, value in paths.items() if key != "report"},
    }
    write_json(paths["manifest"], {"datasets": manifest})
    write_json(paths["report"], report)
    return report


def run_silver_stage(ctx: PipelineContext, spark: SparkSession, force: bool) -> dict[str, Any]:
    paths = silver_paths(ctx)
    ensure_parent_dirs(paths.values())
    if paths["report"].exists() and not force:
        ctx.logger.info("reusing silver report: %s", repo_rel(paths["report"]))
        return json.loads(paths["report"].read_text())

    bronze = bronze_paths(ctx)
    ibes_bronze = spark.read.parquet(str(bronze["ibes"]))
    silver_ibes, silver_ibes_stats = build_silver(ibes_bronze)
    silver_ibes.write.mode("overwrite").parquet(str(paths["ibes"]))

    presence_bronze = spark.read.parquet(str(bronze["crsp_presence"]))
    before = presence_bronze.count()
    presence_silver = presence_bronze.filter(F.col("permno").isNotNull() & F.col("trade_date").isNotNull())
    presence_silver.write.mode("overwrite").parquet(str(paths["crsp_presence"]))

    daily_bronze = spark.read.parquet(str(bronze["crsp_daily"]))
    before_daily = daily_bronze.count()
    daily_silver = (
        daily_bronze.filter(
            F.col("permno").isNotNull()
            & F.col("trade_date").isNotNull()
            & F.col("ret").isNotNull()
        )
        .dropDuplicates(["permno", "trade_date", "cusip8", "ncusip8", "ticker_norm"])
    )
    daily_window = Window.partitionBy("permno").orderBy(F.col("trade_date").asc())
    daily_silver = daily_silver.withColumn("trade_idx", F.row_number().over(daily_window))
    after_daily = daily_silver.count()
    daily_silver.write.mode("overwrite").parquet(str(paths["crsp_daily"]))

    stock_bronze = spark.read.parquet(str(bronze["stock_header"]))
    before_stock = stock_bronze.count()
    stock_silver = stock_bronze.filter(F.col("permno").isNotNull() & F.col("beg_date").isNotNull()).dropDuplicates(
        ["permno", "cusip8", "header_ticker_norm", "beg_date", "end_date"]
    )
    after_stock = stock_silver.count()
    stock_silver.write.mode("overwrite").parquet(str(paths["stock_header"]))

    benchmark_bronze = spark.read.parquet(str(bronze["benchmark"]))
    before_bench = benchmark_bronze.count()
    benchmark_silver = benchmark_bronze.filter(F.col("trade_date").isNotNull() & F.col("benchmark_return").isNotNull()).dropDuplicates(["trade_date"])
    after_bench = benchmark_silver.count()
    benchmark_silver.write.mode("overwrite").parquet(str(paths["benchmark"]))

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "silver_stats": {
            "ibes": silver_ibes_stats,
            "crsp_presence": {
                "rows_before_filter": int(before),
                "rows_after_filter": None,
                "exact_dedup_skipped": True,
                "note": "The 83M-row CRSP presence dataset is auxiliary for this run, so Silver keeps it as a filtered linear pass to avoid an unnecessary wide dedup shuffle.",
            },
            "crsp_daily": dedup_stats(before_daily, after_daily),
            "crsp_stock_header": dedup_stats(before_stock, after_stock),
            "market_benchmark": dedup_stats(before_bench, after_bench),
        },
        "artifacts": {key: str(value) for key, value in paths.items() if key != "report"},
    }
    write_json(paths["report"], report)
    return report


def union_candidate_matches(dfs: list[DataFrame]) -> DataFrame:
    current = dfs[0]
    for other in dfs[1:]:
        current = current.unionByName(other)
    return current


def enrich_event_panel(ibes_gold: DataFrame, crsp_daily: DataFrame, stock_header: DataFrame) -> tuple[DataFrame, dict[str, Any]]:
    base = ibes_gold.filter(F.col("prior_consensus_median").isNotNull()).cache()

    cusip_daily = (
        base.alias("e")
        .join(
            crsp_daily.alias("d"),
            (F.col("e.cusip8") == F.col("d.cusip8"))
            & (F.col("d.trade_date") >= F.col("e.event_date"))
            & (F.col("d.trade_date") <= F.date_add(F.col("e.event_date"), 7)),
            "inner",
        )
        .select(
            "e.*",
            F.col("d.permno").alias("matched_permno"),
            F.col("d.trade_date").alias("anchor_trade_date"),
            F.col("d.trade_idx").alias("anchor_trade_idx"),
            F.lit("cusip8_daily").alias("join_method"),
            F.lit("cusip8").alias("join_key_type"),
            F.lit(0.95).alias("join_confidence"),
        )
    )

    ncusip_daily = (
        base.alias("e")
        .join(
            crsp_daily.alias("d"),
            (F.col("e.cusip8") == F.col("d.ncusip8"))
            & (F.col("d.trade_date") >= F.col("e.event_date"))
            & (F.col("d.trade_date") <= F.date_add(F.col("e.event_date"), 7)),
            "inner",
        )
        .select(
            "e.*",
            F.col("d.permno").alias("matched_permno"),
            F.col("d.trade_date").alias("anchor_trade_date"),
            F.col("d.trade_idx").alias("anchor_trade_idx"),
            F.lit("ncusip8_daily").alias("join_method"),
            F.lit("ncusip8").alias("join_key_type"),
            F.lit(0.90).alias("join_confidence"),
        )
    )

    ticker_daily = (
        base.alias("e")
        .join(
            crsp_daily.alias("d"),
            (F.col("e.oftic_norm") == F.col("d.ticker_norm"))
            & (F.col("d.trade_date") >= F.col("e.event_date"))
            & (F.col("d.trade_date") <= F.date_add(F.col("e.event_date"), 7)),
            "inner",
        )
        .select(
            "e.*",
            F.col("d.permno").alias("matched_permno"),
            F.col("d.trade_date").alias("anchor_trade_date"),
            F.col("d.trade_idx").alias("anchor_trade_idx"),
            F.lit("ticker_daily").alias("join_method"),
            F.lit("ticker").alias("join_key_type"),
            F.lit(0.75).alias("join_confidence"),
        )
    )

    header_cusip = (
        base.alias("e")
        .join(
            stock_header.alias("h"),
            (F.col("e.cusip8") == F.col("h.cusip8"))
            & (F.col("e.event_date") >= F.col("h.beg_date"))
            & (F.col("h.end_date").isNull() | (F.col("e.event_date") <= F.col("h.end_date"))),
            "inner",
        )
        .join(
            crsp_daily.alias("d"),
            (F.col("h.permno") == F.col("d.permno"))
            & (F.col("d.trade_date") >= F.col("e.event_date"))
            & (F.col("d.trade_date") <= F.date_add(F.col("e.event_date"), 7)),
            "inner",
        )
        .select(
            "e.*",
            F.col("d.permno").alias("matched_permno"),
            F.col("d.trade_date").alias("anchor_trade_date"),
            F.col("d.trade_idx").alias("anchor_trade_idx"),
            F.lit("cusip8_stock_header").alias("join_method"),
            F.lit("cusip8").alias("join_key_type"),
            F.lit(0.70).alias("join_confidence"),
        )
    )

    header_ticker = (
        base.alias("e")
        .join(
            stock_header.alias("h"),
            (F.col("e.oftic_norm") == F.col("h.header_ticker_norm"))
            & (F.col("e.event_date") >= F.col("h.beg_date"))
            & (F.col("h.end_date").isNull() | (F.col("e.event_date") <= F.col("h.end_date"))),
            "inner",
        )
        .join(
            crsp_daily.alias("d"),
            (F.col("h.permno") == F.col("d.permno"))
            & (F.col("d.trade_date") >= F.col("e.event_date"))
            & (F.col("d.trade_date") <= F.date_add(F.col("e.event_date"), 7)),
            "inner",
        )
        .select(
            "e.*",
            F.col("d.permno").alias("matched_permno"),
            F.col("d.trade_date").alias("anchor_trade_date"),
            F.col("d.trade_idx").alias("anchor_trade_idx"),
            F.lit("ticker_stock_header").alias("join_method"),
            F.lit("ticker").alias("join_key_type"),
            F.lit(0.60).alias("join_confidence"),
        )
    )

    candidate_matches = union_candidate_matches([cusip_daily, ncusip_daily, ticker_daily, header_cusip, header_ticker])
    pick_window = Window.partitionBy("event_id").orderBy(
        F.col("join_confidence").desc(),
        F.col("anchor_trade_date").asc(),
        F.col("matched_permno").asc(),
    )
    matched = (
        candidate_matches.withColumn("join_rank", F.row_number().over(pick_window))
        .filter(F.col("join_rank") == 1)
        .drop("join_rank")
        .withColumn("join_status", F.lit("matched"))
        .withColumnRenamed("matched_permno", "permno")
    )

    unmatched = (
        base.alias("e")
        .join(matched.select("event_id").alias("m"), F.col("e.event_id") == F.col("m.event_id"), "left_anti")
        .withColumn("permno", F.lit(None).cast("int"))
        .withColumn("anchor_trade_date", F.lit(None).cast("date"))
        .withColumn("anchor_trade_idx", F.lit(None).cast("int"))
        .withColumn("join_method", F.lit("unmatched"))
        .withColumn("join_key_type", F.lit("unmatched"))
        .withColumn("join_confidence", F.lit(0.0))
        .withColumn("join_status", F.lit("unmatched"))
    )

    event_panel = matched.unionByName(unmatched)
    report = {
        "event_rows_in": int(base.count()),
        "matched_rows": int(matched.count()),
        "unmatched_rows": int(unmatched.count()),
        "join_method_distribution": {
            row["join_method"]: int(row["count"])
            for row in event_panel.groupBy("join_method").count().collect()
        },
    }
    return event_panel, report


def signed_token(value: int) -> str:
    return f"m{abs(value)}" if value < 0 else f"p{value}"


def window_column(slug_prefix: str, start: int, end: int) -> str:
    return f"{slug_prefix}_w_{signed_token(start)}_{signed_token(end)}"


def cumulative_return_expr(start: int, end: int, value_column: str) -> F.Column:
    valid = (
        (F.col("relative_offset") >= F.lit(start))
        & (F.col("relative_offset") <= F.lit(end))
        & F.col(value_column).isNotNull()
        & (F.col(value_column) > F.lit(-0.999999))
    )
    count_expr = F.sum(F.when(valid, 1).otherwise(0))
    log_sum = F.sum(F.when(valid, F.log1p(F.col(value_column))).otherwise(F.lit(0.0)))
    return F.when(count_expr > 0, F.exp(log_sum) - F.lit(1.0)).otherwise(F.lit(None).cast("double"))


def observation_count_expr(start: int, end: int, value_column: str) -> F.Column:
    valid = (
        (F.col("relative_offset") >= F.lit(start))
        & (F.col("relative_offset") <= F.lit(end))
        & F.col(value_column).isNotNull()
    )
    return F.sum(F.when(valid, 1).otherwise(0))


def add_event_windows(
    event_panel: DataFrame,
    crsp_daily: DataFrame,
    benchmark: DataFrame,
    window_specs: list[str],
) -> tuple[DataFrame, dict[str, Any]]:
    matched = event_panel.filter(F.col("join_status") == "matched").cache()
    event_columns = matched.columns
    joined = (
        matched.alias("e")
        .join(
            crsp_daily.alias("d"),
            (F.col("e.permno") == F.col("d.permno"))
            & (F.col("d.trade_idx") >= F.col("e.anchor_trade_idx") - F.lit(1))
            & (F.col("d.trade_idx") <= F.col("e.anchor_trade_idx") + F.lit(5)),
            "left",
        )
        .join(benchmark.alias("b"), F.col("d.trade_date") == F.col("b.trade_date"), "left")
        .withColumn("relative_offset", F.col("d.trade_idx") - F.col("e.anchor_trade_idx"))
        .select(
            *[F.col(f"e.{column}").alias(column) for column in event_columns],
            F.col("d.ret").alias("ret"),
            F.col("b.benchmark_return").alias("benchmark_return"),
            F.col("relative_offset"),
        )
    )

    group_cols = event_columns
    agg_exprs = []
    for spec in window_specs:
        start_text, end_text = spec.split(":")
        start = int(start_text)
        end = int(end_text)
        raw_col = window_column("raw_return", start, end)
        bench_col = window_column("benchmark_return", start, end)
        abnormal_col = window_column("abnormal_return", start, end)
        count_col = window_column("obs_count", start, end)
        agg_exprs.extend(
            [
                cumulative_return_expr(start, end, "ret").alias(raw_col),
                cumulative_return_expr(start, end, "benchmark_return").alias(bench_col),
                observation_count_expr(start, end, "ret").alias(count_col),
            ]
        )

    aggregated = joined.groupBy(*group_cols).agg(*agg_exprs)
    for spec in window_specs:
        start_text, end_text = spec.split(":")
        start = int(start_text)
        end = int(end_text)
        raw_col = window_column("raw_return", start, end)
        bench_col = window_column("benchmark_return", start, end)
        abnormal_col = window_column("abnormal_return", start, end)
        aggregated = aggregated.withColumn(
            abnormal_col,
            F.when(F.col(raw_col).isNotNull() & F.col(bench_col).isNotNull(), F.col(raw_col) - F.col(bench_col)).otherwise(F.lit(None).cast("double")),
        )

    unmatched = event_panel.filter(F.col("join_status") != "matched")
    for spec in window_specs:
        start_text, end_text = spec.split(":")
        start = int(start_text)
        end = int(end_text)
        for prefix in ("raw_return", "benchmark_return", "abnormal_return"):
            aggregated = aggregated
            unmatched = unmatched.withColumn(window_column(prefix, start, end), F.lit(None).cast("double"))
        unmatched = unmatched.withColumn(window_column("obs_count", start, end), F.lit(0))

    event_windows = aggregated.unionByName(unmatched, allowMissingColumns=True)
    report = {
        "matched_rows_with_windows": int(aggregated.count()),
        "unmatched_rows_without_windows": int(unmatched.count()),
    }
    return event_windows, report


def add_signal_labels(df: DataFrame, neutral_threshold: float, large_threshold: float) -> DataFrame:
    reaction_metric = F.coalesce(F.col("abnormal_return_w_p0_p1"), F.col("raw_return_w_p0_p1"))
    return (
        df.withColumn(
            "realized_direction_label",
            F.when(reaction_metric.isNull(), F.lit("unknown"))
            .when(F.abs(reaction_metric) < F.lit(neutral_threshold), F.lit("neutral"))
            .when(reaction_metric > 0, F.lit("positive"))
            .otherwise(F.lit("negative")),
        )
        .withColumn(
            "realized_magnitude_bucket",
            F.when(reaction_metric.isNull(), F.lit("unknown"))
            .when(F.abs(reaction_metric) < F.lit(neutral_threshold), F.lit("flat"))
            .when(F.abs(reaction_metric) < F.lit(large_threshold), F.lit("small"))
            .otherwise(F.lit("large")),
        )
        .withColumn(
            "alignment_label",
            F.when(F.col("realized_direction_label") == "unknown", F.lit("unobserved"))
            .when(F.col("direction_label") == F.col("realized_direction_label"), F.lit("aligned"))
            .when((F.col("direction_label") == "neutral") | (F.col("realized_direction_label") == "neutral"), F.lit("mixed"))
            .otherwise(F.lit("contrarian")),
        )
        .withColumn(
            "event_type_signal",
            F.when(F.col("join_status") == "matched", F.lit("analyst_revision_market_reaction"))
            .otherwise(F.lit("analyst_revision_unmatched")),
        )
    )


def run_gold_stage(ctx: PipelineContext, spark: SparkSession, force: bool) -> dict[str, Any]:
    paths = gold_paths(ctx)
    ensure_parent_dirs(paths.values())
    if paths["report"].exists() and not force:
        ctx.logger.info("reusing gold report: %s", repo_rel(paths["report"]))
        return json.loads(paths["report"].read_text())

    silver = silver_paths(ctx)
    silver_ibes = spark.read.parquet(str(silver["ibes"]))
    ibes_gold, ibes_gold_stats = build_gold(silver_ibes, neutral_abs_delta=ctx.config["pipeline"]["neutral_abs_delta"])
    ibes_gold.write.mode("overwrite").parquet(str(paths["ibes_gold"]))

    crsp_daily = spark.read.parquet(str(silver["crsp_daily"]))
    stock_header = spark.read.parquet(str(silver["stock_header"]))
    benchmark = spark.read.parquet(str(silver["benchmark"]))

    event_panel, panel_report = enrich_event_panel(ibes_gold, crsp_daily, stock_header)
    event_panel.write.mode("overwrite").parquet(str(paths["event_panel"]))

    event_windows, window_report = add_event_windows(event_panel, crsp_daily, benchmark, ctx.config["pipeline"]["windows"])
    training_gold = add_signal_labels(
        event_windows,
        neutral_threshold=float(ctx.config["pipeline"]["reaction_neutral_threshold"]),
        large_threshold=float(ctx.config["pipeline"]["reaction_large_threshold"]),
    )
    training_gold.write.mode("overwrite").parquet(str(paths["event_windows"]))
    training_gold.write.mode("overwrite").parquet(str(paths["training_gold"]))

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "ibes_gold_stats": ibes_gold_stats,
        "event_panel_stats": panel_report,
        "event_window_stats": window_report,
        "training_gold_rows": int(training_gold.count()),
        "artifacts": {key: str(value) for key, value in paths.items() if key != "report"},
    }
    write_json(paths["report"], report)
    return report


def confidence_score(row: dict[str, Any]) -> float:
    analyst_count = float(row.get("analyst_count") or 0)
    estimate_count = float(row.get("estimate_count") or 0)
    pct_change = abs(float(row.get("consensus_pct_change") or 0.0))
    join_confidence = float(row.get("join_confidence") or 0.0)
    raw = min(1.0, 0.25 + min(analyst_count / 10.0, 0.25) + min(estimate_count / 12.0, 0.2) + min(pct_change, 0.2) + min(join_confidence / 2.0, 0.1))
    return round(raw, 3)


def investment_signal(direction: str, realized_direction: str) -> str:
    if direction == "positive" and realized_direction in {"positive", "neutral", "unknown"}:
        return "long"
    if direction == "negative" and realized_direction in {"negative", "neutral", "unknown"}:
        return "short"
    if direction == "neutral":
        return "hold"
    return "watch"


def key_driver(row: dict[str, Any]) -> str:
    direction = row["direction_label"]
    reaction = row["realized_direction_label"]
    return f"{direction}_revision__{reaction}_reaction__{row['alignment_label']}"


def reasoning_summary(row: dict[str, Any]) -> str:
    parts = [
        f"Consensus revision was {row['direction_label']}",
        f"with {int(row['analyst_count'] or 0)} analysts",
        f"and estimated market reaction {row['realized_direction_label']}.",
    ]
    return " ".join(parts)


def build_chat_record(row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "task": "classify_event_driven_financial_signal",
        "company": {
            "company_key": row.get("company_key"),
            "ticker": row.get("ticker_norm"),
            "oftic": row.get("oftic_norm"),
            "cusip8": row.get("cusip8"),
            "company_name": row.get("company_name_norm"),
            "permno": row.get("permno"),
        },
        "event": {
            "event_id": row.get("event_id"),
            "event_type": row.get("event_type_signal"),
            "event_date": row.get("event_date"),
            "anchor_trade_date": row.get("anchor_trade_date"),
            "fiscal_period_end_date": row.get("fiscal_period_end_date"),
            "fpi": row.get("fpi_norm"),
            "join_status": row.get("join_status"),
            "join_method": row.get("join_method"),
            "join_confidence": row.get("join_confidence"),
        },
        "features": {
            "estimate_count": row.get("estimate_count"),
            "analyst_count": row.get("analyst_count"),
            "estimator_count": row.get("estimator_count"),
            "consensus_mean": row.get("consensus_mean"),
            "consensus_median": row.get("consensus_median"),
            "prior_consensus_median": row.get("prior_consensus_median"),
            "consensus_delta": row.get("consensus_delta"),
            "consensus_pct_change": row.get("consensus_pct_change"),
            "consensus_std": row.get("consensus_std"),
            "raw_return_w_p0_p1": row.get("raw_return_w_p0_p1"),
            "raw_return_w_p0_p3": row.get("raw_return_w_p0_p3"),
            "raw_return_w_p0_p5": row.get("raw_return_w_p0_p5"),
            "abnormal_return_w_p0_p1": row.get("abnormal_return_w_p0_p1"),
            "abnormal_return_w_p0_p3": row.get("abnormal_return_w_p0_p3"),
            "abnormal_return_w_p0_p5": row.get("abnormal_return_w_p0_p5"),
            "alignment_label": row.get("alignment_label"),
        },
    }
    strict_output = {
        "event_type": row.get("event_type_signal"),
        "direction": row.get("direction_label"),
        "confidence": confidence_score(row),
        "key_driver": key_driver(row),
        "investment_signal": investment_signal(row.get("direction_label"), row.get("realized_direction_label")),
        "reasoning_summary": reasoning_summary(row),
    }
    system = (
        "You are an event-driven financial signal assistant. "
        "Return strict valid JSON only with keys: "
        + ", ".join(DEFAULT_REQUIRED_OUTPUT_FIELDS)
        + "."
    )
    user = "Analyze this structured analyst revision event and produce a strict JSON trading signal.\n" + json.dumps(payload, sort_keys=True)
    assistant = json.dumps(strict_output, sort_keys=True)
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "instruction": "Analyze the event-driven financial signal and respond with strict JSON only.",
        "input": json.dumps(payload, sort_keys=True),
        "output": assistant,
        "event_id": row.get("event_id"),
        "source": "wrds_qwen_pipeline",
        "event_date": row.get("event_date"),
        "direction_label": row.get("direction_label"),
        "realized_direction_label": row.get("realized_direction_label"),
        "alignment_label": row.get("alignment_label"),
    }


def normalize_scalar(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return round(value, 6)
    return value


def select_splits(
    df: DataFrame,
    seed: int,
    max_train: int,
    max_val: int,
    max_test: int,
    train_split_fraction: float,
    val_split_fraction: float,
) -> dict[str, DataFrame]:
    ordered = df.filter(F.col("direction_label").isin("positive", "negative", "neutral")).orderBy(F.col("event_date").asc(), F.col("event_id").asc())
    ranked = ordered.withColumn("split_rank", F.percent_rank().over(Window.orderBy(F.col("event_date").asc(), F.col("event_id").asc())))
    train_cutoff = float(train_split_fraction)
    val_cutoff = float(train_split_fraction) + float(val_split_fraction)
    train = ranked.filter(F.col("split_rank") < train_cutoff)
    val = ranked.filter((F.col("split_rank") >= train_cutoff) & (F.col("split_rank") < val_cutoff))
    test = ranked.filter(F.col("split_rank") >= val_cutoff)
    if max_train:
        train = train.orderBy(F.rand(seed)).limit(max_train)
    if max_val:
        val = val.orderBy(F.rand(seed + 1)).limit(max_val)
    if max_test:
        test = test.orderBy(F.rand(seed + 2)).limit(max_test)
    return {"train": train, "val": val, "test": test}


def write_jsonl_from_df(df: DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in df.toLocalIterator():
            raw = {key: normalize_scalar(value) for key, value in row.asDict().items()}
            record = build_chat_record(raw)
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def run_jsonl_stage(ctx: PipelineContext, spark: SparkSession, force: bool) -> dict[str, Any]:
    paths = jsonl_paths(ctx)
    ensure_parent_dirs(paths.values())
    if paths["report"].exists() and not force:
        ctx.logger.info("reusing jsonl report: %s", repo_rel(paths["report"]))
        return json.loads(paths["report"].read_text())

    gold = spark.read.parquet(str(gold_paths(ctx)["training_gold"]))
    splits = select_splits(
        gold,
        seed=int(ctx.config["pipeline"]["seed"]),
        max_train=int(ctx.config["pipeline"]["max_train_examples"]),
        max_val=int(ctx.config["pipeline"]["max_val_examples"]),
        max_test=int(ctx.config["pipeline"]["max_test_examples"]),
        train_split_fraction=float(ctx.config["pipeline"]["train_split_fraction"]),
        val_split_fraction=float(ctx.config["pipeline"]["val_split_fraction"]),
    )
    counts = {
        "train": write_jsonl_from_df(splits["train"], paths["train"]),
        "val": write_jsonl_from_df(splits["val"], paths["val"]),
        "test": write_jsonl_from_df(splits["test"], paths["test"]),
    }
    train_eval_df = splits["val"].orderBy(F.rand(int(ctx.config["pipeline"]["seed"]) + 50)).limit(
        int(ctx.config["pipeline"]["max_train_eval_examples"])
    )
    counts["train_eval"] = write_jsonl_from_df(train_eval_df, paths["train_eval"])
    sample_df = splits["train"].orderBy(F.rand(int(ctx.config["pipeline"]["seed"]) + 100)).limit(int(ctx.config["pipeline"]["sample_100_examples"]))
    counts["sample_100"] = write_jsonl_from_df(sample_df, paths["sample_100"])

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "split_counts": counts,
        "split_roles": {
            "train": "adapter optimization split",
            "train_eval": "small deterministic subset of val used only for cheap training-time health checks",
            "val": "full validation pool retained locally but not used for repeated trainer evals",
            "test": "full final WRDS holdout for post-train base-vs-adapter reporting",
            "sample_100": "small smoke-training slice from train",
        },
        "artifacts": {key: str(value) for key, value in paths.items() if key != "report"},
    }
    write_json(paths["report"], report)
    return report


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            payload["_line_number"] = line_number
            rows.append(payload)
    return rows


def token_length(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def summarize_lengths(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {"min": 0, "p50": 0, "p95": 0, "max": 0, "mean": 0}
    values = sorted(lengths)
    def percentile(p: float) -> int:
        idx = min(len(values) - 1, int(round((len(values) - 1) * p)))
        return values[idx]
    return {
        "min": values[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": values[-1],
        "mean": round(sum(values) / len(values), 2),
    }


def run_validation_stage(ctx: PipelineContext, force: bool) -> dict[str, Any]:
    paths = validation_paths(ctx)
    ensure_parent_dirs(paths.values())
    if paths["json"].exists() and not force:
        ctx.logger.info("reusing validation report: %s", repo_rel(paths["json"]))
        return json.loads(paths["json"].read_text())

    jsonl_report = json.loads(jsonl_paths(ctx)["report"].read_text())

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        ctx.config["pipeline"]["model_id"],
        trust_remote_code=True,
        local_files_only=bool(ctx.config["pipeline"]["local_files_only"]),
    )

    split_files = {name: path for name, path in jsonl_paths(ctx).items() if name in {"train", "train_eval", "val", "test", "sample_100"}}
    split_rows = {name: load_jsonl(path) for name, path in split_files.items()}
    duplicate_events: dict[str, list[str]] = {}
    event_to_splits: dict[str, set[str]] = {}
    errors: list[str] = []
    label_distribution: dict[str, dict[str, int]] = {}
    token_stats: dict[str, dict[str, dict[str, float]]] = {}

    for split_name, rows in split_rows.items():
        label_distribution[split_name] = {}
        input_lengths: list[int] = []
        output_lengths: list[int] = []
        for row in rows:
            event_id = row.get("event_id")
            if event_id and split_name not in {"sample_100", "train_eval"}:
                event_to_splits.setdefault(event_id, set()).add(split_name)
            if not row.get("input"):
                errors.append(f"{split_name}:{row['_line_number']} empty input")
            if "messages" not in row or not row["messages"] or row["messages"][-1].get("role") != "assistant":
                errors.append(f"{split_name}:{row['_line_number']} missing assistant chat message")
                continue
            assistant_text = row["messages"][-1].get("content", "")
            if assistant_text != row.get("output", ""):
                errors.append(f"{split_name}:{row['_line_number']} output mismatch between chat and legacy field")
            try:
                parsed = json.loads(assistant_text)
            except json.JSONDecodeError as exc:
                errors.append(f"{split_name}:{row['_line_number']} invalid assistant json: {exc}")
                continue
            missing = [field for field in DEFAULT_REQUIRED_OUTPUT_FIELDS if field not in parsed]
            if missing:
                errors.append(f"{split_name}:{row['_line_number']} missing assistant fields: {', '.join(missing)}")
            label = parsed.get("direction", "unknown")
            label_distribution[split_name][label] = label_distribution[split_name].get(label, 0) + 1
            input_lengths.append(token_length(tokenizer, row["input"]))
            output_lengths.append(token_length(tokenizer, row["output"]))
        token_stats[split_name] = {
            "input_tokens": summarize_lengths(input_lengths),
            "output_tokens": summarize_lengths(output_lengths),
        }

    for event_id, splits in event_to_splits.items():
        if len(splits) > 1:
            duplicate_events[event_id] = sorted(splits)
    if duplicate_events:
        errors.append(f"duplicate leakage across train/val/test: {len(duplicate_events)} event_ids")

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "valid": not errors,
        "errors": errors,
        "split_counts": {name: len(rows) for name, rows in split_rows.items()},
        "split_roles": jsonl_report.get("split_roles", {}),
        "duplicate_event_ids": duplicate_events,
        "label_distribution": label_distribution,
        "token_stats": token_stats,
    }
    write_json(paths["json"], report)

    md_lines = [
        "# Validation Report",
        "",
        f"- valid: `{report['valid']}`",
        f"- train rows: `{report['split_counts']['train']}`",
        f"- train_eval rows: `{report['split_counts']['train_eval']}`",
        f"- val rows: `{report['split_counts']['val']}`",
        f"- test rows: `{report['split_counts']['test']}`",
        f"- sample_100 rows: `{report['split_counts']['sample_100']}`",
        "",
        "## Split Roles",
        "",
    ]
    md_lines.extend([f"- {name}: {description}" for name, description in report.get("split_roles", {}).items()])
    md_lines.extend(
        [
            "",
        "## Label Distribution",
        "",
        "```json",
        json.dumps(label_distribution, indent=2, sort_keys=True),
        "```",
        "",
        "## Token Stats",
        "",
        "```json",
        json.dumps(token_stats, indent=2, sort_keys=True),
        "```",
        "",
        "## Errors",
        "",
        ]
    )
    if errors:
        md_lines.extend([f"- {error}" for error in errors])
    else:
        md_lines.append("- none")
    write_markdown(paths["md"], md_lines)
    return report


def run_subprocess(command: list[str], log_path: Path, cwd: Path, background: bool = False) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if background:
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        return {"pid": process.pid, "command": command, "log_path": str(log_path)}
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return {"returncode": result.returncode, "command": command, "log_path": str(log_path)}


def smoke_report_path(ctx: PipelineContext) -> Path:
    return processed_dirs(ctx)["reports"] / "smoke_report.json"


def run_smoke_stage(ctx: PipelineContext, force: bool) -> dict[str, Any]:
    report_path = smoke_report_path(ctx)
    if report_path.exists() and not force:
        ctx.logger.info("reusing smoke report: %s", repo_rel(report_path))
        return json.loads(report_path.read_text())

    validate_report = json.loads(validation_paths(ctx)["json"].read_text())
    if not validate_report["valid"]:
        raise RuntimeError("validation failed; refusing to run smoke training")

    smoke_cfg = ctx.config["pipeline"]["smoke"]
    output_dir = resolve_path(smoke_cfg["output_dir"]) / timestamp()
    log_path = ctx.logs_root / f"smoke_{timestamp()}.log"
    command = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        "training/train_finance_lora.py",
        "--model-id",
        ctx.config["pipeline"]["model_id"],
        "--train-file",
        repo_rel(jsonl_paths(ctx)["sample_100"]),
        "--output-dir",
        repo_rel(output_dir),
        "--epochs",
        str(smoke_cfg["epochs"]),
        "--max-total-examples",
        str(smoke_cfg["max_total_examples"]),
        "--per-device-train-batch-size",
        str(smoke_cfg["per_device_train_batch_size"]),
        "--gradient-accumulation-steps",
        str(smoke_cfg["gradient_accumulation_steps"]),
        "--save-steps",
        str(smoke_cfg["save_steps"]),
        "--logging-steps",
        str(smoke_cfg["logging_steps"]),
        "--max-seq-length",
        str(smoke_cfg["max_seq_length"]),
        "--local-files-only",
    ]
    result = run_subprocess(command, log_path, REPO_ROOT, background=False)
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "succeeded": result["returncode"] == 0,
        "returncode": result["returncode"],
        "command": command,
        "log_path": str(log_path),
        "output_dir": str(output_dir),
    }
    write_json(report_path, payload)
    if result["returncode"] != 0:
        raise RuntimeError(f"smoke training failed; see {log_path}")
    return payload


def train_report_path(ctx: PipelineContext) -> Path:
    return processed_dirs(ctx)["reports"] / "train_launch_report.json"


def run_train_stage(ctx: PipelineContext, force: bool) -> dict[str, Any]:
    report_path = train_report_path(ctx)
    if report_path.exists() and not force:
        ctx.logger.info("reusing train launch report: %s", repo_rel(report_path))
        return json.loads(report_path.read_text())

    validate_report = json.loads(validation_paths(ctx)["json"].read_text())
    smoke_report = json.loads(smoke_report_path(ctx).read_text())
    if not validate_report["valid"]:
        raise RuntimeError("validation failed; refusing to launch overnight training")
    if not smoke_report["succeeded"]:
        raise RuntimeError("smoke training failed; refusing to launch overnight training")

    overnight = ctx.config["pipeline"]["overnight"]
    output_dir = resolve_path(overnight["output_dir_root"]) / timestamp()
    log_path = ctx.logs_root / f"overnight_train_{timestamp()}.log"
    command = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        "training/train_finance_lora.py",
        "--model-id",
        ctx.config["pipeline"]["model_id"],
        "--train-file",
        repo_rel(jsonl_paths(ctx)["train"]),
        "--eval-file",
        repo_rel(jsonl_paths(ctx)["train_eval"]),
        "--test-file",
        repo_rel(jsonl_paths(ctx)["test"]),
        "--output-dir",
        repo_rel(output_dir),
        "--epochs",
        str(overnight["epochs"]),
        "--lr",
        str(overnight["lr"]),
        "--per-device-train-batch-size",
        str(overnight["per_device_train_batch_size"]),
        "--gradient-accumulation-steps",
        str(overnight["gradient_accumulation_steps"]),
        "--eval-steps",
        str(overnight["eval_steps"]),
        "--save-steps",
        str(overnight["save_steps"]),
        "--logging-steps",
        str(overnight["logging_steps"]),
        "--max-seq-length",
        str(overnight["max_seq_length"]),
        "--max-total-examples",
        "0",
        "--local-files-only",
    ]
    result = run_subprocess(command, log_path, REPO_ROOT, background=True)
    time.sleep(2)
    alive = Path(f"/proc/{result['pid']}").exists()
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "launched": alive,
        "pid": result["pid"],
        "command": command,
        "log_path": str(log_path),
        "output_dir": str(output_dir),
    }
    write_json(report_path, payload)
    if not alive:
        raise RuntimeError(f"overnight training process exited immediately; see {log_path}")
    return payload


def write_summary(ctx: PipelineContext) -> None:
    bronze_report = json.loads(bronze_paths(ctx)["report"].read_text())
    silver_report = json.loads(silver_paths(ctx)["report"].read_text())
    gold_report = json.loads(gold_paths(ctx)["report"].read_text())
    jsonl_report = json.loads(jsonl_paths(ctx)["report"].read_text())
    validation_report = json.loads(validation_paths(ctx)["json"].read_text())
    smoke_report = json.loads(smoke_report_path(ctx).read_text()) if smoke_report_path(ctx).exists() else None
    train_report = json.loads(train_report_path(ctx).read_text()) if train_report_path(ctx).exists() else None

    lines = [
        "# Overnight Run Summary",
        "",
        "## Command",
        "",
        "```bash",
        "python scripts/run_pipeline.py --config configs/data_pipeline.yaml --start bronze --through train",
        "```",
        "",
        "## Raw Data Located",
        "",
    ]
    for entry in bronze_report["manifest_entries"]:
        lines.append(f"- `{entry['dataset_name']}`: `{entry['row_count']}` rows from `{entry['source_path']}`")

    lines.extend(
        [
            "",
            "## Silver And Gold Counts",
            "",
            f"- IBES silver rows: `{silver_report['silver_stats']['ibes']['rows_after_filters']}`",
            f"- IBES gold rows: `{gold_report['ibes_gold_stats']['event_rows_out']}`",
            f"- Event-panel matched rows: `{gold_report['event_panel_stats']['matched_rows']}`",
            f"- Event-panel unmatched rows: `{gold_report['event_panel_stats']['unmatched_rows']}`",
            f"- Training gold rows: `{gold_report['training_gold_rows']}`",
            "",
            "## JSONL Splits",
            "",
            f"- train: `{jsonl_report['split_counts']['train']}`",
            f"- train_eval: `{jsonl_report['split_counts']['train_eval']}`",
            f"- val: `{jsonl_report['split_counts']['val']}`",
            f"- test: `{jsonl_report['split_counts']['test']}`",
            f"- sample_100: `{jsonl_report['split_counts']['sample_100']}`",
            "",
            "## Split Contract",
            "",
            "- `train`: adapter optimization rows",
            "- `train_eval`: small deterministic subset of `val` used only for fast training-time health checks",
            "- `val`: full validation pool retained locally, not used for repeated trainer validation during long runs",
            "- `test`: full WRDS holdout reserved for final post-train base-vs-adapter measurement",
            "- public benchmarks: separate generalization measurement and never part of Trainer evaluation",
            "",
            "## Label Distribution",
            "",
            "```json",
            json.dumps(validation_report["label_distribution"], indent=2, sort_keys=True),
            "```",
            "",
            "## Token Stats",
            "",
            "```json",
            json.dumps(validation_report["token_stats"], indent=2, sort_keys=True),
            "```",
            "",
            "## Validation And Training Gates",
            "",
            f"- validation passed: `{validation_report['valid']}`",
            f"- smoke training succeeded: `{smoke_report['succeeded'] if smoke_report else False}`",
            f"- overnight training started: `{train_report['launched'] if train_report else False}`",
            "",
            "## Artifact Paths",
            "",
            f"- bronze manifest: `{repo_rel(bronze_paths(ctx)['manifest'])}`",
            f"- silver report: `{repo_rel(silver_paths(ctx)['report'])}`",
            f"- gold report: `{repo_rel(gold_paths(ctx)['report'])}`",
            f"- validation report: `{repo_rel(validation_paths(ctx)['json'])}`",
            f"- jsonl directory: `{repo_rel(processed_dirs(ctx)['jsonl'])}`",
            f"- smoke log: `{repo_rel(Path(smoke_report['log_path'])) if smoke_report else 'n/a'}`",
            f"- smoke output dir: `{repo_rel(Path(smoke_report['output_dir'])) if smoke_report else 'n/a'}`",
            f"- overnight log: `{repo_rel(Path(train_report['log_path'])) if train_report else 'n/a'}`",
            f"- overnight output dir: `{repo_rel(Path(train_report['output_dir'])) if train_report else 'n/a'}`",
        ]
    )
    write_markdown(ctx.docs_summary_path, lines)


def main() -> int:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    ctx = create_context(config)
    dirs = processed_dirs(ctx)
    ensure_dirs(dirs.values())

    spark: SparkSession | None = None
    try:
        spark = create_spark(config["pipeline"]["name"])
        if stage_enabled(args.start, args.through, "bronze"):
            ctx.logger.info("running stage: bronze")
            run_bronze_stage(ctx, spark, force=args.force)
        if stage_enabled(args.start, args.through, "silver"):
            ctx.logger.info("running stage: silver")
            run_silver_stage(ctx, spark, force=args.force)
        if stage_enabled(args.start, args.through, "gold"):
            ctx.logger.info("running stage: gold")
            run_gold_stage(ctx, spark, force=args.force)
        if stage_enabled(args.start, args.through, "jsonl"):
            ctx.logger.info("running stage: jsonl")
            run_jsonl_stage(ctx, spark, force=args.force)
    finally:
        if spark is not None:
            spark.stop()

    if stage_enabled(args.start, args.through, "validate"):
        ctx.logger.info("running stage: validate")
        run_validation_stage(ctx, force=args.force)
    if stage_enabled(args.start, args.through, "smoke"):
        ctx.logger.info("running stage: smoke")
        run_smoke_stage(ctx, force=args.force)
    if stage_enabled(args.start, args.through, "train"):
        ctx.logger.info("running stage: train")
        run_train_stage(ctx, force=args.force)

    write_summary(ctx)
    ctx.logger.info("summary written to %s", repo_rel(ctx.docs_summary_path))
    return 0
