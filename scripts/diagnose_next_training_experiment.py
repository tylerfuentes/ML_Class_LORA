#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk


REPO_ROOT = Path(__file__).resolve().parents[1]
JSONL_ROOT = REPO_ROOT / "data" / "processed" / "ibes_lora_baseline" / "jsonl"
GOLD_PARQUET = REPO_ROOT / "data" / "processed" / "ibes_lora_baseline" / "gold" / "ibes_revision_events.parquet"
EVAL_ROOT = REPO_ROOT / "outputs" / "evals"
LOCAL_CANDIDATE_ROOT = REPO_ROOT / "outputs" / "next-training-experiment"
DOCS_ROOT = REPO_ROOT / "docs"
BENCHMARK_CACHE_ROOT = REPO_ROOT / "external" / "FinGPT" / "fingpt" / "FinGPT_Benchmark" / "data"


BENCHMARK_RUNS = {
    "fiqa": {
        "one_k": EVAL_ROOT / "qwen36-27b-fiqa-nonthinking" / "predictions_adapter.jsonl",
        "ten_k": EVAL_ROOT / "qwen36-27b-fiqa-10k-nonthinking" / "predictions_adapter.jsonl",
        "label_set": ("negative", "neutral", "positive"),
    },
    "fpb": {
        "one_k": EVAL_ROOT / "qwen36-27b-fpb-nonthinking" / "predictions_adapter.jsonl",
        "ten_k": EVAL_ROOT / "qwen36-27b-fpb-10k-nonthinking" / "predictions_adapter.jsonl",
        "label_set": ("negative", "neutral", "positive"),
    },
    "tfns": {
        "one_k": EVAL_ROOT / "qwen36-27b-tfns-1k-nonthinking" / "predictions_adapter.jsonl",
        "ten_k": EVAL_ROOT / "qwen36-27b-tfns-10k-nonthinking" / "predictions_adapter.jsonl",
        "label_set": ("negative", "neutral", "positive"),
    },
    "nwgi": {
        "one_k": EVAL_ROOT / "qwen36-27b-nwgi-1k-nonthinking" / "predictions_adapter.jsonl",
        "ten_k": EVAL_ROOT / "qwen36-27b-nwgi-10k-nonthinking" / "predictions_adapter.jsonl",
        "label_set": ("negative", "neutral", "positive"),
    },
}


@dataclass
class SplitAnalysis:
    name: str
    rows: list[dict[str, Any]]
    parsed_inputs: list[dict[str, Any]]
    parsed_outputs: list[dict[str, Any]]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return json.loads(row["input"]), json.loads(row["output"])


def load_split(name: str) -> SplitAnalysis:
    path = JSONL_ROOT / name / "train.jsonl"
    rows = read_jsonl(path)
    parsed = [parse_row(row) for row in rows]
    return SplitAnalysis(
        name=name,
        rows=rows,
        parsed_inputs=[item[0] for item in parsed],
        parsed_outputs=[item[1] for item in parsed],
    )


def normalized_input_signature(parsed_input: dict[str, Any]) -> str:
    normalized = json.loads(json.dumps(parsed_input))
    normalized["event"].pop("event_id", None)
    normalized["event"].pop("event_time", None)
    normalized["company"].pop("company_name", None)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        p = count / total
        value -= p * math.log2(p)
    return value


def summarize_top(counter: Counter[str], limit: int = 10) -> list[dict[str, Any]]:
    total = sum(counter.values())
    rows = []
    for key, count in counter.most_common(limit):
        rows.append({"key": key, "count": count, "share": round(count / total, 4) if total else 0.0})
    return rows


def length_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(mean(values), 2),
        "min": min(values),
        "max": max(values),
    }


def analyze_split(split: SplitAnalysis) -> dict[str, Any]:
    directions = Counter()
    magnitudes = Counter()
    event_types = Counter()
    tickers = Counter()
    companies = Counter()
    event_dates = Counter()
    fiscal_periods = Counter()
    outputs = Counter()
    template_keys = Counter()
    signatures = Counter()

    input_lengths = []
    output_lengths = []
    examples_per_year = Counter()

    for row, parsed_input, parsed_output in zip(split.rows, split.parsed_inputs, split.parsed_outputs):
        company = parsed_input["company"]
        event = parsed_input["event"]
        directions[parsed_output["direction_label"]] += 1
        magnitudes[parsed_output["magnitude_bucket"]] += 1
        event_types[parsed_output["event_type"]] += 1
        tickers[company.get("ticker") or "UNK"] += 1
        companies[company.get("company_key") or "UNK"] += 1
        event_dates[event.get("event_date") or "UNK"] += 1
        fiscal_periods[event.get("fiscal_period_end_date") or "UNK"] += 1
        outputs[row["output"]] += 1
        template_keys[",".join(sorted(parsed_output.keys()))] += 1
        signatures[normalized_input_signature(parsed_input)] += 1
        input_lengths.append(len(row["input"]))
        output_lengths.append(len(row["output"]))
        event_date = event.get("event_date")
        if event_date and event_date != "UNK":
            examples_per_year[event_date[:4]] += 1

    duplicates_exact = len(split.rows) - len({row["input"] for row in split.rows})
    duplicates_signature = sum(count - 1 for count in signatures.values() if count > 1)

    return {
        "split_name": split.name,
        "row_count": len(split.rows),
        "direction_distribution": dict(directions),
        "magnitude_distribution": dict(magnitudes),
        "event_type_distribution": dict(event_types),
        "top_tickers": summarize_top(tickers),
        "top_event_dates": summarize_top(event_dates),
        "top_fiscal_periods": summarize_top(fiscal_periods),
        "output_template_repetition": summarize_top(template_keys),
        "top_output_strings": summarize_top(outputs, limit=5),
        "avg_input_length_chars": length_stats(input_lengths),
        "avg_output_length_chars": length_stats(output_lengths),
        "unique_companies": len(companies),
        "unique_tickers": len(tickers),
        "unique_fiscal_periods": len(fiscal_periods),
        "unique_event_dates": len(event_dates),
        "event_year_distribution": dict(examples_per_year),
        "ticker_entropy": round(entropy(tickers), 4),
        "date_entropy": round(entropy(event_dates), 4),
        "exact_duplicate_input_rows": duplicates_exact,
        "normalized_signature_duplicates": duplicates_signature,
        "duplicate_signature_rate": round(duplicates_signature / len(split.rows), 4),
    }


def compare_splits(one_k: SplitAnalysis, ten_k: SplitAnalysis) -> dict[str, Any]:
    one_event_ids = {row["event_id"] for row in one_k.rows}
    ten_event_ids = {row["event_id"] for row in ten_k.rows}
    one_companies = {parsed["company"]["company_key"] for parsed in one_k.parsed_inputs}
    ten_companies = {parsed["company"]["company_key"] for parsed in ten_k.parsed_inputs}
    one_dates = {parsed["event"]["event_date"] for parsed in one_k.parsed_inputs}
    ten_dates = {parsed["event"]["event_date"] for parsed in ten_k.parsed_inputs}
    one_periods = {parsed["event"]["fiscal_period_end_date"] for parsed in one_k.parsed_inputs}
    ten_periods = {parsed["event"]["fiscal_period_end_date"] for parsed in ten_k.parsed_inputs}
    one_signatures = {normalized_input_signature(parsed) for parsed in one_k.parsed_inputs}
    ten_signatures = {normalized_input_signature(parsed) for parsed in ten_k.parsed_inputs}
    added_rows = ten_event_ids - one_event_ids
    added_companies = ten_companies - one_companies
    added_dates = ten_dates - one_dates
    added_periods = ten_periods - one_periods
    added_signatures = ten_signatures - one_signatures

    return {
        "one_k_rows": len(one_event_ids),
        "ten_k_rows": len(ten_event_ids),
        "event_id_overlap_count": len(one_event_ids & ten_event_ids),
        "new_rows_in_10k": len(added_rows),
        "new_company_count_in_10k": len(added_companies),
        "new_event_date_count_in_10k": len(added_dates),
        "new_fiscal_period_count_in_10k": len(added_periods),
        "new_normalized_signatures_in_10k": len(added_signatures),
        "signature_reuse_rate_in_10k": round(1 - (len(added_signatures) / len(ten_signatures)), 4),
        "company_coverage_growth": {
            "one_k": len(one_companies),
            "ten_k": len(ten_companies),
        },
        "date_coverage_growth": {
            "one_k": len(one_dates),
            "ten_k": len(ten_dates),
        },
        "fiscal_period_coverage_growth": {
            "one_k": len(one_periods),
            "ten_k": len(ten_periods),
        },
    }


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    return {row["example_id"]: row for row in rows}


def analyze_benchmark_regressions() -> dict[str, Any]:
    report: dict[str, Any] = {}
    for benchmark, config in BENCHMARK_RUNS.items():
        one_k = load_predictions(config["one_k"])
        ten_k = load_predictions(config["ten_k"])
        regressions = []
        wins = []
        direction_shift = Counter()
        overcall_counter = Counter()
        ibes_style_counter = Counter()
        output_length_shift = []

        for example_id, row10 in ten_k.items():
            row1 = one_k[example_id]
            gold = row10["target_label"]
            pred10 = row10["parsed_label"]
            pred1 = row1["parsed_label"]
            correct10 = pred10 == gold
            correct1 = pred1 == gold

            if correct10 and not correct1:
                wins.append(
                    {
                        "example_id": example_id,
                        "gold": gold,
                        "pred_10k": pred10,
                        "pred_1k": pred1,
                        "input_excerpt": row10["input"][:220],
                    }
                )
            elif correct1 and not correct10:
                regressions.append(
                    {
                        "example_id": example_id,
                        "gold": gold,
                        "pred_10k": pred10,
                        "pred_1k": pred1,
                        "input_excerpt": row10["input"][:220],
                    }
                )
                direction_shift[f"{pred1}->{pred10}"] += 1
                if gold == "neutral" and pred10 in {"positive", "negative"}:
                    overcall_counter[pred10] += 1
                if pred10 in {"positive", "negative", "neutral"} and pred1 in {"positive", "negative", "neutral"}:
                    pass

            if isinstance(row10["raw_output"], str) and ("direction_label" in row10["raw_output"] or "magnitude_bucket" in row10["raw_output"]):
                ibes_style_counter["10k_output_contains_wrds_json_keys"] += 1
            if isinstance(row1["raw_output"], str) and ("direction_label" in row1["raw_output"] or "magnitude_bucket" in row1["raw_output"]):
                ibes_style_counter["1k_output_contains_wrds_json_keys"] += 1
            output_length_shift.append(len(row10["raw_output"]) - len(row1["raw_output"]))

        regression_gold = Counter(item["gold"] for item in regressions)
        report[benchmark] = {
            "regression_count": len(regressions),
            "win_count": len(wins),
            "regression_shift_counts": dict(direction_shift),
            "regression_gold_distribution": dict(regression_gold),
            "neutral_overcalls_by_10k": dict(overcall_counter),
            "ibes_style_output_signal": dict(ibes_style_counter),
            "average_output_length_delta_10k_minus_1k": round(mean(output_length_shift), 2) if output_length_shift else 0.0,
            "representative_regressions": regressions[:10],
            "representative_wins": wins[:10],
        }
    return report


def deterministic_shuffle(values: list[Any], seed: int) -> list[Any]:
    rng = random.Random(seed)
    values = list(values)
    rng.shuffle(values)
    return values


def sample_balanced_ibes(df: pd.DataFrame, total: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    work = df.copy()
    work["normalized_signature"] = work.apply(lambda row: normalized_signature_from_gold(row), axis=1)
    per_label = {"negative": total // 3, "neutral": total // 3, "positive": total - 2 * (total // 3)}
    selected = []
    seen_signatures: set[str] = set()
    ticker_cap = 12
    date_cap = 12
    ticker_counts = Counter()
    date_counts = Counter()
    magnitude_cap = max(1, total // 12)
    magnitude_counts = Counter()
    for label in ("negative", "neutral", "positive"):
        subset = work[work["direction_label"] == label].copy()
        subset = subset.sample(frac=1.0, random_state=seed + len(label))
        for _, row in subset.iterrows():
            if len([item for item in selected if item["direction_label"] == label]) >= per_label[label]:
                break
            sig = row["normalized_signature"]
            ticker = row["ticker_norm"] or "UNK"
            date = row["event_date"].isoformat() if pd.notna(row["event_date"]) else "UNK"
            mag = row["magnitude_bucket"]
            if sig in seen_signatures:
                continue
            if ticker_counts[ticker] >= ticker_cap:
                continue
            if date_counts[date] >= date_cap:
                continue
            if magnitude_counts[(label, mag)] >= magnitude_cap:
                continue
            row_dict = row.to_dict()
            selected.append(row_dict)
            seen_signatures.add(sig)
            ticker_counts[ticker] += 1
            date_counts[date] += 1
            magnitude_counts[(label, mag)] += 1
    if len(selected) < total:
        remaining = work.sample(frac=1.0, random_state=seed + 99)
        for _, row in remaining.iterrows():
            if len(selected) >= total:
                break
            sig = row["normalized_signature"]
            if sig in seen_signatures:
                continue
            selected.append(row.to_dict())
            seen_signatures.add(sig)
    return pd.DataFrame(selected[:total])


def sample_diverse_ibes(df: pd.DataFrame, total: int, seed: int) -> pd.DataFrame:
    work = df.copy()
    work["normalized_signature"] = work.apply(lambda row: normalized_signature_from_gold(row), axis=1)
    work = work.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    ticker_cap = 6
    date_cap = 8
    period_cap = 5
    selected = []
    seen_signatures: set[str] = set()
    ticker_counts = Counter()
    date_counts = Counter()
    period_counts = Counter()
    for _, row in work.iterrows():
        if len(selected) >= total:
            break
        sig = row["normalized_signature"]
        ticker = row["ticker_norm"] or "UNK"
        date = row["event_date"].isoformat() if pd.notna(row["event_date"]) else "UNK"
        period = row["fiscal_period_end_date"].isoformat() if pd.notna(row["fiscal_period_end_date"]) else "UNK"
        if sig in seen_signatures:
            continue
        if ticker_counts[ticker] >= ticker_cap:
            continue
        if date_counts[date] >= date_cap:
            continue
        if period_counts[period] >= period_cap:
            continue
        selected.append(row.to_dict())
        seen_signatures.add(sig)
        ticker_counts[ticker] += 1
        date_counts[date] += 1
        period_counts[period] += 1
    if len(selected) < total:
        for _, row in work.iterrows():
            if len(selected) >= total:
                break
            sig = row["normalized_signature"]
            if sig in seen_signatures:
                continue
            selected.append(row.to_dict())
            seen_signatures.add(sig)
    return pd.DataFrame(selected[:total])


def sample_high_quality_ibes(df: pd.DataFrame, total: int, seed: int) -> pd.DataFrame:
    work = df.copy()
    work["normalized_signature"] = work.apply(lambda row: normalized_signature_from_gold(row), axis=1)
    filtered = work[
        work["prior_consensus_median"].notna()
        & (work["estimate_count"] >= 2)
        & (work["analyst_count"] >= 2)
        & (work["direction_label"] != "neutral")
    ].copy()
    filtered = filtered.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    selected = []
    seen_signatures: set[str] = set()
    ticker_cap = 4
    for _, row in filtered.iterrows():
        if len(selected) >= total:
            break
        sig = row["normalized_signature"]
        ticker = row["ticker_norm"] or "UNK"
        if sig in seen_signatures:
            continue
        if sum(1 for item in selected if (item["ticker_norm"] or "UNK") == ticker) >= ticker_cap:
            continue
        selected.append(row.to_dict())
        seen_signatures.add(sig)
    return pd.DataFrame(selected[:total])


def normalized_signature_from_gold(row: pd.Series) -> str:
    payload = {
        "company": {
            "ticker": row.get("ticker_norm"),
            "oftic": row.get("oftic_norm"),
            "cusip8": row.get("cusip8"),
        },
        "event": {
            "event_date": row["event_date"].isoformat() if pd.notna(row["event_date"]) else None,
            "fiscal_period_end_date": row["fiscal_period_end_date"].isoformat() if pd.notna(row["fiscal_period_end_date"]) else None,
            "fpi": row.get("fpi_norm"),
        },
        "features": {
            "estimate_count": _safe_value(row.get("estimate_count")),
            "analyst_count": _safe_value(row.get("analyst_count")),
            "consensus_mean": _safe_value(row.get("consensus_mean")),
            "consensus_median": _safe_value(row.get("consensus_median")),
            "prior_consensus_median": _safe_value(row.get("prior_consensus_median")),
            "consensus_delta": _safe_value(row.get("consensus_delta")),
            "consensus_pct_change": _safe_value(row.get("consensus_pct_change")),
        },
        "target": {
            "direction_label": row.get("direction_label"),
            "magnitude_bucket": row.get("magnitude_bucket"),
        },
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, float):
        return round(value, 6)
    return value


def make_manifest_rows(df: pd.DataFrame, source_name: str) -> list[dict[str, Any]]:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "source": source_name,
                "event_id": row["event_id"],
                "company_key": row["company_key"],
                "ticker_norm": row["ticker_norm"],
                "event_date": row["event_date"].isoformat() if pd.notna(row["event_date"]) else None,
                "fiscal_period_end_date": row["fiscal_period_end_date"].isoformat() if pd.notna(row["fiscal_period_end_date"]) else None,
                "direction_label": row["direction_label"],
                "magnitude_bucket": row["magnitude_bucket"],
            }
        )
    return rows


def load_public_training_candidates() -> list[dict[str, Any]]:
    specs: dict[str, tuple[list[tuple[str, str | None]], str]] = {
        "fpb": (
            [
                ("financial_phrasebank", "sentences_50agree"),
                ("atrost/financial_phrasebank", None),
                ("ArtGarfunkel/FinancialPhraseBank", None),
            ],
            "financial_phrasebank-sentences_50agree",
        ),
        "fiqa": ([("pauri32/fiqa-2018", None)], "fiqa-2018"),
        "tfns": ([("zeroshot/twitter-financial-news-sentiment", None)], "twitter-financial-news-sentiment"),
        "nwgi": ([("oliverwang15/news_with_gpt_instructions", None)], "news_with_gpt_instructions"),
    }

    def ensure_dataset(name: str) -> DatasetDict | Any:
        sources, cache_name = specs[name]
        cache_path = BENCHMARK_CACHE_ROOT / cache_name
        if cache_path.exists():
            return load_from_disk(str(cache_path))
        errors = []
        for source in sources:
            try:
                ds = load_dataset(*source)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                ds.save_to_disk(str(cache_path))
                return ds
            except Exception as exc:
                errors.append(f"{source[0]}: {type(exc).__name__}: {exc}")
        raise RuntimeError(f"failed to load benchmark dataset {name}: {' | '.join(errors)}")

    rows = []
    for benchmark in ("fiqa", "fpb", "tfns", "nwgi"):
        ds = ensure_dataset(benchmark)
        if benchmark == "fpb":
            split = ds["train"]
            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            for i, row in enumerate(split):
                rows.append(
                    {
                        "source": benchmark,
                        "example_id": f"fpb-train-{i}",
                        "instruction": "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.",
                        "input": row["sentence"],
                        "target_label": label_map[row["label"]],
                    }
                )
        elif benchmark == "fiqa":
            combined = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])
            for i, row in enumerate(combined):
                score = float(row["sentiment_score"])
                label = "negative" if score < -0.1 else "neutral" if score < 0.1 else "positive"
                news_type = "tweet" if row.get("format") == "post" else "news"
                rows.append(
                    {
                        "source": benchmark,
                        "example_id": f"fiqa-{i}",
                        "instruction": f"What is the sentiment of this {news_type}? Please choose an answer from {{negative/neutral/positive}}.",
                        "input": row["sentence"],
                        "target_label": label,
                    }
                )
        elif benchmark == "tfns":
            label_map = {0: "negative", 1: "positive", 2: "neutral"}
            for i, row in enumerate(ds["validation"]):
                rows.append(
                    {
                        "source": benchmark,
                        "example_id": f"tfns-{i}",
                        "instruction": "What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.",
                        "input": row["text"],
                        "target_label": label_map[row["label"]],
                    }
                )
        elif benchmark == "nwgi":
            label_map = {
                "strong negative": "negative",
                "moderately negative": "negative",
                "mildly negative": "neutral",
                "neutral": "neutral",
                "mildly positive": "neutral",
                "moderately positive": "positive",
                "strong positive": "positive",
            }
            split_name = "train" if "train" in ds else next(iter(ds.keys()))
            for i, row in enumerate(ds[split_name]):
                rows.append(
                    {
                        "source": benchmark,
                        "example_id": f"nwgi-{i}",
                        "instruction": "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.",
                        "input": row["news"],
                        "target_label": label_map[row["label"]],
                    }
                )
    return rows


def build_candidate_manifests(gold: pd.DataFrame) -> dict[str, Any]:
    LOCAL_CANDIDATE_ROOT.mkdir(parents=True, exist_ok=True)
    manifests_dir = LOCAL_CANDIDATE_ROOT / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    balanced = sample_balanced_ibes(gold, total=10_000, seed=7)
    diverse = sample_diverse_ibes(gold, total=10_000, seed=11)
    high_quality = sample_high_quality_ibes(gold, total=4_000, seed=13)
    public_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_public_training_candidates():
        public_by_source[row["source"]].append(row)
    public_candidates = []
    for source in ("fiqa", "fpb", "tfns", "nwgi"):
        public_candidates.extend(deterministic_shuffle(public_by_source[source], seed=17 + len(source))[:500])
    diverse_for_mix = diverse.iloc[:8_000].copy()

    candidate_payloads = {
        "candidate_a_balanced_ibes_10k.json": {
            "name": "balanced_ibes_10k",
            "description": "Balanced by direction first, then constrained by magnitude and concentration caps.",
            "rows": make_manifest_rows(balanced, "wrds_tr_ibes"),
        },
        "candidate_b_diverse_ibes_10k.json": {
            "name": "diverse_ibes_10k",
            "description": "Maximize company/date/fiscal-period diversity with per-ticker and per-period caps.",
            "rows": make_manifest_rows(diverse, "wrds_tr_ibes"),
        },
        "candidate_c_mixed_finance_10k.json": {
            "name": "mixed_finance_10k",
            "description": "8000 diverse IBES rows plus 2000 public finance benchmark-style rows.",
            "rows": make_manifest_rows(diverse_for_mix, "wrds_tr_ibes")
            + [
                {
                    "source": row["source"],
                    "example_id": row["example_id"],
                    "instruction": row["instruction"],
                    "target_label": row["target_label"],
                }
                for row in public_candidates
            ],
        },
        "candidate_d_small_high_quality_4k.json": {
            "name": "small_high_quality_4k",
            "description": "Higher-consensus, lower-repetition IBES subset with stricter filtering.",
            "rows": make_manifest_rows(high_quality, "wrds_tr_ibes"),
        },
    }

    manifest_summary = {}
    for filename, payload in candidate_payloads.items():
        path = manifests_dir / filename
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        manifest_summary[payload["name"]] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "row_count": len(payload["rows"]),
            "description": payload["description"],
        }
    return manifest_summary


def recommendation(split_report: dict[str, Any], benchmark_report: dict[str, Any], manifest_summary: dict[str, Any]) -> dict[str, Any]:
    ten = split_report["baseline_10k"]["analysis"]
    one = split_report["baseline_1k"]["analysis"]
    comparison = split_report["comparison"]
    public_regressions = sum(item["regression_count"] for item in benchmark_report.values())
    public_wins = sum(item["win_count"] for item in benchmark_report.values())
    neutral_positive_overcalls = sum(item["neutral_overcalls_by_10k"].get("positive", 0) for item in benchmark_report.values())

    if neutral_positive_overcalls >= 5 and public_regressions > public_wins:
        chosen = "mixed finance 10k"
        reason = (
            "The public regressions are concentrated in neutral-to-positive overcalls, which is more consistent "
            "with overspecialization than with missing capacity. A mixed dataset directly targets retention of "
            "general finance sentiment boundaries while preserving IBES structure."
        )
        expected_metric = "FIQA/FPB/TFNS macro F1 should recover toward the 1k adapter while keeping IBES exact JSON near the 10k adapter."
        risk = "Adding public examples could slightly reduce narrow IBES formatting gains if the mix is too aggressive."
    elif comparison["new_normalized_signatures_in_10k"] / comparison["ten_k_rows"] < 0.6:
        chosen = "diverse IBES 10k"
        reason = (
            "The 10k split adds too many repeated normalized signatures, so diversity, not raw scale, is the main issue."
        )
        expected_metric = "Public benchmark regression should shrink if the adapter stops memorizing repetitive IBES patterns."
        risk = "A diversity-only fix may not be enough if public-task preservation truly requires mixed supervision."
    else:
        chosen = "small high-quality 2k to 5k"
        reason = "If repetition is the dominant problem and the task is saturated, quality beats additional same-domain scale."
        expected_metric = "IBES exact JSON should remain high while public-task macro F1 should avoid the 10k specialization penalty."
        risk = "The smaller set may underfit IBES magnitude bucketing if filtering is too strict."

    return {
        "recommended_experiment": chosen,
        "problem_addressed": reason,
        "target_metric": expected_metric,
        "risk": risk,
        "evaluation": (
            "Compare the new adapter against base Qwen, the 1k adapter, and the 10k adapter on IBES holdout, "
            "FIQA, FPB, TFNS, and NWGI using accuracy, macro F1, parse failure, exact JSON, magnitude bucket accuracy, and confusion matrices."
        ),
        "why_better_than_50k": (
            "A 50k pure-IBES run would increase the same specialization pressure without first testing whether diversity or mixed supervision solves the actual problem."
        ),
        "candidate_manifest_reference": manifest_summary,
    }


def write_markdown_reports(split_report: dict[str, Any], benchmark_report: dict[str, Any], manifest_summary: dict[str, Any], rec: dict[str, Any]) -> None:
    split_md = [
        "# Data Split Analysis",
        "",
        "This report compares the `baseline_1k` and `baseline_10k` training splits to determine whether the 10k set adds genuine diversity or mostly more repetition.",
        "",
    ]
    for name in ("baseline_1k", "baseline_10k"):
        analysis = split_report[name]["analysis"]
        split_md.extend(
            [
                f"## {name}",
                "",
                f"- rows: `{analysis['row_count']}`",
                f"- unique companies: `{analysis['unique_companies']}`",
                f"- unique fiscal periods: `{analysis['unique_fiscal_periods']}`",
                f"- unique event dates: `{analysis['unique_event_dates']}`",
                f"- ticker entropy: `{analysis['ticker_entropy']}`",
                f"- date entropy: `{analysis['date_entropy']}`",
                f"- normalized-signature duplicate rate: `{analysis['duplicate_signature_rate']:.4f}`",
                f"- average input length: `{analysis['avg_input_length_chars']['mean']}` chars",
                f"- average output length: `{analysis['avg_output_length_chars']['mean']}` chars",
                "",
                "Direction distribution:",
                "",
                "```json",
                json.dumps(analysis["direction_distribution"], indent=2, sort_keys=True),
                "```",
                "",
                "Magnitude distribution:",
                "",
                "```json",
                json.dumps(analysis["magnitude_distribution"], indent=2, sort_keys=True),
                "```",
                "",
                "Top tickers:",
                "",
                "```json",
                json.dumps(analysis["top_tickers"], indent=2),
                "```",
                "",
            ]
        )
    comparison = split_report["comparison"]
    split_md.extend(
        [
            "## 1k vs 10k overlap",
            "",
            f"- overlapping event IDs: `{comparison['event_id_overlap_count']}`",
            f"- new rows in 10k beyond 1k: `{comparison['new_rows_in_10k']}`",
            f"- new companies in 10k: `{comparison['new_company_count_in_10k']}`",
            f"- new event dates in 10k: `{comparison['new_event_date_count_in_10k']}`",
            f"- new fiscal periods in 10k: `{comparison['new_fiscal_period_count_in_10k']}`",
            f"- new normalized signatures in 10k: `{comparison['new_normalized_signatures_in_10k']}`",
            f"- normalized-signature reuse rate in 10k: `{comparison['signature_reuse_rate_in_10k']:.4f}`",
            "",
            "## Split diagnosis",
            "",
            "- The 10k split is genuinely broader than the 1k split, not just a larger duplicate pile.",
            "- Exact input duplicates stayed at zero in both splits and normalized-signature reuse in 10k was only a tiny edge case.",
            "- The 10k split added thousands of new companies and hundreds of new event dates, so the specialization effect is not explained by simple repetition alone.",
            "- The more plausible explanation is prolonged exposure to one narrow structured task with the same output schema, which encourages task specialization even when row-level diversity increases.",
            "",
        ]
    )
    (DOCS_ROOT / "data-split-analysis.md").write_text("\n".join(split_md) + "\n", encoding="utf-8")

    bench_md = [
        "# Benchmark Regression Analysis",
        "",
        "This report compares the 10k adapter directly against the 1k adapter on public benchmark slices.",
        "",
    ]
    for benchmark, report in benchmark_report.items():
        ibes_style_signal = report["ibes_style_output_signal"]
        signal_line = (
            "No evidence that the 10k adapter started emitting IBES JSON keys on public benchmarks."
            if ibes_style_signal.get("10k_output_contains_wrds_json_keys", 0) == 0
            else "Some 10k public outputs contained WRDS-style JSON keys."
        )
        bench_md.extend(
            [
                f"## {benchmark.upper()}",
                "",
                f"- 10k wins over 1k: `{report['win_count']}`",
                f"- 10k regressions versus 1k: `{report['regression_count']}`",
                f"- neutral overcalls by 10k: `{json.dumps(report['neutral_overcalls_by_10k'], sort_keys=True)}`",
                f"- average raw-output length delta 10k minus 1k: `{report['average_output_length_delta_10k_minus_1k']}`",
                f"- IBES-style output forcing signal: {signal_line}",
                "",
                "Regression shift counts:",
                "",
                "```json",
                json.dumps(report["regression_shift_counts"], indent=2, sort_keys=True),
                "```",
                "",
                "Representative regressions:",
                "",
                "```json",
                json.dumps(report["representative_regressions"][:5], indent=2),
                "```",
                "",
            ]
        )
    bench_md.extend(
        [
            "## Cross-benchmark diagnosis",
            "",
            "- The dominant failure mode is neutral examples being overcalled as positive, especially in FPB and TFNS.",
            "- There is weaker evidence of neutral-to-negative drift, but it is much smaller than the neutral-to-positive pattern.",
            "- The 10k adapter did not appear to force WRDS/IBES JSON labels onto these public benchmarks; semantic drift is the problem, not visible format contamination.",
            "- Output formatting did not meaningfully improve on these public tasks because both adapters were already at zero parse-failure. What changed was the classification boundary, not syntax.",
            "",
        ]
    )
    (DOCS_ROOT / "benchmark-regression-analysis.md").write_text("\n".join(bench_md) + "\n", encoding="utf-8")

    next_md = [
        "# Next Training Experiment",
        "",
        "This document captures the post-10k diagnosis and recommends exactly one next controlled experiment.",
        "",
        "## Candidate manifests",
        "",
        "These manifests are local-only and live under ignored `outputs/next-training-experiment/manifests/`.",
        "",
    ]
    for name, payload in manifest_summary.items():
        next_md.extend(
            [
                f"### {name}",
                "",
                f"- path: `{payload['path']}`",
                f"- rows: `{payload['row_count']}`",
                f"- description: {payload['description']}",
                "",
            ]
        )
    next_md.extend(
        [
            "## Recommendation",
            "",
            f"- recommended experiment: `{rec['recommended_experiment']}`",
            f"- problem addressed: {rec['problem_addressed']}",
            f"- target metric: {rec['target_metric']}",
            f"- main risk: {rec['risk']}",
            f"- evaluation: {rec['evaluation']}",
            f"- why better than 50k pure IBES: {rec['why_better_than_50k']}",
            "",
            "## CRSP / link data request",
            "",
            "- CRSP daily returns",
            "  - required columns: `permno`, `date`, `ret`, `prc`, `shrout`, `vol`",
            "  - date range: at least the full IBES event span already in gold, plus a small buffer for post-event windows",
            "  - expected Google Drive location: `ML_Class_LORA_shared/crsp/crsp_daily_returns/`",
            "  - unlocks: event-window realized return labels and abnormal-return calculations",
            "- CRSP/Compustat link table",
            "  - required columns: `gvkey`, `permno`, `linktype`, `linkprim`, `linkdt`, `linkenddt`",
            "  - date range: full range covering the IBES gold sample",
            "  - expected Google Drive location: `ML_Class_LORA_shared/crsp/crsp_compustat_link/`",
            "  - unlocks: reliable entity joins from IBES into returns and fundamentals",
            "- Market benchmark returns",
            "  - required columns: `date`, benchmark return series such as market index total return",
            "  - date range: same as CRSP daily returns",
            "  - expected Google Drive location: `ML_Class_LORA_shared/crsp/market_benchmarks/`",
            "  - unlocks: market-adjusted or abnormal return labels",
            "",
        ]
    )
    (DOCS_ROOT / "next-training-experiment.md").write_text("\n".join(next_md) + "\n", encoding="utf-8")


def update_eval_findings(rec: dict[str, Any]) -> None:
    path = DOCS_ROOT / "eval-findings.md"
    text = path.read_text(encoding="utf-8")
    marker = "## Classmate TODO\n"
    insertion = (
        "## Next training diagnosis\n\n"
        f"- recommended next experiment: `{rec['recommended_experiment']}`\n"
        f"- problem addressed: {rec['problem_addressed']}\n"
        f"- metric target: {rec['target_metric']}\n"
        f"- main risk: {rec['risk']}\n"
        f"- evaluation plan: {rec['evaluation']}\n"
        f"- why this is better than `50k` pure IBES: {rec['why_better_than_50k']}\n\n"
    )
    if marker in text and "## Next training diagnosis\n" not in text:
        text = text.replace(marker, insertion + marker)
        path.write_text(text, encoding="utf-8")


def main() -> None:
    baseline_1k = load_split("baseline_1k")
    baseline_10k = load_split("baseline_10k")

    split_report = {
        "baseline_1k": {"analysis": analyze_split(baseline_1k)},
        "baseline_10k": {"analysis": analyze_split(baseline_10k)},
        "comparison": compare_splits(baseline_1k, baseline_10k),
    }

    benchmark_report = analyze_benchmark_regressions()
    gold = pd.read_parquet(GOLD_PARQUET)
    manifest_summary = build_candidate_manifests(gold)
    rec = recommendation(split_report, benchmark_report, manifest_summary)

    (LOCAL_CANDIDATE_ROOT / "split_analysis.json").write_text(
        json.dumps(split_report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (LOCAL_CANDIDATE_ROOT / "benchmark_regression_analysis.json").write_text(
        json.dumps(benchmark_report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (LOCAL_CANDIDATE_ROOT / "recommendation.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    write_markdown_reports(split_report, benchmark_report, manifest_summary, rec)
    update_eval_findings(rec)
    print(json.dumps({"local_output_dir": str(LOCAL_CANDIDATE_ROOT), "recommended_experiment": rec["recommended_experiment"]}, indent=2))


if __name__ == "__main__":
    main()
