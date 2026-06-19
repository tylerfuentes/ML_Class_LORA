#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, TypeVar

import pandas as pd
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ibes_pipeline import build_lora_row, write_jsonl

GOLD_PARQUET = REPO_ROOT / "data" / "processed" / "ibes_lora_baseline" / "gold" / "ibes_revision_events.parquet"
COMMON_EVAL_FILE = REPO_ROOT / "data" / "processed" / "ibes_lora_baseline" / "jsonl" / "baseline_10k" / "eval.jsonl"
COMMON_HOLDOUT_FILE = REPO_ROOT / "data" / "processed" / "ibes_lora_baseline" / "jsonl" / "baseline_10k" / "holdout.jsonl"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "overnight_tournament"
CANDIDATE_ROOT = OUTPUT_ROOT / "candidates"
BENCHMARK_CACHE_ROOT = REPO_ROOT / "external" / "FinGPT" / "fingpt" / "FinGPT_Benchmark" / "data"

PUBLIC_DATASET_SPECS: dict[str, tuple[list[tuple[str, str | None]], str]] = {
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

CANDIDATE_ORDER = [
    "high_quality_ibes_4k",
    "balanced_ibes_10k",
    "mixed_finance_10k",
    "diverse_ibes_10k",
]

T = TypeVar("T")


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def normalized_ibes_signature(row: pd.Series) -> str:
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
            "estimate_count": safe_value(row.get("estimate_count")),
            "analyst_count": safe_value(row.get("analyst_count")),
            "consensus_mean": safe_value(row.get("consensus_mean")),
            "consensus_median": safe_value(row.get("consensus_median")),
            "prior_consensus_median": safe_value(row.get("prior_consensus_median")),
            "consensus_delta": safe_value(row.get("consensus_delta")),
            "consensus_pct_change": safe_value(row.get("consensus_pct_change")),
        },
        "target": {
            "direction_label": row.get("direction_label"),
            "magnitude_bucket": row.get("magnitude_bucket"),
        },
    }
    return sha1_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def normalized_public_signature(row: dict[str, Any]) -> str:
    payload = {
        "source": row["source"],
        "instruction": normalize_whitespace(row["instruction"]),
        "input": normalize_whitespace(row["input"]),
        "label": row["target_label"],
    }
    return sha1_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, float):
        return round(value, 6)
    return value


def deterministic_shuffle(values: list[T], seed: int) -> list[T]:
    rng = random.Random(seed)
    copied = list(values)
    rng.shuffle(copied)
    return copied


def ensure_public_dataset(name: str) -> DatasetDict | Any:
    sources, cache_name = PUBLIC_DATASET_SPECS[name]
    cache_path = BENCHMARK_CACHE_ROOT / cache_name
    if cache_path.exists():
        return load_from_disk(str(cache_path))
    errors = []
    for source, config in sources:
        try:
            dataset = load_dataset(source, config) if config is not None else load_dataset(source)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(cache_path))
            return dataset
        except Exception as exc:  # pragma: no cover - exercised only when remote fetch fails
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"failed to load public dataset {name}: {' | '.join(errors)}")


def load_public_examples() -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    for benchmark in ("fiqa", "fpb", "tfns", "nwgi"):
        try:
            dataset = ensure_public_dataset(benchmark)
        except Exception as exc:  # pragma: no cover - depends on network/local cache
            errors.append(f"{benchmark}: {exc}")
            continue

        if benchmark == "fpb":
            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            split = dataset["train"]
            for idx, row in enumerate(split):
                rows.append(
                    {
                        "source": benchmark,
                        "example_id": f"fpb-train-{idx}",
                        "instruction": "What is the sentiment of this finance news sentence? Please choose an answer from {negative/neutral/positive}.",
                        "input": row["sentence"],
                        "output": label_map[row["label"]],
                        "target_label": label_map[row["label"]],
                    }
                )
        elif benchmark == "fiqa":
            combined = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
            for idx, row in enumerate(combined):
                score = float(row["sentiment_score"])
                label = "negative" if score < -0.1 else "neutral" if score < 0.1 else "positive"
                item_type = "tweet" if row.get("format") == "post" else "news"
                rows.append(
                    {
                        "source": benchmark,
                        "example_id": f"fiqa-{idx}",
                        "instruction": f"What is the sentiment of this financial {item_type}? Please choose an answer from {{negative/neutral/positive}}.",
                        "input": row["sentence"],
                        "output": label,
                        "target_label": label,
                    }
                )
        elif benchmark == "tfns":
            label_map = {0: "negative", 1: "positive", 2: "neutral"}
            for idx, row in enumerate(dataset["validation"]):
                rows.append(
                    {
                        "source": benchmark,
                        "example_id": f"tfns-{idx}",
                        "instruction": "What is the sentiment of this finance tweet? Please choose an answer from {negative/neutral/positive}.",
                        "input": row["text"],
                        "output": label_map[row["label"]],
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
            split_name = "train" if "train" in dataset else next(iter(dataset.keys()))
            for idx, row in enumerate(dataset[split_name]):
                rows.append(
                    {
                        "source": benchmark,
                        "example_id": f"nwgi-{idx}",
                        "instruction": "What is the sentiment of this finance news item? Please choose an answer from {negative/neutral/positive}.",
                        "input": row["news"],
                        "output": label_map[row["label"]],
                        "target_label": label_map[row["label"]],
                    }
                )
    return rows, errors


def top_share(counter: Counter[str], limit: int = 10) -> list[dict[str, Any]]:
    total = sum(counter.values())
    rows = []
    for key, count in counter.most_common(limit):
        rows.append({"key": key, "count": count, "share": round(count / total, 4) if total else 0.0})
    return rows


def length_summary(values: list[int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": round(mean(values), 2), "min": min(values), "max": max(values)}


def build_ibes_examples(df: pd.DataFrame) -> list[dict[str, Any]]:
    examples = []
    for _, row in df.iterrows():
        example = build_lora_row(row)
        example["candidate_signature"] = row["candidate_signature"]
        examples.append(example)
    return examples


def summarize_candidate(
    name: str,
    candidate_type: str,
    reason: str,
    risk_tested: str,
    train_examples: list[dict[str, Any]],
    ibes_rows: pd.DataFrame,
    public_rows: list[dict[str, Any]],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    label_distribution = Counter()
    magnitude_distribution = Counter()
    source_distribution = Counter()
    ticker_counter = Counter()
    company_counter = Counter()
    event_dates = Counter()
    fiscal_periods = Counter()
    signatures = Counter()
    input_lengths = []
    output_lengths = []

    for example in train_examples:
        source_distribution[example.get("source", "unknown")] += 1
        input_lengths.append(len(example["input"]))
        output_lengths.append(len(example["output"]))
        if example.get("source") == "wrds_tr_ibes":
            parsed_output = json.loads(example["output"])
            parsed_input = json.loads(example["input"])
            label_distribution[parsed_output["direction_label"]] += 1
            magnitude_distribution[parsed_output["magnitude_bucket"]] += 1
            ticker_counter[parsed_input["company"].get("ticker") or "UNK"] += 1
            company_counter[parsed_input["company"].get("company_key") or "UNK"] += 1
            event_dates[parsed_input["event"].get("event_date") or "UNK"] += 1
            fiscal_periods[parsed_input["event"].get("fiscal_period_end_date") or "UNK"] += 1
            signatures[example["candidate_signature"]] += 1
        else:
            label_distribution[example["output"]] += 1
            signatures[example["candidate_signature"]] += 1

    duplicate_estimate = sum(count - 1 for count in signatures.values() if count > 1)
    ibes_date_min = None if ibes_rows.empty else ibes_rows["event_date"].min()
    ibes_date_max = None if ibes_rows.empty else ibes_rows["event_date"].max()
    public_label_distribution = Counter(row["target_label"] for row in public_rows)

    return {
        "name": name,
        "candidate_type": candidate_type,
        "row_count": len(train_examples),
        "ibes_row_count": int(len(ibes_rows)),
        "public_row_count": len(public_rows),
        "label_distribution": dict(label_distribution),
        "magnitude_distribution": dict(magnitude_distribution),
        "source_distribution": dict(source_distribution),
        "top_tickers": top_share(ticker_counter),
        "top_event_dates": top_share(event_dates),
        "top_fiscal_periods": top_share(fiscal_periods),
        "unique_ticker_count": len(ticker_counter),
        "unique_company_count": len(company_counter),
        "date_range": {
            "min": ibes_date_min.isoformat() if pd.notna(ibes_date_min) else None,
            "max": ibes_date_max.isoformat() if pd.notna(ibes_date_max) else None,
        },
        "avg_input_length_chars": length_summary(input_lengths),
        "avg_output_length_chars": length_summary(output_lengths),
        "duplicate_or_near_duplicate_estimate": duplicate_estimate,
        "duplicate_or_near_duplicate_rate": round(duplicate_estimate / len(train_examples), 4) if train_examples else 0.0,
        "mix_proportions": {
            source: round(count / len(train_examples), 4) for source, count in source_distribution.items()
        },
        "public_label_distribution": dict(public_label_distribution),
        "why_this_candidate_exists": reason,
        "risk_tested": risk_tested,
        "notes": notes or [],
        "common_eval_file": str(COMMON_EVAL_FILE.relative_to(REPO_ROOT)),
        "common_holdout_file": str(COMMON_HOLDOUT_FILE.relative_to(REPO_ROOT)),
    }


def write_candidate_card(card: dict[str, Any], output_path: Path) -> None:
    lines = [
        f"# {card['name']}",
        "",
        f"- candidate type: `{card['candidate_type']}`",
        f"- rows: `{card['row_count']}`",
        f"- IBES rows: `{card['ibes_row_count']}`",
        f"- public rows: `{card['public_row_count']}`",
        f"- why this candidate exists: {card['why_this_candidate_exists']}",
        f"- risk tested: {card['risk_tested']}",
        f"- common eval file: `{card['common_eval_file']}`",
        f"- common holdout file: `{card['common_holdout_file']}`",
        "",
        "## Core stats",
        "",
        f"- label distribution: `{json.dumps(card['label_distribution'], sort_keys=True)}`",
        f"- magnitude distribution: `{json.dumps(card['magnitude_distribution'], sort_keys=True)}`",
        f"- source distribution: `{json.dumps(card['source_distribution'], sort_keys=True)}`",
        f"- unique companies: `{card['unique_company_count']}`",
        f"- unique tickers: `{card['unique_ticker_count']}`",
        f"- date range: `{card['date_range']['min']}` to `{card['date_range']['max']}`",
        f"- average input length: `{card['avg_input_length_chars']['mean']}` chars",
        f"- average output length: `{card['avg_output_length_chars']['mean']}` chars",
        f"- duplicate/near-duplicate estimate: `{card['duplicate_or_near_duplicate_estimate']}` rows (`{card['duplicate_or_near_duplicate_rate']:.4f}`)",
        "",
        "## Mix proportions",
        "",
        "```json",
        json.dumps(card["mix_proportions"], indent=2, sort_keys=True),
        "```",
        "",
        "## Top ticker concentration",
        "",
        "```json",
        json.dumps(card["top_tickers"], indent=2),
        "```",
        "",
        "## Notes",
        "",
    ]
    if card["notes"]:
        lines.extend(f"- {note}" for note in card["notes"])
    else:
        lines.append("- no extra notes")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sample_balanced_ibes(df: pd.DataFrame, total: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    work = df.copy()
    quotas = {"negative": total // 3, "neutral": total // 3, "positive": total - 2 * (total // 3)}
    ticker_cap = max(8, total // 800)
    date_cap = max(8, total // 700)
    period_cap = max(6, total // 1200)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    ticker_counts = Counter()
    date_counts = Counter()
    period_counts = Counter()
    magnitude_counts = Counter()

    for direction in ("negative", "neutral", "positive"):
        subset = work[work["direction_label"] == direction].copy()
        subset["magnitude_sort_key"] = subset["magnitude_bucket"].fillna("unknown")
        per_magnitude_quota = max(1, math.ceil(quotas[direction] / max(1, subset["magnitude_sort_key"].nunique())))
        for magnitude in deterministic_shuffle(sorted(subset["magnitude_sort_key"].unique().tolist()), seed + len(direction)):
            magnitude_rows = subset[subset["magnitude_sort_key"] == magnitude].sample(frac=1.0, random_state=seed + len(magnitude))
            for _, row in magnitude_rows.iterrows():
                if sum(1 for item in selected if item["direction_label"] == direction) >= quotas[direction]:
                    break
                if magnitude_counts[(direction, magnitude)] >= per_magnitude_quota:
                    continue
                ticker = row["ticker_norm"] or "UNK"
                event_date = row["event_date"].isoformat() if pd.notna(row["event_date"]) else "UNK"
                period = row["fiscal_period_end_date"].isoformat() if pd.notna(row["fiscal_period_end_date"]) else "UNK"
                if row["candidate_signature"] in selected_ids:
                    continue
                if ticker_counts[ticker] >= ticker_cap or date_counts[event_date] >= date_cap or period_counts[period] >= period_cap:
                    continue
                selected.append(row.to_dict())
                selected_ids.add(row["candidate_signature"])
                ticker_counts[ticker] += 1
                date_counts[event_date] += 1
                period_counts[period] += 1
                magnitude_counts[(direction, magnitude)] += 1

    if len(selected) < total:
        remainder = work.sample(frac=1.0, random_state=seed + 99)
        for _, row in remainder.iterrows():
            if len(selected) >= total:
                break
            if row["candidate_signature"] in selected_ids:
                continue
            selected.append(row.to_dict())
            selected_ids.add(row["candidate_signature"])
    return pd.DataFrame(selected[:total])


def sample_diverse_ibes(df: pd.DataFrame, total: int, seed: int) -> pd.DataFrame:
    work = df.copy().sample(frac=1.0, random_state=seed).reset_index(drop=True)
    ticker_cap = max(4, total // 1600)
    date_cap = max(5, total // 1400)
    period_cap = max(4, total // 1800)
    selected = []
    selected_ids: set[str] = set()
    ticker_counts = Counter()
    date_counts = Counter()
    period_counts = Counter()
    company_counts = Counter()

    for _, row in work.iterrows():
        if len(selected) >= total:
            break
        ticker = row["ticker_norm"] or "UNK"
        company = row["company_key"] or "UNK"
        event_date = row["event_date"].isoformat() if pd.notna(row["event_date"]) else "UNK"
        period = row["fiscal_period_end_date"].isoformat() if pd.notna(row["fiscal_period_end_date"]) else "UNK"
        if row["candidate_signature"] in selected_ids:
            continue
        if ticker_counts[ticker] >= ticker_cap:
            continue
        if company_counts[company] >= ticker_cap:
            continue
        if date_counts[event_date] >= date_cap or period_counts[period] >= period_cap:
            continue
        selected.append(row.to_dict())
        selected_ids.add(row["candidate_signature"])
        ticker_counts[ticker] += 1
        company_counts[company] += 1
        date_counts[event_date] += 1
        period_counts[period] += 1

    if len(selected) < total:
        for _, row in work.iterrows():
            if len(selected) >= total:
                break
            if row["candidate_signature"] in selected_ids:
                continue
            selected.append(row.to_dict())
            selected_ids.add(row["candidate_signature"])
    return pd.DataFrame(selected[:total])


def sample_high_quality_ibes(df: pd.DataFrame, total: int, seed: int) -> pd.DataFrame:
    filtered = df[
        df["prior_consensus_median"].notna()
        & (df["estimate_count"] >= 2)
        & (df["analyst_count"] >= 2)
        & df["direction_label"].isin(["negative", "neutral", "positive"])
    ].copy()
    filtered["quality_score"] = (
        filtered["analyst_count"].fillna(0) * 3
        + filtered["estimate_count"].fillna(0) * 2
        - filtered["consensus_std"].fillna(filtered["consensus_std"].median())
    )
    filtered = filtered.sort_values(["quality_score", "event_date"], ascending=[False, True]).reset_index(drop=True)
    ticker_cap = max(3, total // 1200)
    direction_targets = {"negative": total // 3, "neutral": total // 3, "positive": total - 2 * (total // 3)}
    direction_counts = Counter()
    ticker_counts = Counter()
    selected = []
    selected_ids: set[str] = set()

    for _, row in filtered.iterrows():
        direction = row["direction_label"]
        ticker = row["ticker_norm"] or "UNK"
        if len(selected) >= total:
            break
        if row["candidate_signature"] in selected_ids:
            continue
        if direction_counts[direction] >= direction_targets[direction]:
            continue
        if ticker_counts[ticker] >= ticker_cap:
            continue
        selected.append(row.to_dict())
        selected_ids.add(row["candidate_signature"])
        direction_counts[direction] += 1
        ticker_counts[ticker] += 1

    if len(selected) < total:
        for _, row in filtered.iterrows():
            if len(selected) >= total:
                break
            if row["candidate_signature"] in selected_ids:
                continue
            selected.append(row.to_dict())
            selected_ids.add(row["candidate_signature"])
    return pd.DataFrame(selected[:total])


def sample_public_mix(public_rows: list[dict[str, Any]], per_source: int = 500) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in public_rows:
        grouped[row["source"]].append(row)
    selected: list[dict[str, Any]] = []
    for source in ("fiqa", "fpb", "tfns", "nwgi"):
        rows = grouped.get(source, [])
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in deterministic_shuffle(rows, seed=40 + len(source)):
            by_label[row["target_label"]].append(row)
        quotas = {"negative": per_source // 3, "neutral": per_source // 3, "positive": per_source - 2 * (per_source // 3)}
        chosen: list[dict[str, Any]] = []
        seen_signatures: set[str] = set()
        for label in ("negative", "neutral", "positive"):
            for row in by_label.get(label, []):
                if len([item for item in chosen if item["target_label"] == label]) >= quotas[label]:
                    break
                sig = normalized_public_signature(row)
                if sig in seen_signatures:
                    continue
                row_copy = dict(row)
                row_copy["candidate_signature"] = sig
                chosen.append(row_copy)
                seen_signatures.add(sig)
        selected.extend(chosen[:per_source])
    return selected


def write_candidate(name: str, train_examples: list[dict[str, Any]], card: dict[str, Any]) -> None:
    candidate_dir = CANDIDATE_ROOT / name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    clean_examples = []
    for example in train_examples:
        clean = dict(example)
        clean.pop("candidate_signature", None)
        clean_examples.append(clean)
    write_jsonl(clean_examples, candidate_dir / "train.jsonl")
    (candidate_dir / "dataset_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_candidate_card(card, candidate_dir / "dataset_card.md")
    manifest = {
        "name": name,
        "train_file": str((candidate_dir / "train.jsonl").relative_to(REPO_ROOT)),
        "dataset_card_json": str((candidate_dir / "dataset_card.json").relative_to(REPO_ROOT)),
        "dataset_card_md": str((candidate_dir / "dataset_card.md").relative_to(REPO_ROOT)),
        "common_eval_file": card["common_eval_file"],
        "common_holdout_file": card["common_holdout_file"],
        "row_count": card["row_count"],
        "ibes_row_count": card["ibes_row_count"],
        "public_row_count": card["public_row_count"],
    }
    (candidate_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    CANDIDATE_ROOT.mkdir(parents=True, exist_ok=True)
    gold = pd.read_parquet(GOLD_PARQUET)
    gold["candidate_signature"] = gold.apply(normalized_ibes_signature, axis=1)

    public_rows, public_errors = load_public_examples()

    high_quality = sample_high_quality_ibes(gold, total=4_000, seed=13)
    balanced = sample_balanced_ibes(gold, total=10_000, seed=7)
    diverse = sample_diverse_ibes(gold, total=10_000, seed=11)

    mixed_public = sample_public_mix(public_rows, per_source=500) if public_rows else []
    mixed_ibes = diverse.iloc[:8_000].copy()

    candidates = [
        (
            "high_quality_ibes_4k",
            "ibes_only",
            high_quality,
            [],
            "Test whether cleaner, smaller IBES supervision beats larger repetitive supervision for generalization.",
            "A smaller but higher-diversity, higher-consensus subset may preserve IBES skill without the public-benchmark regression seen in the 10k pure-IBES adapter.",
            [
                "Stricter filters: prior consensus present, at least 2 estimates, at least 2 analysts.",
                "Balanced across direction labels where feasible.",
                "Per-ticker concentration cap is stricter than the broader candidates.",
            ],
        ),
        (
            "balanced_ibes_10k",
            "ibes_only",
            balanced,
            [],
            "Test whether the 10k specialization came from class imbalance or concentration rather than raw scale.",
            "Balancing direction and magnitude while capping ticker/date concentration may recover public benchmark behavior without giving up IBES coverage.",
            [
                "Direction-balanced to roughly one third each for negative, neutral, positive.",
                "Magnitude quotas applied within direction when possible.",
            ],
        ),
        (
            "mixed_finance_10k",
            "mixed_finance",
            mixed_ibes,
            mixed_public,
            "Test whether explicit mixed supervision is needed to keep public finance sentiment boundaries while retaining IBES structured formatting.",
            "If pure IBES specialization is the problem, an 80/20 IBES/public mix should recover benchmark generalization better than another pure-IBES scale-up.",
            [
                "Exact mix target: 8,000 IBES structured rows + 2,000 public finance rows.",
                "Public target: 500 rows each from FiQA, Financial PhraseBank, TFNS, and NWGI.",
                *([f"Public-data load warning: {msg}" for msg in public_errors] if public_errors else []),
            ],
        ),
        (
            "diverse_ibes_10k",
            "ibes_only",
            diverse,
            [],
            "Test whether diversity alone, without public mixing, is enough to avoid over-specialization.",
            "Maximizing company/date/fiscal-period spread should reduce narrow template memorization if repetition was the main failure mode.",
            [
                "Lower per-ticker caps than the balanced candidate.",
                "Prioritizes company/date/fiscal-period coverage over exact label quotas.",
            ],
        ),
    ]

    summary = {"candidate_order": CANDIDATE_ORDER, "candidates": {}}
    for name, candidate_type, ibes_rows, public_subset, reason, risk_tested, notes in candidates:
        train_examples = build_ibes_examples(ibes_rows)
        for row in public_subset:
            train_examples.append(
                {
                    "instruction": row["instruction"],
                    "input": row["input"],
                    "output": row["output"],
                    "event_id": row["example_id"],
                    "source": row["source"],
                    "candidate_signature": row["candidate_signature"],
                }
            )
        card = summarize_candidate(
            name=name,
            candidate_type=candidate_type,
            reason=reason,
            risk_tested=risk_tested,
            train_examples=train_examples,
            ibes_rows=ibes_rows,
            public_rows=public_subset,
            notes=notes,
        )
        write_candidate(name, train_examples, card)
        summary["candidates"][name] = {
            "row_count": card["row_count"],
            "ibes_row_count": card["ibes_row_count"],
            "public_row_count": card["public_row_count"],
            "dataset_card": str((CANDIDATE_ROOT / name / "dataset_card.md").relative_to(REPO_ROOT)),
        }

    summary_path = OUTPUT_ROOT / "candidate_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_root": str(CANDIDATE_ROOT), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
