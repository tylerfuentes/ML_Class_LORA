#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs" / "evals"
SHAREABLE_DIR = REPO_ROOT / "docs" / "shareable-evals"


RUNS = {
    "wrds_baseline_holdout": {
        "path": "qwen36-27b-ibes-baseline-holdout/metrics.json",
        "kind": "wrds",
        "label": "WRDS baseline 1k holdout",
    },
    "wrds_10k_holdout": {
        "path": "qwen36-27b-ibes-10k-holdout/metrics.json",
        "kind": "wrds",
        "label": "WRDS 10k holdout with 10k adapter",
    },
    "wrds_1k_on_10k_holdout": {
        "path": "qwen36-27b-ibes-1k-on-10k-holdout/metrics.json",
        "kind": "wrds",
        "label": "WRDS 10k holdout with prior 1k adapter",
    },
    "fiqa_1k": {
        "path": "qwen36-27b-fiqa-nonthinking/metrics.json",
        "kind": "public",
        "label": "FIQA 1k adapter",
    },
    "fiqa_10k": {
        "path": "qwen36-27b-fiqa-10k-nonthinking/metrics.json",
        "kind": "public",
        "label": "FIQA 10k adapter",
    },
    "fpb_1k": {
        "path": "qwen36-27b-fpb-nonthinking/metrics.json",
        "kind": "public",
        "label": "FPB 1k adapter",
    },
    "fpb_10k": {
        "path": "qwen36-27b-fpb-10k-nonthinking/metrics.json",
        "kind": "public",
        "label": "FPB 10k adapter",
    },
    "tfns_1k": {
        "path": "qwen36-27b-tfns-1k-nonthinking/metrics.json",
        "kind": "public",
        "label": "TFNS 1k adapter",
    },
    "tfns_10k": {
        "path": "qwen36-27b-tfns-10k-nonthinking/metrics.json",
        "kind": "public",
        "label": "TFNS 10k adapter",
    },
    "nwgi_1k": {
        "path": "qwen36-27b-nwgi-1k-nonthinking/metrics.json",
        "kind": "public",
        "label": "NWGI 1k adapter",
    },
    "nwgi_10k": {
        "path": "qwen36-27b-nwgi-10k-nonthinking/metrics.json",
        "kind": "public",
        "label": "NWGI 10k adapter",
    },
}


PUBLIC_COMPARISONS = (
    ("fiqa", "fiqa_1k", "fiqa_10k"),
    ("fpb", "fpb_1k", "fpb_10k"),
    ("tfns", "tfns_1k", "tfns_10k"),
    ("nwgi", "nwgi_1k", "nwgi_10k"),
)


def load_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    task_name, task = next(iter(payload["tasks"].items()))
    mode_name = "nonthinking" if "nonthinking" in task else next(iter(task.keys()))
    mode = task[mode_name]
    return {
        "task_name": task_name,
        "prompt_mode": mode_name,
        "base": mode["base"],
        "adapter": mode["adapter"],
        "delta": mode["delta"],
        "verdict": mode["verdict"],
        "source_metrics_path": str(path.relative_to(REPO_ROOT)),
    }


def reduce_metrics(kind: str, metrics: dict) -> dict:
    base = metrics["base"]
    adapter = metrics["adapter"]
    reduced = {
        "task_name": metrics["task_name"],
        "prompt_mode": metrics["prompt_mode"],
        "verdict": metrics["verdict"],
        "base": {
            "accuracy": base["accuracy"],
            "macro_f1": base["macro_f1"],
            "parse_failure_rate": base["parse_failure_rate"],
        },
        "adapter": {
            "accuracy": adapter["accuracy"],
            "macro_f1": adapter["macro_f1"],
            "parse_failure_rate": adapter["parse_failure_rate"],
        },
        "delta": {
            "accuracy": adapter["accuracy"] - base["accuracy"],
            "macro_f1": adapter["macro_f1"] - base["macro_f1"],
            "parse_failure_rate": adapter["parse_failure_rate"] - base["parse_failure_rate"],
        },
        "source_metrics_path": metrics["source_metrics_path"],
    }
    if kind == "wrds":
        reduced["adapter"]["exact_json_match_rate"] = adapter.get("exact_json_match_rate")
        reduced["adapter"]["magnitude_bucket_accuracy"] = adapter.get("magnitude_bucket_accuracy")
        reduced["adapter"]["confusion"] = adapter.get("confusion")
        reduced["base"]["confusion"] = base.get("confusion")
    else:
        reduced["adapter"]["confusion"] = adapter.get("confusion")
    return reduced


def comparison_payload(name: str, one_k: dict, ten_k: dict) -> dict:
    return {
        "task": name,
        "accuracy_delta_10k_minus_1k": ten_k["adapter"]["accuracy"] - one_k["adapter"]["accuracy"],
        "macro_f1_delta_10k_minus_1k": ten_k["adapter"]["macro_f1"] - one_k["adapter"]["macro_f1"],
        "parse_failure_delta_10k_minus_1k": ten_k["adapter"]["parse_failure_rate"] - one_k["adapter"]["parse_failure_rate"],
    }


def build_summary(runs: dict[str, dict]) -> dict:
    public_runs = {name: payload for name, payload in runs.items() if RUNS[name]["kind"] == "public"}
    wrds_runs = {name: payload for name, payload in runs.items() if RUNS[name]["kind"] == "wrds"}
    comparisons = {
        task: comparison_payload(task, public_runs[one_k], public_runs[ten_k])
        for task, one_k, ten_k in PUBLIC_COMPARISONS
    }
    wrds_delta = {
        "accuracy_delta_10k_minus_1k": wrds_runs["wrds_10k_holdout"]["adapter"]["accuracy"] - wrds_runs["wrds_1k_on_10k_holdout"]["adapter"]["accuracy"],
        "macro_f1_delta_10k_minus_1k": wrds_runs["wrds_10k_holdout"]["adapter"]["macro_f1"] - wrds_runs["wrds_1k_on_10k_holdout"]["adapter"]["macro_f1"],
        "parse_failure_delta_10k_minus_1k": wrds_runs["wrds_10k_holdout"]["adapter"]["parse_failure_rate"] - wrds_runs["wrds_1k_on_10k_holdout"]["adapter"]["parse_failure_rate"],
        "exact_json_match_delta_10k_minus_1k": wrds_runs["wrds_10k_holdout"]["adapter"]["exact_json_match_rate"] - wrds_runs["wrds_1k_on_10k_holdout"]["adapter"]["exact_json_match_rate"],
        "magnitude_bucket_delta_10k_minus_1k": wrds_runs["wrds_10k_holdout"]["adapter"]["magnitude_bucket_accuracy"] - wrds_runs["wrds_1k_on_10k_holdout"]["adapter"]["magnitude_bucket_accuracy"],
    }
    return {
        "shareable_note": "Raw WRDS-derived predictions, examples, and ignored outputs stay local. This folder contains only code, docs, and aggregate metrics that are safe to review in git.",
        "wrds": wrds_runs,
        "public_benchmarks": public_runs,
        "comparisons": {
            "public_10k_minus_1k": comparisons,
            "wrds_10k_minus_1k": wrds_delta,
        },
    }


def write_markdown(summary: dict) -> str:
    lines = [
        "# Shareable Evaluation Summary",
        "",
        "This folder is the git-safe view of the recent evaluation work.",
        "",
        "- Included: aggregate metrics, confusion counts, comparison deltas, and links back to the ignored local metrics files.",
        "- Excluded: WRDS raw data, WRDS-derived predictions, raw holdout examples, adapters, checkpoints, caches, and logs.",
        "",
        "## WRDS holdout",
        "",
    ]
    for key in ("wrds_baseline_holdout", "wrds_10k_holdout", "wrds_1k_on_10k_holdout"):
        item = summary["wrds"][key]
        lines.extend(
            [
                f"### {RUNS[key]['label']}",
                "",
                f"- base accuracy: `{item['base']['accuracy']:.4f}`",
                f"- adapter accuracy: `{item['adapter']['accuracy']:.4f}`",
                f"- base macro F1: `{item['base']['macro_f1']:.4f}`",
                f"- adapter macro F1: `{item['adapter']['macro_f1']:.4f}`",
                f"- base parse failure rate: `{item['base']['parse_failure_rate']:.4f}`",
                f"- adapter parse failure rate: `{item['adapter']['parse_failure_rate']:.4f}`",
                f"- adapter exact JSON match: `{item['adapter']['exact_json_match_rate']:.4f}`",
                f"- adapter magnitude bucket accuracy: `{item['adapter']['magnitude_bucket_accuracy']:.4f}`",
                f"- source metrics file: `{item['source_metrics_path']}`",
                "",
            ]
        )

    wrds_delta = summary["comparisons"]["wrds_10k_minus_1k"]
    lines.extend(
        [
            "### WRDS 10k minus 1k",
            "",
            f"- accuracy delta: `{wrds_delta['accuracy_delta_10k_minus_1k']:+.4f}`",
            f"- macro F1 delta: `{wrds_delta['macro_f1_delta_10k_minus_1k']:+.4f}`",
            f"- parse failure delta: `{wrds_delta['parse_failure_delta_10k_minus_1k']:+.4f}`",
            f"- exact JSON match delta: `{wrds_delta['exact_json_match_delta_10k_minus_1k']:+.4f}`",
            f"- magnitude bucket delta: `{wrds_delta['magnitude_bucket_delta_10k_minus_1k']:+.4f}`",
            "",
            "## Public benchmarks",
            "",
        ]
    )

    for task, one_k, ten_k in PUBLIC_COMPARISONS:
        one = summary["public_benchmarks"][one_k]
        ten = summary["public_benchmarks"][ten_k]
        delta = summary["comparisons"]["public_10k_minus_1k"][task]
        lines.extend(
            [
                f"### {task.upper()}",
                "",
                f"- 1k adapter accuracy: `{one['adapter']['accuracy']:.4f}`",
                f"- 10k adapter accuracy: `{ten['adapter']['accuracy']:.4f}`",
                f"- 1k adapter macro F1: `{one['adapter']['macro_f1']:.4f}`",
                f"- 10k adapter macro F1: `{ten['adapter']['macro_f1']:.4f}`",
                f"- 10k minus 1k accuracy delta: `{delta['accuracy_delta_10k_minus_1k']:+.4f}`",
                f"- 10k minus 1k macro F1 delta: `{delta['macro_f1_delta_10k_minus_1k']:+.4f}`",
                f"- source metrics files: `{one['source_metrics_path']}`, `{ten['source_metrics_path']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Recommendation",
            "",
            "- Keep the non-thinking structured IBES path.",
            "- Do not start 50k yet.",
            "- Use CRSP daily returns plus the CRSP/Compustat link table as the next unblocker for market-reaction work.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    SHAREABLE_DIR.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict] = {}
    for name, config in RUNS.items():
        metrics = load_metrics(OUTPUTS_DIR / config["path"])
        runs[name] = reduce_metrics(config["kind"], metrics)

    summary = build_summary(runs)
    (SHAREABLE_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (SHAREABLE_DIR / "summary.md").write_text(write_markdown(summary) + "\n", encoding="utf-8")
    (SHAREABLE_DIR / "README.md").write_text(
        "# Shareable Eval Artifacts\n\n"
        "This directory contains git-safe summaries derived from local ignored eval outputs.\n\n"
        "- `summary.md`: short human-readable review packet for classmates.\n"
        "- `summary.json`: structured aggregate metrics and deltas.\n"
        "- Raw WRDS-derived predictions and example dumps remain under ignored `outputs/evals/`.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
