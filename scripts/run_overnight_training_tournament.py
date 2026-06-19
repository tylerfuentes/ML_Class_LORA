#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
CANDIDATE_ROOT = REPO_ROOT / "outputs" / "overnight_tournament" / "candidates"
RUN_ROOT = REPO_ROOT / "outputs" / "overnight_tournament" / "runs"
REPORT_PATH = REPO_ROOT / "docs" / "overnight-training-report.md"
BASELINE_SUMMARY = REPO_ROOT / "docs" / "shareable-evals" / "summary.json"
BASELINE_1K_ADAPTER = REPO_ROOT / "outputs" / "qwen36-27b-ibes-baseline"
BASELINE_10K_ADAPTER = REPO_ROOT / "outputs" / "qwen36-27b-ibes-10k-controlled" / "checkpoint-500"
COMMON_EVAL_FILE = REPO_ROOT / "data" / "processed" / "ibes_lora_baseline" / "jsonl" / "baseline_10k" / "eval.jsonl"
COMMON_HOLDOUT_FILE = REPO_ROOT / "data" / "processed" / "ibes_lora_baseline" / "jsonl" / "baseline_10k" / "holdout.jsonl"

CANDIDATE_ORDER = [
    "high_quality_ibes_4k",
    "balanced_ibes_10k",
    "mixed_finance_10k",
    "diverse_ibes_10k",
]
PUBLIC_BENCHMARK_KEYS = ["fiqa", "fpb", "tfns", "nwgi"]


@dataclass
class CandidateResult:
    name: str
    status: str
    run_dir: Path
    train_summary: dict[str, Any] | None
    eval_results: dict[str, dict[str, float]]
    public_wins_vs_1k: int
    public_wins_vs_10k: int
    public_regressions_vs_1k: int
    stop_reason: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a controlled overnight adapter tournament.")
    parser.add_argument("--python", type=Path, default=VENV_PYTHON)
    parser.add_argument("--candidate-root", type=Path, default=CANDIDATE_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--baseline-summary", type=Path, default=BASELINE_SUMMARY)
    parser.add_argument("--baseline-1k-adapter", type=Path, default=BASELINE_1K_ADAPTER)
    parser.add_argument("--baseline-10k-adapter", type=Path, default=BASELINE_10K_ADAPTER)
    parser.add_argument("--common-eval-file", type=Path, default=COMMON_EVAL_FILE)
    parser.add_argument("--common-holdout-file", type=Path, default=COMMON_HOLDOUT_FILE)
    parser.add_argument("--model-id", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--benchmark-split-size", type=int, default=128)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-candidates", type=int, default=0, help="0 means no explicit cap.")
    parser.add_argument("--start-from", default="")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--local-files-only", action="store_true", default=False)
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline_metrics(path: Path) -> dict[str, Any]:
    summary = load_json(path)
    return {
        "wrds_holdout": {
            "1k": summary["wrds"]["wrds_1k_on_10k_holdout"]["adapter"],
            "10k": summary["wrds"]["wrds_10k_holdout"]["adapter"],
        },
        "fiqa": {
            "1k": summary["public_benchmarks"]["fiqa_1k"]["adapter"],
            "10k": summary["public_benchmarks"]["fiqa_10k"]["adapter"],
        },
        "fpb": {
            "1k": summary["public_benchmarks"]["fpb_1k"]["adapter"],
            "10k": summary["public_benchmarks"]["fpb_10k"]["adapter"],
        },
        "tfns": {
            "1k": summary["public_benchmarks"]["tfns_1k"]["adapter"],
            "10k": summary["public_benchmarks"]["tfns_10k"]["adapter"],
        },
        "nwgi": {
            "1k": summary["public_benchmarks"]["nwgi_1k"]["adapter"],
            "10k": summary["public_benchmarks"]["nwgi_10k"]["adapter"],
        },
    }


def candidate_manifest(candidate_root: Path, name: str) -> dict[str, Any]:
    path = candidate_root / name / "manifest.json"
    require_file(path, f"{name} manifest")
    return load_json(path)


def run_command(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def train_candidate(args: argparse.Namespace, manifest: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    adapter_dir = run_dir / "adapter"
    cmd = [
        str(args.python),
        str(REPO_ROOT / "training" / "train_finance_lora.py"),
        "--model-id",
        args.model_id,
        "--train-file",
        str(REPO_ROOT / manifest["train_file"]),
        "--eval-file",
        str(args.common_eval_file),
        "--test-file",
        str(args.common_holdout_file),
        "--output-dir",
        str(adapter_dir),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--per-device-train-batch-size",
        str(args.per_device_train_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--save-steps",
        str(args.save_steps),
        "--logging-steps",
        str(args.logging_steps),
        "--max-total-examples",
        "0",
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.dry_run:
        return {"command": cmd}
    run_command(cmd, REPO_ROOT)
    return load_json(adapter_dir / "run_summary.json")


def evaluate_candidate_task(
    args: argparse.Namespace,
    adapter_path: Path,
    output_dir: Path,
    task_name: str,
    benchmark: str | None = None,
    holdout_file: Path | None = None,
) -> dict[str, float]:
    cmd = [
        str(args.python),
        str(REPO_ROOT / "eval" / "evaluate_base_vs_adapter.py"),
        "--model-id",
        args.model_id,
        "--adapter-path",
        str(adapter_path),
        "--output-dir",
        str(output_dir),
        "--qwen-thinking-mode",
        "nonthinking",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--benchmark-split-size",
        str(args.benchmark_split_size),
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")
    if benchmark:
        cmd.extend(["--benchmark", benchmark])
    if holdout_file:
        cmd.extend(["--holdout-file", str(holdout_file)])
    if args.dry_run:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    run_command(cmd, REPO_ROOT)
    metrics = load_json(output_dir / "metrics.json")
    task_metrics = metrics["tasks"][task_name]["nonthinking"]["adapter"]
    keys = ["accuracy", "macro_f1", "parse_failure_rate", "exact_json_match_rate", "magnitude_bucket_accuracy"]
    return {key: task_metrics.get(key) for key in keys if key in task_metrics}


def evaluate_candidate(args: argparse.Namespace, run_dir: Path) -> dict[str, dict[str, float]]:
    adapter_path = run_dir / "adapter"
    results = {
        "wrds_holdout": evaluate_candidate_task(
            args,
            adapter_path=adapter_path,
            output_dir=run_dir / "eval_wrds_holdout",
            task_name="wrds_holdout",
            holdout_file=args.common_holdout_file,
        )
    }
    for benchmark in PUBLIC_BENCHMARK_KEYS:
        results[benchmark] = evaluate_candidate_task(
            args,
            adapter_path=adapter_path,
            output_dir=run_dir / f"eval_{benchmark}",
            task_name=f"benchmark_{benchmark}",
            benchmark=benchmark,
        )
    return results


def metric_win(candidate_value: float | None, baseline_value: float | None) -> bool:
    if candidate_value is None or baseline_value is None:
        return False
    return candidate_value > baseline_value


def ibes_preserved(candidate_metrics: dict[str, float], baseline_metrics: dict[str, Any]) -> bool:
    accuracy = candidate_metrics.get("accuracy", 0.0)
    macro_f1 = candidate_metrics.get("macro_f1", 0.0)
    exact_json = candidate_metrics.get("exact_json_match_rate", 0.0)
    magnitude = candidate_metrics.get("magnitude_bucket_accuracy", 0.0)
    return (
        accuracy >= baseline_metrics.get("accuracy", 0.0)
        and macro_f1 >= baseline_metrics.get("macro_f1", 0.0)
        and exact_json >= baseline_metrics.get("exact_json_match_rate", 0.0) - 0.01
        and magnitude >= baseline_metrics.get("magnitude_bucket_accuracy", 0.0) - 0.01
    )


def compare_against_baselines(candidate_eval: dict[str, dict[str, float]], baselines: dict[str, Any]) -> tuple[int, int, int]:
    wins_vs_1k = 0
    wins_vs_10k = 0
    regressions_vs_1k = 0
    for benchmark in PUBLIC_BENCHMARK_KEYS:
        candidate = candidate_eval[benchmark]
        one_k = baselines[benchmark]["1k"]
        ten_k = baselines[benchmark]["10k"]
        if metric_win(candidate.get("macro_f1"), one_k.get("macro_f1")):
            wins_vs_1k += 1
        elif candidate.get("macro_f1", 0.0) < one_k.get("macro_f1", 0.0):
            regressions_vs_1k += 1
        if metric_win(candidate.get("macro_f1"), ten_k.get("macro_f1")):
            wins_vs_10k += 1
    return wins_vs_1k, wins_vs_10k, regressions_vs_1k


def load_task_adapter_metrics(metrics_path: Path, task_name: str) -> dict[str, float]:
    metrics = load_json(metrics_path)
    task_metrics = metrics["tasks"][task_name]["nonthinking"]["adapter"]
    keys = ["accuracy", "macro_f1", "parse_failure_rate", "exact_json_match_rate", "magnitude_bucket_accuracy"]
    return {key: task_metrics.get(key) for key in keys if key in task_metrics}


def existing_candidate_result(run_root: Path, candidate_name: str, baselines: dict[str, Any]) -> CandidateResult | None:
    run_dir = run_root / candidate_name
    adapter_summary = run_dir / "adapter" / "run_summary.json"
    if not adapter_summary.exists():
        return None
    expected = {
        "wrds_holdout": run_dir / "eval_wrds_holdout" / "metrics.json",
        "fiqa": run_dir / "eval_fiqa" / "metrics.json",
        "fpb": run_dir / "eval_fpb" / "metrics.json",
        "tfns": run_dir / "eval_tfns" / "metrics.json",
        "nwgi": run_dir / "eval_nwgi" / "metrics.json",
    }
    if not all(path.exists() for path in expected.values()):
        return None
    eval_results = {
        "wrds_holdout": load_task_adapter_metrics(expected["wrds_holdout"], "wrds_holdout"),
        "fiqa": load_task_adapter_metrics(expected["fiqa"], "benchmark_fiqa"),
        "fpb": load_task_adapter_metrics(expected["fpb"], "benchmark_fpb"),
        "tfns": load_task_adapter_metrics(expected["tfns"], "benchmark_tfns"),
        "nwgi": load_task_adapter_metrics(expected["nwgi"], "benchmark_nwgi"),
    }
    wins_vs_1k, wins_vs_10k, regressions_vs_1k = compare_against_baselines(eval_results, baselines)
    return CandidateResult(
        name=candidate_name,
        status="completed",
        run_dir=run_dir,
        train_summary=load_json(adapter_summary),
        eval_results=eval_results,
        public_wins_vs_1k=wins_vs_1k,
        public_wins_vs_10k=wins_vs_10k,
        public_regressions_vs_1k=regressions_vs_1k,
        stop_reason=None,
    )


def write_report(
    path: Path,
    results: list[CandidateResult],
    baselines: dict[str, Any],
    stop_reason: str | None,
) -> None:
    lines = [
        "# Overnight Training Report",
        "",
        "This report tracks the controlled overnight tournament for finding a stronger general finance/event adapter without blind pure-IBES scaling.",
        "",
        "## Baseline roles",
        "",
        "- `1k adapter`: best general finance adapter so far",
        "- `10k adapter`: best narrow IBES structured JSON specialist",
        "- pure `50k` IBES scaling: not started",
        "- no alpha or trading claim until market-reaction data and backtests exist",
        "",
        "## Tournament results",
        "",
        "| candidate | status | public wins vs 1k | public wins vs 10k | public regressions vs 1k | IBES accuracy | IBES macro F1 | IBES exact JSON |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        ibes = result.eval_results.get("wrds_holdout", {})
        lines.append(
            f"| {result.name} | {result.status} | {result.public_wins_vs_1k} | {result.public_wins_vs_10k} | "
            f"{result.public_regressions_vs_1k} | {ibes.get('accuracy', 0.0):.4f} | {ibes.get('macro_f1', 0.0):.4f} | "
            f"{ibes.get('exact_json_match_rate', 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Baseline comparison anchors",
            "",
            f"- 1k IBES exact JSON: `{baselines['wrds_holdout']['1k'].get('exact_json_match_rate', 0.0):.4f}`",
            f"- 10k IBES exact JSON: `{baselines['wrds_holdout']['10k'].get('exact_json_match_rate', 0.0):.4f}`",
            "",
        ]
    )
    if stop_reason:
        lines.extend(["## Stop rule", "", f"- {stop_reason}", ""])
    lines.extend(
        [
            "## Decision summary",
            "",
            "- best model for structured IBES: pending until the tournament finishes",
            "- best model for public finance generalization: pending until the tournament finishes",
            "- best candidate for the next event-reasoning stage: pending until the tournament finishes",
            "- more pure IBES scaling justified: only if a diversity/mix candidate still fails cleanly for reasons unrelated to specialization",
            "- mixed-finance training justified: yes if it improves at least three public benchmarks without giving up IBES performance",
            "- missing data for market-reaction/backtesting: formal CRSP/Compustat link table is still missing",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def derive_stop_reason_from_results(results: list[CandidateResult], baselines: dict[str, Any]) -> str | None:
    if not results:
        return None
    for item in results:
        if item.eval_results and item.public_wins_vs_1k >= 3 and ibes_preserved(
            item.eval_results["wrds_holdout"], baselines["wrds_holdout"]["1k"]
        ):
            return (
                f"Stopped after {item.name}: it beat the 1k adapter on at least three public benchmarks "
                "while preserving IBES performance."
            )
    if len(results) >= 2 and all(item.public_regressions_vs_1k >= 3 for item in results[:2]):
        return "Stopped after the first two candidates because both clearly regressed public benchmarks versus the 1k adapter."
    return None


def main() -> None:
    args = parse_args()
    require_file(args.python, "tournament python")
    require_file(args.baseline_summary, "baseline summary")
    require_file(args.baseline_1k_adapter / "adapter_config.json", "1k adapter")
    require_file(args.baseline_10k_adapter / "adapter_config.json", "10k adapter")
    require_file(args.common_eval_file, "common eval file")
    require_file(args.common_holdout_file, "common holdout file")

    baselines = load_baseline_metrics(args.baseline_summary)
    args.run_root.mkdir(parents=True, exist_ok=True)

    candidate_names = list(CANDIDATE_ORDER)
    if args.start_from:
        if args.start_from not in candidate_names:
            raise ValueError(f"Unknown candidate {args.start_from}. Choices: {candidate_names}")
        start_index = candidate_names.index(args.start_from)
        candidate_names = candidate_names[start_index:]
    else:
        start_index = 0
    if args.max_candidates > 0:
        candidate_names = candidate_names[: args.max_candidates]

    results: list[CandidateResult] = []
    for prior_name in CANDIDATE_ORDER[:start_index]:
        prior = existing_candidate_result(args.run_root, prior_name, baselines)
        if prior is not None:
            results.append(prior)
    stop_reason = derive_stop_reason_from_results(results, baselines)

    if stop_reason and candidate_names:
        write_report(args.report_path, results, baselines, stop_reason)
        print(
            json.dumps(
                {
                    "run_root": str(args.run_root),
                    "report_path": str(args.report_path),
                    "candidates_run": [item.name for item in results],
                    "stop_reason": stop_reason,
                    "dry_run": args.dry_run,
                },
                indent=2,
            )
        )
        return

    for candidate_name in candidate_names:
        manifest = candidate_manifest(args.candidate_root, candidate_name)
        run_dir = args.run_root / candidate_name
        run_dir.mkdir(parents=True, exist_ok=True)
        train_summary = train_candidate(args, manifest, run_dir)
        eval_results = evaluate_candidate(args, run_dir) if not args.dry_run else {}
        
        if not args.dry_run:
            try:
                doc_cmd = [
                    str(args.python),
                    str(REPO_ROOT / "scripts" / "document_tournament_outputs.py"),
                    "--run-dir", str(run_dir),
                    "--candidate-dir", str(args.candidate_root / candidate_name)
                ]
                run_command(doc_cmd, REPO_ROOT)
            except Exception as e:
                print(f"Warning: documentation script failed for {candidate_name}: {e}")

        wins_vs_1k, wins_vs_10k, regressions_vs_1k = compare_against_baselines(eval_results, baselines) if eval_results else (0, 0, 0)

        result = CandidateResult(
            name=candidate_name,
            status="completed" if not args.dry_run else "planned",
            run_dir=run_dir,
            train_summary=train_summary if not args.dry_run else None,
            eval_results=eval_results,
            public_wins_vs_1k=wins_vs_1k,
            public_wins_vs_10k=wins_vs_10k,
            public_regressions_vs_1k=regressions_vs_1k,
            stop_reason=None,
        )
        results.append(result)

        if not args.dry_run:
            if wins_vs_1k >= 3 and ibes_preserved(eval_results["wrds_holdout"], baselines["wrds_holdout"]["1k"]):
                stop_reason = (
                    f"Stopped after {candidate_name}: it beat the 1k adapter on at least three public benchmarks "
                    "while preserving IBES performance."
                )
            elif len(results) >= 2 and all(item.public_regressions_vs_1k >= 3 for item in results[:2]):
                stop_reason = (
                    "Stopped after the first two candidates because both clearly regressed public benchmarks versus the 1k adapter."
                )
            elif eval_results["wrds_holdout"].get("parse_failure_rate", 0.0) > 0.05:
                stop_reason = f"Stopped after {candidate_name}: output format stability regressed on the IBES holdout."

        write_report(args.report_path, results, baselines, stop_reason)
        if stop_reason:
            break

    print(
        json.dumps(
            {
                "run_root": str(args.run_root),
                "report_path": str(args.report_path),
                "candidates_run": [item.name for item in results],
                "stop_reason": stop_reason,
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
