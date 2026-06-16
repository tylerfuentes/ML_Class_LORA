#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_SUMMARY = REPO_ROOT / "docs" / "shareable-evals" / "summary.json"
DEFAULT_HOLDOUT_FILE = REPO_ROOT / "data" / "processed" / "wrds_qwen_pipeline" / "jsonl" / "test.jsonl"
DEFAULT_DOC_SUMMARY = REPO_ROOT / "docs" / "overnight_run_summary.md"
PUBLIC_BENCHMARK_KEYS = ("fiqa", "fpb", "tfns", "nwgi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Babysit a WRDS Qwen training run through post-train eval and documentation.")
    parser.add_argument("--python", type=Path, default=REPO_ROOT / ".venv" / "bin" / "python")
    parser.add_argument("--train-pid", type=int, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--run-label", required=True, help="Stable run id, for example 20260613T202724Z.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--status-log", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--report-json-path", type=Path, required=True)
    parser.add_argument("--holdout-file", type=Path, default=DEFAULT_HOLDOUT_FILE)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--doc-summary-path", type=Path, default=DEFAULT_DOC_SUMMARY)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--benchmark-split-size", type=int, default=128)
    parser.add_argument("--local-files-only", action="store_true", default=False)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def log_line(path: Path, message: str) -> None:
    line = f"{utc_now()} {message}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def checkpoint_sort_key(path: Path) -> int:
    try:
        return int(path.name.split("-")[-1])
    except ValueError:
        return -1


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=checkpoint_sort_key)
    return checkpoints[-1] if checkpoints else None


def read_checkpoint_progress(output_dir: Path) -> dict[str, Any]:
    checkpoint = latest_checkpoint(output_dir)
    if checkpoint is None:
        return {}
    trainer_state = checkpoint / "trainer_state.json"
    if not trainer_state.exists():
        return {"checkpoint": str(checkpoint)}
    payload = load_json(trainer_state)
    tail = payload.get("log_history", [])[-1] if payload.get("log_history") else {}
    return {
        "checkpoint": str(checkpoint),
        "global_step": payload.get("global_step"),
        "epoch": payload.get("epoch"),
        "best_model_checkpoint": payload.get("best_model_checkpoint"),
        "latest_log": tail,
    }


def run_command(cmd: list[str], cwd: Path, status_log: Path) -> None:
    log_line(status_log, f"running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def run_eval_task(
    args: argparse.Namespace,
    adapter_path: Path,
    task_key: str,
    status_log: Path,
) -> tuple[Path, dict[str, Any]]:
    output_dir = args.eval_root / task_key
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
    if task_key == "wrds_holdout":
        cmd.extend(["--holdout-file", str(args.holdout_file)])
        metrics_key = "wrds_holdout"
    else:
        cmd.extend(["--benchmark", task_key])
        metrics_key = f"benchmark_{task_key}"
    run_command(cmd, REPO_ROOT, status_log)
    metrics = load_json(output_dir / "metrics.json")
    return output_dir, metrics["tasks"][metrics_key]["nonthinking"]


def load_baseline_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    summary = load_json(path)
    return {
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


def task_snapshot(task_metrics: dict[str, Any]) -> dict[str, Any]:
    base = task_metrics["base"]
    adapter = task_metrics["adapter"]
    snapshot = {
        "base_accuracy": base.get("accuracy"),
        "adapter_accuracy": adapter.get("accuracy"),
        "accuracy_delta": (
            adapter.get("accuracy") - base.get("accuracy")
            if base.get("accuracy") is not None and adapter.get("accuracy") is not None
            else None
        ),
        "base_macro_f1": base.get("macro_f1"),
        "adapter_macro_f1": adapter.get("macro_f1"),
        "macro_f1_delta": (
            adapter.get("macro_f1") - base.get("macro_f1")
            if base.get("macro_f1") is not None and adapter.get("macro_f1") is not None
            else None
        ),
        "adapter_parse_failure_rate": adapter.get("parse_failure_rate"),
    }
    if "exact_json_match_rate" in adapter:
        snapshot["adapter_exact_json_match_rate"] = adapter.get("exact_json_match_rate")
        snapshot["adapter_magnitude_bucket_accuracy"] = adapter.get("magnitude_bucket_accuracy")
        snapshot["adapter_event_type_accuracy"] = adapter.get("event_type_accuracy")
    return snapshot


def compare_public_baselines(task_key: str, task_metrics: dict[str, Any], baselines: dict[str, Any]) -> dict[str, Any]:
    if task_key not in baselines:
        return {}
    adapter = task_metrics["adapter"]
    current_macro_f1 = adapter.get("macro_f1")
    one_k = baselines[task_key]["1k"]
    ten_k = baselines[task_key]["10k"]
    return {
        "macro_f1_vs_1k": (
            current_macro_f1 - one_k.get("macro_f1")
            if current_macro_f1 is not None and one_k.get("macro_f1") is not None
            else None
        ),
        "macro_f1_vs_10k": (
            current_macro_f1 - ten_k.get("macro_f1")
            if current_macro_f1 is not None and ten_k.get("macro_f1") is not None
            else None
        ),
        "accuracy_vs_1k": (
            adapter.get("accuracy") - one_k.get("accuracy")
            if adapter.get("accuracy") is not None and one_k.get("accuracy") is not None
            else None
        ),
        "accuracy_vs_10k": (
            adapter.get("accuracy") - ten_k.get("accuracy")
            if adapter.get("accuracy") is not None and ten_k.get("accuracy") is not None
            else None
        ),
        "baseline_1k_macro_f1": one_k.get("macro_f1"),
        "baseline_10k_macro_f1": ten_k.get("macro_f1"),
    }


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def append_doc_summary(
    doc_summary_path: Path,
    report_path: Path,
    train_summary: dict[str, Any],
    eval_summary: dict[str, Any],
) -> None:
    marker = "## Final Training And Eval"
    if doc_summary_path.exists():
        existing = doc_summary_path.read_text(encoding="utf-8")
        if marker in existing:
            return
    lines = [
        "",
        marker,
        "",
        f"- final train summary: `{train_summary.get('output_dir')}/run_summary.json`",
        f"- final train loss: `{format_metric(train_summary.get('train_loss'))}`",
        f"- final eval loss: `{format_metric(train_summary.get('eval_loss'))}`",
        f"- peak reserved GPU memory GB: `{format_metric(train_summary.get('peak_gpu_mem_reserved_gb'))}`",
        f"- detailed eval report: `{report_path.relative_to(REPO_ROOT)}`",
        f"- WRDS holdout adapter accuracy: `{format_metric(eval_summary['wrds_holdout']['snapshot'].get('adapter_accuracy'))}`",
        f"- WRDS holdout exact JSON: `{format_metric(eval_summary['wrds_holdout']['snapshot'].get('adapter_exact_json_match_rate'))}`",
        f"- WRDS holdout magnitude bucket accuracy: `{format_metric(eval_summary['wrds_holdout']['snapshot'].get('adapter_magnitude_bucket_accuracy'))}`",
    ]
    with doc_summary_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def build_report(
    args: argparse.Namespace,
    train_summary: dict[str, Any],
    eval_summary: dict[str, Any],
) -> list[str]:
    lines = [
        "# WRDS Qwen Pipeline Run Report",
        "",
        f"Run label: `{args.run_label}`",
        f"Model: `{args.model_id}`",
        "",
        "## Training",
        "",
        f"- output dir: `{args.output_dir.relative_to(REPO_ROOT)}`",
        f"- log: `{args.train_log.relative_to(REPO_ROOT)}`",
        f"- train examples: `{train_summary.get('num_train_examples')}`",
        f"- eval examples: `{train_summary.get('num_eval_examples')}`",
        f"- test examples: `{train_summary.get('num_test_examples')}`",
        f"- max steps: `{train_summary.get('max_steps')}`",
        f"- effective batch size: `{train_summary.get('effective_batch_size')}`",
        f"- train runtime seconds: `{format_metric(train_summary.get('train_runtime'))}`",
        f"- train loss: `{format_metric(train_summary.get('train_loss'))}`",
        f"- final eval loss: `{format_metric(train_summary.get('eval_loss'))}`",
        f"- peak GPU allocated GB: `{format_metric(train_summary.get('peak_gpu_mem_allocated_gb'))}`",
        f"- peak GPU reserved GB: `{format_metric(train_summary.get('peak_gpu_mem_reserved_gb'))}`",
        "",
        "## Eval Overview",
        "",
        "| task | base acc | adapter acc | delta | base macro F1 | adapter macro F1 | delta | parse fail |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task_key in ["wrds_holdout", *PUBLIC_BENCHMARK_KEYS]:
        snapshot = eval_summary[task_key]["snapshot"]
        lines.append(
            f"| {task_key} | {format_metric(snapshot.get('base_accuracy'))} | {format_metric(snapshot.get('adapter_accuracy'))} | "
            f"{format_metric(snapshot.get('accuracy_delta'))} | {format_metric(snapshot.get('base_macro_f1'))} | "
            f"{format_metric(snapshot.get('adapter_macro_f1'))} | {format_metric(snapshot.get('macro_f1_delta'))} | "
            f"{format_metric(snapshot.get('adapter_parse_failure_rate'))} |"
        )
    wrds = eval_summary["wrds_holdout"]["snapshot"]
    lines.extend(
        [
            "",
            "## WRDS Holdout",
            "",
            f"- exact JSON match rate: `{format_metric(wrds.get('adapter_exact_json_match_rate'))}`",
            f"- magnitude bucket accuracy: `{format_metric(wrds.get('adapter_magnitude_bucket_accuracy'))}`",
            f"- event type accuracy: `{format_metric(wrds.get('adapter_event_type_accuracy'))}`",
            "",
            "## Public Benchmark Comparison Versus Prior Adapters",
            "",
            "| task | current macro F1 | delta vs 1k | delta vs 10k | current acc | delta vs 1k acc | delta vs 10k acc |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for task_key in PUBLIC_BENCHMARK_KEYS:
        snapshot = eval_summary[task_key]["snapshot"]
        baseline_delta = eval_summary[task_key]["baseline_delta"]
        lines.append(
            f"| {task_key} | {format_metric(snapshot.get('adapter_macro_f1'))} | {format_metric(baseline_delta.get('macro_f1_vs_1k'))} | "
            f"{format_metric(baseline_delta.get('macro_f1_vs_10k'))} | {format_metric(snapshot.get('adapter_accuracy'))} | "
            f"{format_metric(baseline_delta.get('accuracy_vs_1k'))} | {format_metric(baseline_delta.get('accuracy_vs_10k'))} |"
        )
    lines.extend(
        [
            "",
            "## Artifact Paths",
            "",
            f"- report json: `{args.report_json_path.relative_to(REPO_ROOT)}`",
            f"- status log: `{args.status_log.relative_to(REPO_ROOT)}`",
        ]
    )
    for task_key in ["wrds_holdout", *PUBLIC_BENCHMARK_KEYS]:
        lines.append(f"- {task_key}: `{eval_summary[task_key]['output_dir']}`")
    return lines


def resolve_adapter_path(output_dir: Path) -> Path:
    if (output_dir / "adapter_config.json").exists():
        return output_dir
    checkpoint = latest_checkpoint(output_dir)
    if checkpoint is not None and (checkpoint / "adapter_config.json").exists():
        return checkpoint
    raise FileNotFoundError(f"No adapter artifact found under {output_dir}")


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.train_log = args.train_log.resolve()
    args.status_log = args.status_log.resolve()
    args.eval_root = args.eval_root.resolve()
    args.report_path = args.report_path.resolve()
    args.report_json_path = args.report_json_path.resolve()
    args.holdout_file = args.holdout_file.resolve()
    args.baseline_summary = args.baseline_summary.resolve()
    args.doc_summary_path = args.doc_summary_path.resolve()

    last_logged_step: int | None = None
    log_line(args.status_log, f"babysitter started for run `{args.run_label}` on pid {args.train_pid}")
    while pid_is_running(args.train_pid):
        progress = read_checkpoint_progress(args.output_dir)
        step = progress.get("global_step")
        if step != last_logged_step:
            log_line(
                args.status_log,
                f"training still running; latest checkpoint step={step} epoch={progress.get('epoch')} "
                f"checkpoint={progress.get('checkpoint')}",
            )
            last_logged_step = step
        time.sleep(args.poll_seconds)

    log_line(args.status_log, f"training process {args.train_pid} exited")
    run_summary_path = args.output_dir / "run_summary.json"
    exit_code_path = args.output_dir / "train.exitcode"
    if exit_code_path.exists():
        exit_code = exit_code_path.read_text(encoding="utf-8").strip()
        log_line(args.status_log, f"train exit code file reports: {exit_code}")
    if not run_summary_path.exists():
        log_line(args.status_log, f"missing run summary at {run_summary_path}; leaving without evals")
        return 1

    train_summary = load_json(run_summary_path)
    adapter_path = resolve_adapter_path(args.output_dir)
    baselines = load_baseline_summary(args.baseline_summary)

    eval_summary: dict[str, Any] = {}
    for task_key in ["wrds_holdout", *PUBLIC_BENCHMARK_KEYS]:
        output_dir, task_metrics = run_eval_task(args, adapter_path, task_key, args.status_log)
        eval_summary[task_key] = {
            "output_dir": str(output_dir.relative_to(REPO_ROOT)),
            "snapshot": task_snapshot(task_metrics),
            "baseline_delta": compare_public_baselines(task_key, task_metrics, baselines),
        }
        log_line(args.status_log, f"completed eval task {task_key}")

    report_payload = {
        "run_label": args.run_label,
        "completed_utc": utc_now(),
        "model_id": args.model_id,
        "train_summary": train_summary,
        "eval_summary": eval_summary,
        "report_path": str(args.report_path),
    }
    write_json(args.report_json_path, report_payload)
    write_markdown(args.report_path, build_report(args, train_summary, eval_summary))
    append_doc_summary(args.doc_summary_path, args.report_path, train_summary, eval_summary)
    log_line(args.status_log, f"wrote report markdown to {args.report_path}")
    log_line(args.status_log, f"wrote report json to {args.report_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
