#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_BENCHMARKS = ("fiqa", "fpb", "tfns", "nwgi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for the old WRDS run, then launch the controlled run and final evals.")
    parser.add_argument("--python", type=Path, default=REPO_ROOT / ".venv" / "bin" / "python")
    parser.add_argument("--old-train-pid", type=int, required=True)
    parser.add_argument("--old-babysitter-pid", type=int, default=0)
    parser.add_argument("--old-output-dir", type=Path, required=True)
    parser.add_argument("--old-train-log", type=Path, required=True)
    parser.add_argument("--old-run-name", default="qwen36-27b-wrds-100k-old-eval-heavy")
    parser.add_argument("--new-run-name", default="qwen36-27b-wrds-100k-controlled")
    parser.add_argument("--train-file", type=Path, default=REPO_ROOT / "data" / "processed" / "wrds_qwen_pipeline" / "jsonl" / "train.jsonl")
    parser.add_argument("--train-eval-file", type=Path, default=REPO_ROOT / "data" / "processed" / "wrds_qwen_pipeline" / "jsonl" / "train_eval.jsonl")
    parser.add_argument("--holdout-file", type=Path, default=REPO_ROOT / "data" / "processed" / "wrds_qwen_pipeline" / "jsonl" / "test.jsonl")
    parser.add_argument("--baseline-summary", type=Path, default=REPO_ROOT / "docs" / "shareable-evals" / "summary.json")
    parser.add_argument("--status-log", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--model-id", default="Qwen/Qwen3.6-27B")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def log_line(path: Path, message: str) -> None:
    line = f"{utc_now()} {message}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def run_command(cmd: list[str], cwd: Path, status_log: Path, log_path: Path | None = None) -> None:
    log_line(status_log, f"running: {' '.join(cmd)}")
    if log_path is None:
        subprocess.run(cmd, cwd=cwd, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=cwd, check=True, stdout=handle, stderr=subprocess.STDOUT)


def stop_process(pid: int, status_log: Path, label: str) -> None:
    if pid <= 0 or not pid_is_running(pid):
        return
    log_line(status_log, f"stopping {label} pid {pid}")
    os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if not pid_is_running(pid):
            log_line(status_log, f"{label} pid {pid} stopped")
            return
        time.sleep(1)
    if pid_is_running(pid):
        log_line(status_log, f"{label} pid {pid} did not exit after SIGTERM; sending SIGKILL")
        os.kill(pid, signal.SIGKILL)


def latest_old_checkpoint_state(output_dir: Path) -> dict[str, Any]:
    checkpoint = latest_checkpoint(output_dir)
    if checkpoint is None:
        return {}
    state_path = checkpoint / "trainer_state.json"
    payload = load_json(state_path) if state_path.exists() else {}
    return {
        "checkpoint_path": str(checkpoint),
        "adapter_config_exists": (checkpoint / "adapter_config.json").exists(),
        "global_step": payload.get("global_step"),
        "epoch": payload.get("epoch"),
        "latest_log": payload.get("log_history", [])[-1] if payload.get("log_history") else {},
    }


def write_old_run_marker(args: argparse.Namespace, status_log: Path) -> dict[str, Any]:
    marker_path = args.old_output_dir / "old_config_not_final.json"
    old_state = latest_old_checkpoint_state(args.old_output_dir)
    payload = {
        "tagged_at_utc": utc_now(),
        "run_name": args.old_run_name,
        "status": "old_config_eval_heavy_not_final",
        "note": "This adapter/checkpoint was produced with the obsolete eval-heavy training config and is preserved only as a historical comparison artifact.",
        "output_dir": str(args.old_output_dir),
        "train_log": str(args.old_train_log),
        "latest_checkpoint": old_state,
    }
    write_json(marker_path, payload)
    log_line(status_log, f"wrote old-run marker to {marker_path}")
    return payload


def wait_for_old_run(args: argparse.Namespace) -> dict[str, Any]:
    status_log = args.status_log
    stop_process(args.old_babysitter_pid, status_log, "old babysitter")
    old_state = latest_old_checkpoint_state(args.old_output_dir)
    checkpoint_path = old_state.get("checkpoint_path")
    step = old_state.get("global_step")
    latest_log = old_state.get("latest_log") or {}
    log_line(
        status_log,
        f"old run state: step={step} checkpoint={checkpoint_path} adapter_checkpoint_valid={old_state.get('adapter_config_exists')} latest_log={latest_log}",
    )
    log_line(
        status_log,
        "old run is within the final segment after checkpoint-1200; allowing it to finish rather than killing a nearly-complete eval-heavy job",
    )
    while pid_is_running(args.old_train_pid):
        time.sleep(args.poll_seconds)
    log_line(status_log, f"old train pid {args.old_train_pid} exited")
    return write_old_run_marker(args, status_log)


def paths_for_new_run(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "output_dir": REPO_ROOT / "outputs" / "wrds_qwen_pipeline" / "train" / args.new_run_name,
        "train_log": REPO_ROOT / "logs" / "wrds_qwen_pipeline" / f"{args.new_run_name}.log",
        "eval_root": REPO_ROOT / "outputs" / "evals" / "wrds_qwen_pipeline" / args.new_run_name,
        "report_md": REPO_ROOT / "docs" / "reports" / f"{args.new_run_name}.md",
        "report_json": REPO_ROOT / "data" / "processed" / "wrds_qwen_pipeline" / "reports" / f"{args.new_run_name}.json",
    }


def ensure_absent_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing output dir: {path}")
    path.mkdir(parents=True, exist_ok=True)


def launch_new_train(args: argparse.Namespace, paths: dict[str, Path]) -> tuple[subprocess.Popen[str], list[str]]:
    ensure_absent_output_dir(paths["output_dir"])
    cmd = [
        str(args.python),
        str(REPO_ROOT / "training" / "train_finance_lora_unsloth.py"),
        "--model-id",
        args.model_id,
        "--train-file",
        str(args.train_file),
        "--eval-file",
        str(args.train_eval_file),
        "--test-file",
        str(args.holdout_file),
        "--output-dir",
        str(paths["output_dir"]),
        "--epochs",
        "0.2",
        "--lr",
        "0.0002",
        "--per-device-train-batch-size",
        "1",
        "--gradient-accumulation-steps",
        "16",
        "--eval-steps",
        "500",
        "--save-steps",
        "500",
        "--logging-steps",
        "25",
        "--max-seq-length",
        "2048",
        "--gpu-memory-utilization",
        "0.9",
        "--max-total-examples",
        "0",
        "--local-files-only",
    ]
    paths["train_log"].parent.mkdir(parents=True, exist_ok=True)
    handle = paths["train_log"].open("w", encoding="utf-8")
    process = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    log_line(args.status_log, f"launched new controlled run pid {process.pid}")
    return process, cmd


def resolve_adapter_path(output_dir: Path) -> Path:
    if (output_dir / "adapter_config.json").exists():
        return output_dir
    checkpoint = latest_checkpoint(output_dir)
    if checkpoint is not None and (checkpoint / "adapter_config.json").exists():
        return checkpoint
    raise RuntimeError(f"No adapter artifact found under {output_dir}")


def run_eval_task(
    args: argparse.Namespace,
    adapter_path: Path,
    output_dir: Path,
    *,
    benchmark: str | None = None,
    holdout_file: Path | None = None,
) -> dict[str, Any]:
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
        "96",
        "--benchmark-split-size",
        "128",
        "--local-files-only",
    ]
    if benchmark is not None:
        cmd.extend(["--benchmark", benchmark])
        log_path = REPO_ROOT / "logs" / "wrds_qwen_pipeline" / f"{args.new_run_name}_{benchmark}.eval.log"
        run_command(cmd, REPO_ROOT, args.status_log, log_path)
        metrics = load_json(output_dir / "metrics.json")
        return metrics["tasks"][f"benchmark_{benchmark}"]["nonthinking"]
    if holdout_file is not None:
        cmd.extend(["--holdout-file", str(holdout_file)])
        log_path = REPO_ROOT / "logs" / "wrds_qwen_pipeline" / f"{args.new_run_name}_wrds_holdout.eval.log"
        run_command(cmd, REPO_ROOT, args.status_log, log_path)
        metrics = load_json(output_dir / "metrics.json")
        return metrics["tasks"]["wrds_holdout"]["nonthinking"]
    raise ValueError("benchmark or holdout_file required")


def load_baselines(path: Path) -> dict[str, Any]:
    summary = load_json(path)
    return {
        "wrds": {
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


def maybe_old_adapter_reference(old_marker: dict[str, Any]) -> str | None:
    latest = old_marker.get("latest_checkpoint", {})
    checkpoint_path = latest.get("checkpoint_path")
    if checkpoint_path and latest.get("adapter_config_exists"):
        return checkpoint_path
    return None


def metric(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_report(
    args: argparse.Namespace,
    train_summary: dict[str, Any],
    old_marker: dict[str, Any],
    baselines: dict[str, Any],
    evals: dict[str, dict[str, Any]],
    paths: dict[str, Path],
) -> tuple[list[str], dict[str, Any]]:
    lines = [
        "# WRDS Controlled Run Report",
        "",
        f"- old run: `{args.old_run_name}`",
        f"- new run: `{args.new_run_name}`",
        "",
        "## Training Contract",
        "",
        f"- train examples: `{train_summary.get('num_train_examples')}`",
        "- trainer eval: `train_eval.jsonl` with `1000` examples",
        "- eval_steps: `500`",
        "- save_steps: `500`",
        "- logging_steps: `25`",
        "- no thinking mode",
        "- backend: `unsloth`",
        "",
        "## New Train Summary",
        "",
        f"- output dir: `{paths['output_dir'].relative_to(REPO_ROOT)}`",
        f"- train log: `{paths['train_log'].relative_to(REPO_ROOT)}`",
        f"- train loss: `{metric(train_summary.get('train_loss'))}`",
        f"- trainer eval loss: `{metric(train_summary.get('eval_loss'))}`",
        f"- train runtime seconds: `{metric(train_summary.get('train_runtime'))}`",
        f"- peak reserved GPU GB: `{metric(train_summary.get('peak_gpu_mem_reserved_gb'))}`",
        "",
        "## WRDS Holdout",
        "",
    ]
    wrds = evals["wrds_holdout"]["adapter"]
    base_wrds = evals["wrds_holdout"]["base"]
    lines.extend(
        [
            f"- base accuracy: `{metric(base_wrds.get('accuracy'))}`",
            f"- adapter accuracy: `{metric(wrds.get('accuracy'))}`",
            f"- adapter macro F1: `{metric(wrds.get('macro_f1'))}`",
            f"- adapter exact JSON: `{metric(wrds.get('exact_json_match_rate'))}`",
            f"- adapter magnitude bucket accuracy: `{metric(wrds.get('magnitude_bucket_accuracy'))}`",
            f"- vs 1k WRDS accuracy: `{metric(wrds.get('accuracy') - baselines['wrds']['1k'].get('accuracy'))}`",
            f"- vs 10k WRDS accuracy: `{metric(wrds.get('accuracy') - baselines['wrds']['10k'].get('accuracy'))}`",
            "",
            "## Public Benchmarks",
            "",
            "| task | base acc | new acc | new macro F1 | delta vs 1k macro F1 | delta vs 10k macro F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for benchmark in PUBLIC_BENCHMARKS:
        base_metrics = evals[benchmark]["base"]
        adapter_metrics = evals[benchmark]["adapter"]
        lines.append(
            f"| {benchmark} | {metric(base_metrics.get('accuracy'))} | {metric(adapter_metrics.get('accuracy'))} | "
            f"{metric(adapter_metrics.get('macro_f1'))} | "
            f"{metric(adapter_metrics.get('macro_f1') - baselines[benchmark]['1k'].get('macro_f1'))} | "
            f"{metric(adapter_metrics.get('macro_f1') - baselines[benchmark]['10k'].get('macro_f1'))} |"
        )
    lines.extend(
        [
            "",
            "## Old Config Reference",
            "",
            f"- marker: `{(args.old_output_dir / 'old_config_not_final.json').relative_to(REPO_ROOT)}`",
            f"- preserved checkpoint: `{old_marker.get('latest_checkpoint', {}).get('checkpoint_path', 'n/a')}`",
            "- old-config artifact is retained for historical comparison only and should not be treated as the final WRDS 100k result",
            "",
            "## Artifact Paths",
            "",
            f"- eval root: `{paths['eval_root'].relative_to(REPO_ROOT)}`",
            f"- report json: `{paths['report_json'].relative_to(REPO_ROOT)}`",
        ]
    )
    report_json = {
        "created_at_utc": utc_now(),
        "old_run_name": args.old_run_name,
        "new_run_name": args.new_run_name,
        "train_summary": train_summary,
        "old_marker": old_marker,
        "baselines": baselines,
        "evals": evals,
        "old_config_reference_adapter": maybe_old_adapter_reference(old_marker),
    }
    return lines, report_json


def main() -> int:
    args = parse_args()
    args.old_output_dir = args.old_output_dir.resolve()
    args.old_train_log = args.old_train_log.resolve()
    args.status_log = args.status_log.resolve()
    args.train_file = args.train_file.resolve()
    args.train_eval_file = args.train_eval_file.resolve()
    args.holdout_file = args.holdout_file.resolve()
    args.baseline_summary = args.baseline_summary.resolve()

    log_line(args.status_log, "controlled orchestrator started")
    old_marker = wait_for_old_run(args)
    paths = paths_for_new_run(args)
    process, cmd = launch_new_train(args, paths)
    log_line(args.status_log, f"new controlled train command: {' '.join(cmd)}")
    returncode = process.wait()
    if returncode != 0:
        log_line(args.status_log, f"new controlled run failed with return code {returncode}")
        return returncode

    log_line(args.status_log, f"new controlled run pid {process.pid} exited successfully")
    adapter_path = resolve_adapter_path(paths["output_dir"])
    train_summary = load_json(paths["output_dir"] / "run_summary.json")

    evals: dict[str, dict[str, Any]] = {}
    evals["wrds_holdout"] = run_eval_task(
        args,
        adapter_path,
        paths["eval_root"] / "wrds_holdout",
        holdout_file=args.holdout_file,
    )
    for benchmark in PUBLIC_BENCHMARKS:
        evals[benchmark] = run_eval_task(
            args,
            adapter_path,
            paths["eval_root"] / benchmark,
            benchmark=benchmark,
        )

    baselines = load_baselines(args.baseline_summary)
    report_lines, report_json = build_report(args, train_summary, old_marker, baselines, evals, paths)
    paths["report_md"].parent.mkdir(parents=True, exist_ok=True)
    paths["report_md"].write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    write_json(paths["report_json"], report_json)
    log_line(args.status_log, f"wrote report markdown to {paths['report_md']}")
    log_line(args.status_log, f"wrote report json to {paths['report_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
