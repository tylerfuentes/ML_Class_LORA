#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch detached WRDS Unsloth train and babysitter processes.")
    parser.add_argument("--run-name", default="qwen36-27b-wrds-500k-unsloth-gb10")
    return parser.parse_args()


def ensure_not_running(pid_path: Path, label: str) -> None:
    if not pid_path.exists():
        return
    pid_text = pid_path.read_text(encoding="utf-8").strip()
    if not pid_text:
        return
    try:
        pid = int(pid_text)
    except ValueError:
        return
    probe = subprocess.run(["kill", "-0", str(pid)], check=False, capture_output=True, text=True)
    if probe.returncode == 0:
        raise RuntimeError(f"Refusing to start: existing {label} pid {pid} is still running")


def main() -> int:
    args = parse_args()
    run_name = args.run_name
    output_dir = REPO_ROOT / "outputs" / "wrds_qwen_pipeline" / "train" / run_name
    train_log = REPO_ROOT / "logs" / "wrds_qwen_pipeline" / f"{run_name}.log"
    train_stdout = REPO_ROOT / "logs" / "wrds_qwen_pipeline" / f"{run_name}.stdout"
    status_log = REPO_ROOT / "logs" / "wrds_qwen_pipeline" / f"babysit_{run_name}.log"
    babysit_stdout = REPO_ROOT / "logs" / "wrds_qwen_pipeline" / f"babysit_{run_name}.stdout"
    eval_root = REPO_ROOT / "outputs" / "evals" / "wrds_qwen_pipeline" / run_name
    report_path = REPO_ROOT / "docs" / "reports" / f"{run_name}.md"
    report_json_path = REPO_ROOT / "data" / "processed" / "wrds_qwen_pipeline" / "reports" / f"{run_name}.json"
    doc_summary_path = REPO_ROOT / "docs" / "overnight_run_summary.md"
    train_pid_path = output_dir / "train_wrapper.pid"
    babysit_pid_path = output_dir / "babysitter.pid"
    exit_code_path = output_dir / "train.exitcode"
    manifest_path = output_dir / "launch_manifest.json"

    train_log.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)

    ensure_not_running(train_pid_path, "train wrapper")
    ensure_not_running(babysit_pid_path, "babysitter")

    for path in (train_stdout, babysit_stdout, status_log, exit_code_path):
        path.unlink(missing_ok=True)

    with train_stdout.open("w", encoding="utf-8") as train_stdout_handle:
        train_process = subprocess.Popen(
            [str(REPO_ROOT / "scripts" / "launch_wrds_unsloth_gb10.sh"), run_name],
            cwd=REPO_ROOT,
            stdout=train_stdout_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )

    train_pid_path.write_text(f"{train_process.pid}\n", encoding="utf-8")

    with babysit_stdout.open("w", encoding="utf-8") as babysit_stdout_handle:
        babysit_process = subprocess.Popen(
            [
                str(REPO_ROOT / ".venv" / "bin" / "python"),
                "-u",
                str(REPO_ROOT / "scripts" / "babysit_wrds_qwen_run.py"),
                "--train-pid",
                str(train_process.pid),
                "--run-label",
                run_name,
                "--output-dir",
                str(output_dir),
                "--train-log",
                str(train_log),
                "--status-log",
                str(status_log),
                "--eval-root",
                str(eval_root),
                "--report-path",
                str(report_path),
                "--report-json-path",
                str(report_json_path),
                "--doc-summary-path",
                str(doc_summary_path),
                "--local-files-only",
            ],
            cwd=REPO_ROOT,
            stdout=babysit_stdout_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )

    babysit_pid_path.write_text(f"{babysit_process.pid}\n", encoding="utf-8")
    manifest = {
        "run_name": run_name,
        "train_pid": train_process.pid,
        "babysit_pid": babysit_process.pid,
        "output_dir": str(output_dir),
        "train_log": str(train_log),
        "train_stdout": str(train_stdout),
        "status_log": str(status_log),
        "babysit_stdout": str(babysit_stdout),
        "eval_root": str(eval_root),
        "report_path": str(report_path),
        "report_json_path": str(report_json_path),
        "train_exit_code_path": str(exit_code_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"started run {run_name}")
    print(f"train_pid={train_process.pid}")
    print(f"babysit_pid={babysit_process.pid}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
