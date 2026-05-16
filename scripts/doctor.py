#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from ibes_pipeline import DEFAULT_RAW_IBES, EXPECTED_IBES_COLUMNS, format_size, load_csv_header, resolve_path


def status_line(level: str, message: str) -> str:
    return f"{level}: {message}"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    raw_path = resolve_path(DEFAULT_RAW_IBES)
    fin_gpt_path = repo_root / "external" / "FinGPT"
    required_dirs = [
        repo_root / "admin" / "local" / "wrds-downloads",
        repo_root / "data" / "processed",
        repo_root / "outputs",
        repo_root / "checkpoints",
    ]

    warnings: list[str] = []
    failures: list[str] = []

    print(status_line("PASS", f"python={platform.python_version()}"))
    print(status_line("PASS", f"cwd={Path.cwd()}"))
    print(status_line("PASS", f"repo_root={repo_root}"))

    if raw_path.exists():
        print(status_line("PASS", f"raw_ibes_csv={raw_path} ({format_size(raw_path.stat().st_size)})"))
        try:
            header = load_csv_header(raw_path)
            missing = [column for column in EXPECTED_IBES_COLUMNS if column not in header]
            if missing:
                failures.append(f"raw IBES file missing expected columns: {missing}")
            else:
                print(status_line("PASS", f"raw_ibes_header_columns={len(header)}"))
        except Exception as exc:
            failures.append(f"failed to read raw IBES header: {exc}")
    else:
        warnings.append(
            f"raw IBES CSV not found at {raw_path}. Download from Google Drive / WRDS before running the pipeline."
        )

    if fin_gpt_path.exists():
        print(status_line("PASS", f"fingpt_submodule={fin_gpt_path}"))
    else:
        failures.append("external/FinGPT is missing. Run: git submodule update --init --recursive")

    try:
        import torch

        print(status_line("PASS", f"torch={torch.__version__}"))
        cuda_ok = bool(torch.cuda.is_available())
        print(status_line("PASS", f"torch.cuda.is_available={cuda_ok}"))
        if cuda_ok:
            print(status_line("PASS", f"cuda_device={torch.cuda.get_device_name(0)}"))
        else:
            warnings.append("CUDA is not visible. QLoRA training will not use the GPU.")
    except Exception as exc:
        failures.append(f"torch import failed: {exc}")

    for module_name in ["transformers", "peft", "bitsandbytes", "pandas", "pyarrow", "yaml"]:
        try:
            module = __import__(module_name)
            print(status_line("PASS", f"{module_name}={getattr(module, '__version__', 'unknown')}"))
        except Exception as exc:
            failures.append(f"{module_name} import failed: {exc}")

    try:
        __import__("unsloth")
        warnings.append("unsloth is installed in the main environment. Do not use it here unless it is isolated and verified.")
    except Exception:
        print(status_line("PASS", "unsloth_not_installed_in_main_env"))

    for directory in required_dirs:
        if directory.exists():
            print(status_line("PASS", f"local_dir={directory}"))
        else:
            warnings.append(f"expected local dir missing: {directory} (mkdir -p {directory})")

    if warnings:
        for warning in warnings:
            print(status_line("WARN", warning))
    if failures:
        for failure in failures:
            print(status_line("FAIL", failure))
        print(status_line("FAIL", f"doctor_summary fail={len(failures)} warn={len(warnings)}"))
        return 1

    print(status_line("PASS", f"doctor_summary fail=0 warn={len(warnings)}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
