#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch


ADAPTER_WEIGHT_FILENAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_model.pt",
)
TRAINER_STATE_FILENAME = "trainer_state.json"
ADAPTER_CONFIG_FILENAME = "adapter_config.json"


def resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def find_adapter_weight_files(path: Path) -> list[Path]:
    return [path / name for name in ADAPTER_WEIGHT_FILENAMES if (path / name).is_file()]


def find_adapter_artifacts(path: Path) -> list[Path]:
    if not path.exists():
        return []

    artifacts: list[Path] = []
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.name == ADAPTER_CONFIG_FILENAME or candidate.name in ADAPTER_WEIGHT_FILENAMES:
            artifacts.append(candidate)
    return sorted(artifacts)


def validate_adapter_dir(path: str | Path) -> Path:
    adapter_path = resolve_path(path)
    assert adapter_path is not None
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
    if not adapter_path.is_dir():
        raise NotADirectoryError(f"Adapter path is not a directory: {adapter_path}")
    if not (adapter_path / ADAPTER_CONFIG_FILENAME).is_file():
        raise FileNotFoundError(
            f"Adapter config is missing: {adapter_path / ADAPTER_CONFIG_FILENAME}"
        )
    weight_files = find_adapter_weight_files(adapter_path)
    if not weight_files:
        expected = ", ".join(ADAPTER_WEIGHT_FILENAMES)
        raise FileNotFoundError(
            f"Adapter weights are missing in {adapter_path}. Expected one of: {expected}"
        )
    return adapter_path


def validate_resume_checkpoint(path: str | Path) -> Path:
    checkpoint_path = validate_adapter_dir(path)
    trainer_state = checkpoint_path / TRAINER_STATE_FILENAME
    if not trainer_state.is_file():
        raise FileNotFoundError(
            f"Trainer checkpoint is malformed: missing {TRAINER_STATE_FILENAME} at {trainer_state}"
        )
    return checkpoint_path


def ensure_safe_output_dir(
    output_dir: str | Path,
    resume_from_checkpoint: str | Path | None = None,
    allow_overwrite_output_dir: bool = False,
) -> Path:
    output_path = resolve_path(output_dir)
    assert output_path is not None
    artifacts = find_adapter_artifacts(output_path)
    if not artifacts:
        return output_path

    if allow_overwrite_output_dir:
        return output_path

    resume_path = resolve_path(resume_from_checkpoint)
    if resume_path is not None and output_path in resume_path.parents:
        return output_path

    sample = ", ".join(str(path.relative_to(output_path)) for path in artifacts[:3])
    raise FileExistsError(
        "Refusing to use output_dir because adapter artifacts already exist in "
        f"{output_path} ({sample}). Pass --allow-overwrite-output-dir to override."
    )


FINGERPRINT_FILENAME = "fingerprint.json"


def compute_lora_fingerprint(state_dict: dict) -> dict[str, float]:
    """Deterministic, load-path-independent summary of LoRA weight magnitudes.

    Records the absolute-value sum (float64) of a fixed sample of LoRA
    tensors by their exact saved key names. Comparing this after a
    checkpoint reload catches the case where weights were silently left at
    their default/zero init due to a key-name mismatch (transformers/Unsloth
    module-path drift, a missing PEFT adapter-name suffix, an environment
    upgrade between save and resume, etc.) even when the loader raised no
    error or warning. This is exactly the failure mode found in
    eval/evaluate_base_vs_adapter.py against checkpoint-4500 of the
    qwen36-27b-wrds-500k-unsloth-gb10-rerun-20260616T2330Z run: PeftModel.from_pretrained
    reported "missing adapter keys" but did not abort, so "adapter" and
    "base" generations were silently identical.
    """
    lora_keys = sorted(k for k in state_dict if "lora_A" in k or "lora_B" in k)
    if not lora_keys:
        return {}
    sample = lora_keys[:3] + lora_keys[-3:] if len(lora_keys) > 6 else lora_keys
    return {
        key: float(state_dict[key].detach().to(torch.float64).abs().sum().item())
        for key in sample
    }


def write_fingerprint(checkpoint_dir: str | Path, state_dict: dict) -> Path | None:
    fingerprint = compute_lora_fingerprint(state_dict)
    if not fingerprint:
        return None
    path = Path(checkpoint_dir) / FINGERPRINT_FILENAME
    path.write_text(json.dumps(fingerprint, indent=2) + "\n")
    return path


def verify_resume_fingerprint(checkpoint_dir: str | Path, state_dict: dict, rel_tol: float = 1e-3) -> None:
    """Raise before training proceeds if the resumed model's LoRA weights do
    not actually match the checkpoint being resumed from. Silently does
    nothing if the checkpoint predates fingerprinting (no fingerprint.json).
    """
    path = Path(checkpoint_dir) / FINGERPRINT_FILENAME
    if not path.is_file():
        return
    expected: dict[str, float] = json.loads(path.read_text())

    missing = [key for key in expected if key not in state_dict]
    if missing:
        raise RuntimeError(
            "Resume fingerprint check failed: "
            f"{len(missing)} expected LoRA key(s) are absent from the resumed model "
            f"(e.g. {missing[0]}). The checkpoint's saved keys do not match this "
            "environment's model structure - resuming would silently restart the "
            "adapter from scratch instead of continuing it. Aborting before training "
            "proceeds."
        )

    mismatched: list[tuple[str, float, float]] = []
    for key, expected_value in expected.items():
        live_value = float(state_dict[key].detach().to(torch.float64).abs().sum().item())
        if abs(live_value - expected_value) > rel_tol * max(abs(expected_value), 1e-9):
            mismatched.append((key, expected_value, live_value))
    if mismatched:
        key, expected_value, live_value = mismatched[0]
        raise RuntimeError(
            "Resume fingerprint check failed: resumed LoRA weights do not match the "
            f"saved checkpoint for {len(mismatched)} sampled tensor(s) (e.g. {key}: "
            f"expected abs-sum {expected_value:.6f}, got {live_value:.6f}). The "
            "adapter was likely loaded as a fresh/default-init LoRA instead of your "
            "trained weights. Aborting before training proceeds."
        )


def training_target_summary(
    model_id: str,
    output_dir: str | Path,
    resume_from_checkpoint: str | Path | None = None,
    adapter_path: str | Path | None = None,
) -> str:
    output_path = resolve_path(output_dir)
    resume_path = resolve_path(resume_from_checkpoint)
    adapter_dir = resolve_path(adapter_path)
    return "\n".join(
        [
            f"base_model: {model_id}",
            f"adapter_path: {adapter_dir if adapter_dir is not None else '<new adapter run>'}",
            f"resume_from_checkpoint: {resume_path if resume_path is not None else '<none>'}",
            f"output_dir: {output_path}",
        ]
    )
