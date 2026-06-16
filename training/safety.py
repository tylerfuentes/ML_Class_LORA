#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


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
