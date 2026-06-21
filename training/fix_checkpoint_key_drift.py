#!/usr/bin/env python3
"""One-time repair for the language_model./missing-".default" LoRA key
drift described in docs/eval-findings.md (checkpoint-4500 of the
qwen36-27b-wrds-500k-unsloth-gb10-rerun-20260616T2330Z run was saved under an
environment whose model exposed LoRA modules at
"base_model.model.model.language_model.layers...lora_A.weight"; this
environment's model exposes them at
"base_model.model.model.layers...lora_A.default.weight"). Backs up the
original checkpoint directory, then remaps adapter_model.safetensors keys
and drops the now-stale fingerprint.json in place so the resume-fingerprint
check in training/safety.py stops treating it as a structural mismatch.
"""
import argparse
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def remap_key(key: str) -> str:
    candidate = key.replace("language_model.", "")
    if candidate.endswith(".weight") and not candidate.endswith(".default.weight"):
        candidate = candidate[: -len(".weight")] + ".default.weight"
    return candidate


def fix_checkpoint_in_place(checkpoint_dir: Path, backup_dir: Path) -> None:
    if backup_dir.exists():
        raise FileExistsError(f"{backup_dir} already exists; refusing to overwrite a prior backup")
    shutil.copytree(checkpoint_dir, backup_dir)

    adapter_file = checkpoint_dir / "adapter_model.safetensors"
    state = load_file(str(adapter_file))
    remapped = {remap_key(key): tensor for key, tensor in state.items()}
    drifted = sum(1 for key in state if remap_key(key) != key)
    save_file(remapped, str(adapter_file))

    fingerprint_file = checkpoint_dir / "fingerprint.json"
    if fingerprint_file.exists():
        fingerprint_file.unlink()

    print(
        f"remapped {drifted}/{len(state)} drifted adapter keys in {adapter_file}; "
        f"original backed up to {backup_dir}; stale fingerprint.json removed"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    args = parser.parse_args()
    fix_checkpoint_in_place(args.checkpoint_dir, args.backup_dir)


if __name__ == "__main__":
    main()
