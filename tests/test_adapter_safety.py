#!/usr/bin/env python3
"""Unit tests for adapter lifecycle safety checks."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from safety import ensure_safe_output_dir, validate_adapter_dir, validate_resume_checkpoint


class AdapterSafetyTests(unittest.TestCase):
    def test_missing_adapter_path_fails(self) -> None:
        with self.assertRaises(FileNotFoundError):
            validate_adapter_dir(REPO_ROOT / "outputs" / "does-not-exist")

    def test_malformed_checkpoint_dir_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint-500"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
            (checkpoint / "adapter_model.safetensors").write_text("stub", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                validate_resume_checkpoint(checkpoint)

    def test_overwrite_protection_triggers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "existing-output"
            output_dir.mkdir()
            (output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (output_dir / "adapter_model.safetensors").write_text("stub", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_safe_output_dir(output_dir)

    def test_resume_inside_existing_output_dir_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            checkpoint = output_dir / "checkpoint-500"
            checkpoint.mkdir(parents=True)
            (output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (output_dir / "adapter_model.safetensors").write_text("stub", encoding="utf-8")
            (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
            (checkpoint / "adapter_model.safetensors").write_text("stub", encoding="utf-8")
            (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
            resolved = ensure_safe_output_dir(
                output_dir,
                resume_from_checkpoint=checkpoint,
                allow_overwrite_output_dir=False,
            )
            self.assertEqual(resolved, output_dir.resolve())


if __name__ == "__main__":
    unittest.main()
