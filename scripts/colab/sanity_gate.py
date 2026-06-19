#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import build_fingpt_prompt, load_jsonl_examples  # noqa: E402

FINAL_ADAPTER_FILES = (
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_model.pt",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "run_summary.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short Colab sanity training pass before the full job.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "colab_paths.example.yaml")
    parser.add_argument("--run-name", default="colab-a100-qwen36-27b-unsloth")
    parser.add_argument("--sanity-steps", type=int, default=5)
    parser.add_argument("--num-generate-examples", type=int, default=5)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    return path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_final_adapter(output_dir: Path, adapters_root: Path, run_name: str) -> Path:
    adapter_dir = ensure_dir(adapters_root / run_name)
    for name in FINAL_ADAPTER_FILES:
        source = output_dir / name
        if source.exists():
            shutil.copy2(source, adapter_dir / name)
    return adapter_dir


def make_model_and_tokenizer(model_id: str, adapter_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        quantization_config=quantization_config,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model, tokenizer


def generate_examples(model_id: str, adapter_path: Path, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model, tokenizer = make_model_and_tokenizer(model_id, adapter_path)
    outputs: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx, example in enumerate(examples, start=1):
            prompt = build_fingpt_prompt(example["instruction"], example.get("input", ""))
            rendered = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
            generated = model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
            outputs.append(
                {
                    "example_index": idx,
                    "instruction": example["instruction"],
                    "input": example.get("input", ""),
                    "gold_output": example["output"],
                    "generated_output": text.strip(),
                }
            )
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    drive_cfg = cfg["drive"]
    datasets_cfg = cfg["datasets"]
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]

    train_file = ensure_file(Path(datasets_cfg["train_file"]))
    train_eval_file = ensure_file(Path(datasets_cfg["train_eval_file"]))
    test_file = ensure_file(Path(datasets_cfg["test_file"]))
    checkpoints_root = ensure_dir(Path(drive_cfg["checkpoints"]))
    adapters_root = ensure_dir(Path(drive_cfg["adapters"]))
    outputs_root = ensure_dir(Path(drive_cfg["outputs"]))
    manifests_root = ensure_dir(Path(drive_cfg["manifests"]))

    sanity_run_name = f"{args.run_name}-sanity"
    output_dir = checkpoints_root / sanity_run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = ensure_dir(outputs_root / "sanity_logs") / f"{sanity_run_name}.log"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "training" / "train_finance_lora_unsloth.py"),
        "--model-id",
        model_cfg["model_id"],
        "--train-file",
        str(train_file),
        "--eval-file",
        str(train_eval_file),
        "--test-file",
        str(test_file),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(training_cfg["epochs"]),
        "--lr",
        str(training_cfg["learning_rate"]),
        "--per-device-train-batch-size",
        str(training_cfg["per_device_train_batch_size"]),
        "--gradient-accumulation-steps",
        str(training_cfg["gradient_accumulation_steps"]),
        "--save-steps",
        str(args.sanity_steps),
        "--logging-steps",
        "1",
        "--eval-steps",
        str(args.sanity_steps),
        "--lora-r",
        str(training_cfg["lora_r"]),
        "--lora-alpha",
        str(training_cfg["lora_alpha"]),
        "--lora-dropout",
        str(training_cfg["lora_dropout"]),
        "--max-seq-length",
        str(training_cfg["max_seq_length"]),
        "--gpu-memory-utilization",
        str(training_cfg["gpu_memory_utilization"]),
        "--max-total-examples",
        str(training_cfg["max_total_examples"]),
        "--max-steps-override",
        str(args.sanity_steps),
        "--allow-overwrite-output-dir",
    ]
    if model_cfg.get("local_files_only", False):
        cmd.append("--local-files-only")
    if training_cfg.get("disable_qlora", False):
        cmd.append("--disable-qlora")

    print("Running sanity train:", " ".join(cmd))
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{datetime.now(UTC).isoformat()}] sanity command: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        print(f"Sanity training failed; see {log_path}", file=sys.stderr)
        return result.returncode

    adapter_dir = copy_final_adapter(output_dir, adapters_root, sanity_run_name)
    examples = load_jsonl_examples(str(test_file))[: args.num_generate_examples]
    generations = generate_examples(model_cfg["model_id"], adapter_dir, examples)

    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "sanity_run_name": sanity_run_name,
        "sanity_steps": args.sanity_steps,
        "output_dir": str(output_dir),
        "adapter_dir": str(adapter_dir),
        "log_path": str(log_path),
        "generations": generations,
    }
    manifest_path = manifests_root / f"{sanity_run_name}_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"sanity_adapter_dir={adapter_dir}")
    print(f"sanity_manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
