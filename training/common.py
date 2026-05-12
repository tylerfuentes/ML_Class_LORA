#!/usr/bin/env python3
import json
import os
from dataclasses import dataclass
from typing import Iterable

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM_PROMPT = "You are a precise financial analysis assistant."
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class ModelConfig:
    model_id: str
    max_seq_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    use_qlora: bool
    local_files_only: bool


def format_example(example: dict) -> str:
    instruction = example["instruction"].strip()
    user_block = f"Instruction: {instruction}"
    input_text = example.get("input", "").strip()
    if input_text:
        user_block += f"\nInput: {input_text}"
    output_text = example["output"].strip()
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_block}<|im_end|>\n"
        f"<|im_start|>assistant\n{output_text}<|im_end|>"
    )


def load_jsonl_examples(path: str) -> list[dict]:
    examples: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            example = json.loads(line)
            missing = {"instruction", "output"} - example.keys()
            if missing:
                raise ValueError(f"{path}:{lineno} missing required keys: {sorted(missing)}")
            examples.append(
                {
                    "instruction": example["instruction"],
                    "input": example.get("input", ""),
                    "output": example["output"],
                }
            )
    if not examples:
        raise ValueError(f"{path} contained no training rows")
    return examples


def build_tokenized_dataset(examples: Iterable[dict], max_seq_length: int, tokenizer) -> Dataset:
    rows = [{"text": format_example(example)} for example in examples]
    dataset = Dataset.from_list(rows)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=max_seq_length,
        )

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def make_model_and_tokenizer(cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=cfg.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": cfg.local_files_only,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = {"": 0}
    else:
        load_kwargs["device_map"] = "cpu"

    if cfg.use_qlora:
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **load_kwargs)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if cfg.use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_cfg)
    return model, tokenizer


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

