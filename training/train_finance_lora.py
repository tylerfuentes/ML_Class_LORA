#!/usr/bin/env python3
import argparse
import math
import os
from datetime import datetime

import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from common import (
    ModelConfig,
    build_tokenized_dataset,
    load_jsonl_examples,
    make_model_and_tokenizer,
    write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--train-file", required=True, help="Path to JSONL training data.")
    parser.add_argument("--output-dir", default="./outputs/qwen36-27b-finance-lora")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-qlora", action="store_true", default=True)
    parser.add_argument("--disable-qlora", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", default=False)
    args = parser.parse_args()

    cfg = ModelConfig(
        model_id=args.model_id,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_qlora=args.use_qlora and not args.disable_qlora,
        local_files_only=args.local_files_only,
    )

    examples = load_jsonl_examples(args.train_file)
    dataset = None
    os.makedirs(args.output_dir, exist_ok=True)
    model, tokenizer = make_model_and_tokenizer(cfg)
    model.print_trainable_parameters()
    dataset = build_tokenized_dataset(examples, cfg.max_seq_length, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    max_steps = max(1, math.ceil(len(dataset) * args.epochs / effective_batch_size))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        max_steps=max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="paged_adamw_8bit" if cfg.use_qlora else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    start = datetime.utcnow().isoformat()
    result = trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    run_summary = {
        "start_utc": start,
        "end_utc": datetime.utcnow().isoformat(),
        "model_id": cfg.model_id,
        "train_file": os.path.abspath(args.train_file),
        "num_examples": len(examples),
        "max_steps": max_steps,
        "effective_batch_size": effective_batch_size,
        "use_qlora": cfg.use_qlora,
        "train_runtime": result.metrics.get("train_runtime"),
        "train_loss": result.metrics.get("train_loss"),
        "peak_gpu_mem_allocated_gb": (
            torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None
        ),
        "peak_gpu_mem_reserved_gb": (
            torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else None
        ),
    }
    write_json(os.path.join(args.output_dir, "run_summary.json"), run_summary)
    print(run_summary)


if __name__ == "__main__":
    main()
