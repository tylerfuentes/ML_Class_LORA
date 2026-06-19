#!/usr/bin/env python3
import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from common import ModelConfig, build_tokenized_dataset, make_model_and_tokenizer, write_json
from safety import ensure_safe_output_dir, training_target_summary, validate_resume_checkpoint


def smoke_examples() -> list[dict]:
    return [
        {
            "instruction": "Summarize quarterly revenue trends from this data.",
            "input": "Q1 $12.1M, Q2 $12.8M, Q3 $13.0M, Q4 $14.4M",
            "output": "Revenue increased steadily through the year, with the strongest improvement in Q4.",
        },
        {
            "instruction": "Classify operating margin performance.",
            "input": "Operating margin moved from 11.2% to 9.7% over two quarters.",
            "output": "Performance deteriorated because operating margin compressed by 1.5 percentage points.",
        },
        {
            "instruction": "Give one risk and one mitigation for FX exposure.",
            "input": "Company has 38% revenue in EUR and reports in USD.",
            "output": "Risk: adverse EUR/USD movement can reduce reported revenue. Mitigation: hedge forecasted EUR cash flows.",
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--output-dir", default="./outputs/qwen36-27b-finance-lora-smoke")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-qlora", action="store_true", default=True)
    parser.add_argument("--disable-qlora", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", default=False)
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Resume from a Trainer checkpoint directory such as checkpoint-500.",
    )
    parser.add_argument(
        "--allow-overwrite-output-dir",
        action="store_true",
        default=False,
        help="Allow writing into an output directory that already contains adapter artifacts.",
    )
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
    output_dir = ensure_safe_output_dir(
        args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        allow_overwrite_output_dir=args.allow_overwrite_output_dir,
    )
    resume_checkpoint = (
        validate_resume_checkpoint(args.resume_from_checkpoint)
        if args.resume_from_checkpoint
        else None
    )

    os.makedirs(output_dir, exist_ok=True)
    print(
        training_target_summary(
            model_id=cfg.model_id,
            output_dir=output_dir,
            resume_from_checkpoint=resume_checkpoint,
        )
    )
    model, tokenizer = make_model_and_tokenizer(cfg)
    model.print_trainable_parameters()
    dataset = build_tokenized_dataset(smoke_examples(), cfg.max_seq_length, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=args.max_steps,
        save_total_limit=1,
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
    train_result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None
    )
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "start_utc": start,
        "end_utc": datetime.utcnow().isoformat(),
        "model_id": cfg.model_id,
        "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "output_dir": str(output_dir),
        "use_qlora": cfg.use_qlora,
        "max_steps": args.max_steps,
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_loss": train_result.metrics.get("train_loss"),
        "peak_gpu_mem_allocated_gb": (
            torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None
        ),
        "peak_gpu_mem_reserved_gb": (
            torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else None
        ),
    }
    write_json(str(Path(output_dir) / "smoke_metrics.json"), metrics)
    print(metrics)


if __name__ == "__main__":
    main()
