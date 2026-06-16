#!/usr/bin/env python3
import argparse
import math
import os
from datetime import UTC, datetime
from pathlib import Path

import torch
from unsloth import FastLanguageModel
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from common import (
    LORA_TARGET_MODULES,
    build_tokenized_dataset,
    load_jsonl_examples,
    write_json,
)
from safety import ensure_safe_output_dir, training_target_summary, validate_resume_checkpoint


def make_model_and_tokenizer(args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=not args.disable_qlora,
        full_finetuning=False,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
        use_gradient_checkpointing="unsloth",
        fast_inference=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_lora_rank=args.lora_r,
        disable_log_stats=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=args.max_seq_length,
    )
    FastLanguageModel.for_training(model)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file")
    parser.add_argument("--test-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--disable-qlora", action="store_true", default=False)
    parser.add_argument("--local-files-only", action="store_true", default=False)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--allow-overwrite-output-dir", action="store_true", default=False)
    parser.add_argument("--max-total-examples", type=int, default=0)
    args = parser.parse_args()

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

    train_examples = load_jsonl_examples(args.train_file)
    eval_examples = load_jsonl_examples(args.eval_file) if args.eval_file else []
    test_examples = load_jsonl_examples(args.test_file) if args.test_file else []
    total_examples = len(train_examples) + len(eval_examples) + len(test_examples)
    if args.max_total_examples and total_examples > args.max_total_examples:
        raise ValueError(
            "Refusing to train on more than "
            f"{args.max_total_examples} total examples. "
            "Trim the dataset or set --max-total-examples 0 to disable."
        )

    os.makedirs(output_dir, exist_ok=True)
    print(
        training_target_summary(
            model_id=args.model_id,
            output_dir=output_dir,
            resume_from_checkpoint=resume_checkpoint,
        )
    )
    model, tokenizer = make_model_and_tokenizer(args)
    model.print_trainable_parameters()

    train_dataset = build_tokenized_dataset(train_examples, args.max_seq_length, tokenizer)
    eval_dataset = (
        build_tokenized_dataset(eval_examples, args.max_seq_length, tokenizer)
        if eval_examples
        else None
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    max_steps = max(1, math.ceil(len(train_dataset) * args.epochs / effective_batch_size))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
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
        optim="adamw_8bit",
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=(args.eval_steps or args.save_steps) if eval_dataset is not None else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    start = datetime.now(UTC).isoformat()
    result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None
    )
    eval_metrics = trainer.evaluate() if eval_dataset is not None else {}
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    run_summary = {
        "start_utc": start,
        "end_utc": datetime.now(UTC).isoformat(),
        "model_id": args.model_id,
        "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "output_dir": str(output_dir),
        "train_file": os.path.abspath(args.train_file),
        "eval_file": os.path.abspath(args.eval_file) if args.eval_file else None,
        "test_file": os.path.abspath(args.test_file) if args.test_file else None,
        "num_train_examples": len(train_examples),
        "num_eval_examples": len(eval_examples),
        "num_test_examples": len(test_examples),
        "num_total_examples": total_examples,
        "max_steps": max_steps,
        "effective_batch_size": effective_batch_size,
        "backend": "unsloth",
        "train_runtime": result.metrics.get("train_runtime"),
        "train_loss": result.metrics.get("train_loss"),
        "eval_loss": eval_metrics.get("eval_loss"),
        "peak_gpu_mem_allocated_gb": (
            torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None
        ),
        "peak_gpu_mem_reserved_gb": (
            torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else None
        ),
    }
    write_json(str(Path(output_dir) / "run_summary.json"), run_summary)
    print(run_summary)


if __name__ == "__main__":
    main()
