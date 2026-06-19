#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test loading, disabling, unloading, and freeing adapter/model memory."
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--adapter-path", help="Optional local PEFT adapter directory.")
    parser.add_argument("--adapter-name", default="lifecycle_adapter")
    parser.add_argument(
        "--prompt",
        default="Summarize: Revenue rose from $10M to $12M while margin fell from 15% to 13%.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true", default=False)
    parser.add_argument("--use-qlora", action="store_true", default=True)
    parser.add_argument("--disable-qlora", action="store_true")
    return parser.parse_args()


def memory_snapshot(label: str) -> None:
    import torch

    print(f"\n[{label}]")
    if not torch.cuda.is_available():
        print("cuda_available: false")
        return
    print(f"cuda_available: true")
    print(f"memory_allocated: {torch.cuda.memory_allocated()}")
    print(f"memory_reserved: {torch.cuda.memory_reserved()}")
    print(f"max_memory_allocated: {torch.cuda.max_memory_allocated()}")


def load_base_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    load_kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else "cpu"
    use_qlora = args.use_qlora and not args.disable_qlora
    if use_qlora:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    model.eval()
    return model, tokenizer


def require_adapter_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Adapter path is not a directory: {path}")
    return path


def active_adapters(model: Any) -> list[str]:
    active = getattr(model, "active_adapters", None)
    if callable(active):
        active = active()
    if active is None:
        current = getattr(model, "active_adapter", None)
        if current is None:
            return []
        if isinstance(current, str):
            return [current]
        if isinstance(current, (list, tuple, set)):
            return [str(item) for item in current]
        return [str(current)]
    if isinstance(active, str):
        return [active]
    return [str(item) for item in active]


def run_smoke_inference(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int, label: str) -> None:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
    print(f"\n[{label}] {tokenizer.decode(output[0], skip_special_tokens=True)}")


def try_disable_adapter(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> None:
    disable_adapter = getattr(model, "disable_adapter", None)
    if disable_adapter is None:
        print("\n[disable_adapter] unsupported by this PEFT version")
        return
    try:
        disabled = disable_adapter()
        if hasattr(disabled, "__enter__") and hasattr(disabled, "__exit__"):
            with disabled:
                run_smoke_inference(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens,
                    label="adapter disabled temporarily",
                )
        else:
            run_smoke_inference(
                model,
                tokenizer,
                prompt,
                max_new_tokens,
                label="adapter disable invoked",
            )
    except Exception as exc:  # pragma: no cover - runtime compatibility path
        print(f"\n[disable_adapter] failed: {exc}")


def try_delete_adapter(model: Any, adapter_name: str) -> None:
    delete_adapter = getattr(model, "delete_adapter", None)
    if delete_adapter is None:
        print("\n[delete_adapter] unsupported by this PEFT version")
        return
    try:
        delete_adapter(adapter_name)
        print(f"\n[delete_adapter] removed adapter: {adapter_name}")
    except Exception as exc:  # pragma: no cover - runtime compatibility path
        print(f"\n[delete_adapter] failed: {exc}")


def main() -> None:
    if any(flag in sys.argv[1:] for flag in ("-h", "--help")):
        parse_args()
        return

    import torch
    from peft import PeftModel

    args = parse_args()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    memory_snapshot("before_load")

    model = None
    tokenizer = None
    try:
        model, tokenizer = load_base_model(args)
        memory_snapshot("after_base_model_load")

        if args.adapter_path:
            adapter_path = require_adapter_path(args.adapter_path)
            model = PeftModel.from_pretrained(model, adapter_path, adapter_name=args.adapter_name)
            model.set_adapter(args.adapter_name)
            print(f"\n[adapter_loaded] path={adapter_path}")
        else:
            adapter_path = None

        print(f"\n[active_adapters] {active_adapters(model)}")
        run_smoke_inference(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            label="smoke_inference",
        )
        memory_snapshot("after_smoke_inference")

        if adapter_path is not None:
            try_disable_adapter(model, tokenizer, args.prompt, args.max_new_tokens)
            try_delete_adapter(model, args.adapter_name)
            print(f"\n[active_adapters_after_delete] {active_adapters(model)}")
            memory_snapshot("after_adapter_delete")
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        memory_snapshot("after_gc_and_empty_cache")


if __name__ == "__main__":
    main()
