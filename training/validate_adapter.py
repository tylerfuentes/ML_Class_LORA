#!/usr/bin/env python3
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--adapter-dir", default="./outputs/qwen36-27b-finance-lora-smoke")
    parser.add_argument(
        "--prompt",
        default="Summarize: Revenue rose from $10M to $12M while margin fell from 15% to 13%.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--local-files-only", action="store_true", default=False)
    parser.add_argument("--use-qlora", action="store_true", default=True)
    parser.add_argument("--disable-qlora", action="store_true")
    args = parser.parse_args()
    use_qlora = args.use_qlora and not args.disable_qlora

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = {"": 0}
    else:
        load_kwargs["device_map"] = "cpu"

    if use_qlora:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    base = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print(f"\n[ok] adapter loaded from: {args.adapter_dir}")


if __name__ == "__main__":
    main()
