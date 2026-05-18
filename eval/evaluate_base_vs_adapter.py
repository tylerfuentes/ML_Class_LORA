#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import build_fingpt_prompt  # noqa: E402


QWEN_THINKING_CHOICES = ("thinking", "nonthinking", "both")
BENCHMARK_CHOICES = ("fpb", "fiqa", "tfns", "nwgi", "headline")
WRDS_OUTPUT_KEYS = ("direction_label", "event_type", "magnitude_bucket")
SENTIMENT3_LABELS = ("negative", "neutral", "positive")
HEADLINE_LABELS = ("no", "yes")


@dataclass
class Example:
    example_id: str
    dataset_name: str
    instruction: str
    input_text: str
    gold_output_text: str
    target_label: str | None
    target_structured: dict[str, Any] | None
    task_type: str
    metadata: dict[str, Any]


@dataclass
class ModelRunOutput:
    variant: str
    prompt_mode: str
    records: list[dict[str, Any]]
    metrics: dict[str, Any]
    support_status: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--eval-file")
    parser.add_argument("--holdout-file")
    parser.add_argument("--benchmark", choices=BENCHMARK_CHOICES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true", default=False)
    parser.add_argument("--qwen-thinking-mode", choices=QWEN_THINKING_CHOICES, default="both")
    parser.add_argument("--benchmark-split-size", type=int, default=128)
    parser.add_argument("--benchmark-seed", type=int, default=42)
    parser.add_argument(
        "--instruction-suffix",
        default="",
        help="Optional suffix appended to every instruction before prompt rendering.",
    )
    parser.add_argument(
        "--thinking-instruction-suffix",
        default="",
        help="Optional suffix appended only for thinking-mode prompts.",
    )
    parser.add_argument(
        "--nonthinking-instruction-suffix",
        default="",
        help="Optional suffix appended only for non-thinking-mode prompts.",
    )
    parser.add_argument(
        "--run-label",
        default="",
        help="Optional human-readable label for this diagnostic run.",
    )
    args = parser.parse_args()

    if not args.eval_file and not args.holdout_file and not args.benchmark:
        parser.error("Provide at least one of --eval-file, --holdout-file, or --benchmark.")
    if args.eval_file and args.holdout_file:
        parser.error("Use only one of --eval-file or --holdout-file.")
    return args


def make_tokenizer(model_id: str, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def make_model(model_id: str, local_files_only: bool, adapter_path: str | None):
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = {"": 0}
    else:
        load_kwargs["device_map"] = "cpu"

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    base = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model = PeftModel.from_pretrained(base, adapter_path) if adapter_path else base
    model.eval()
    return model


def prompt_modes(choice: str) -> list[str]:
    if choice == "both":
        return ["thinking", "nonthinking"]
    return [choice]


def tokenizer_supports_enable_thinking(tokenizer) -> bool:
    if not hasattr(tokenizer, "apply_chat_template"):
        return False
    signature = inspect.signature(tokenizer.apply_chat_template)
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or name == "enable_thinking"
        for name, parameter in signature.parameters.items()
    )


def effective_instruction(args: argparse.Namespace, example: Example, prompt_mode: str) -> str:
    instruction = example.instruction
    if args.instruction_suffix:
        instruction = f"{instruction}\n{args.instruction_suffix.strip()}"
    mode_suffix = args.thinking_instruction_suffix if prompt_mode == "thinking" else args.nonthinking_instruction_suffix
    if mode_suffix:
        instruction = f"{instruction}\n{mode_suffix.strip()}"
    return instruction


def build_plain_prompt(args: argparse.Namespace, example: Example, prompt_mode: str) -> str:
    return build_fingpt_prompt(effective_instruction(args, example, prompt_mode), example.input_text)


def ensure_chat_template_probe(
    tokenizer,
    plain_prompt: str,
    support_status: dict[str, Any],
) -> None:
    if support_status.get("probe_complete"):
        return

    support_status["probe_complete"] = True
    if not hasattr(tokenizer, "apply_chat_template"):
        support_status["fallback_reasons"].add("tokenizer_missing_apply_chat_template")
        return
    if not tokenizer_supports_enable_thinking(tokenizer):
        support_status["fallback_reasons"].add("tokenizer_missing_enable_thinking")
        return

    messages = [{"role": "user", "content": plain_prompt}]
    try:
        thinking_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        nonthinking_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        support_status["fallback_reasons"].add("enable_thinking_call_failed")
        return
    except Exception as exc:
        support_status["fallback_reasons"].add(f"chat_template_error:{type(exc).__name__}")
        return

    if thinking_prompt == nonthinking_prompt:
        support_status["fallback_reasons"].add("enable_thinking_no_effect")
        return

    support_status["chat_template_effective"] = True


def build_prompt(
    args: argparse.Namespace,
    tokenizer,
    example: Example,
    prompt_mode: str,
    support_status: dict[str, Any],
) -> str:
    plain_prompt = build_plain_prompt(args, example, prompt_mode)
    ensure_chat_template_probe(tokenizer, plain_prompt, support_status)
    if support_status["fallback_reasons"]:
        return plain_prompt

    content = plain_prompt
    messages = [{"role": "user", "content": content}]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    kwargs["enable_thinking"] = prompt_mode == "thinking"

    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        support_status["fallback_reasons"].add("enable_thinking_call_failed")
        return plain_prompt
    except Exception as exc:
        support_status["fallback_reasons"].add(f"chat_template_error:{type(exc).__name__}")
        return plain_prompt

    support_status["used_chat_template"] = True
    return rendered


def extract_think_block(text: str) -> dict[str, Any]:
    open_count = text.count("<think>")
    close_count = text.count("</think>")
    malformed = open_count != close_count or open_count > 1 or close_count > 1
    think_text = None
    parseable_after_strip = False

    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if match:
        think_text = match.group(1)
        stripped = text[: match.start()] + text[match.end() :]
    else:
        stripped = text
        if open_count or close_count:
            malformed = True

    stripped = stripped.strip()
    if stripped:
        parseable_after_strip = True

    first_brace = stripped.find("{")
    json_after_reasoning_text = first_brace > 0 and stripped[:first_brace].strip() != ""

    return {
        "contains_think": think_text is not None or open_count > 0 or close_count > 0,
        "think_text": think_text,
        "malformed_think_tags": malformed,
        "stripped_text": stripped,
        "parseable_after_strip": parseable_after_strip,
        "json_after_reasoning_text": json_after_reasoning_text,
    }


def normalize_label_text(text: str) -> str | None:
    lowered = text.lower()
    if "positive" in lowered:
        return "positive"
    if "negative" in lowered:
        return "negative"
    if "neutral" in lowered:
        return "neutral"
    if re.search(r"\byes\b", lowered):
        return "yes"
    if re.search(r"\bno\b", lowered):
        return "no"
    return None


def maybe_parse_json_blob(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def parse_wrds_prediction(text: str) -> dict[str, Any]:
    think_info = extract_think_block(text)
    final_text = think_info["stripped_text"]
    parsed_json = maybe_parse_json_blob(final_text)
    structured: dict[str, Any] = {}
    compliance = False

    if parsed_json is not None:
        compliance = True
        for key in WRDS_OUTPUT_KEYS:
            if key in parsed_json:
                structured[key] = parsed_json[key]

    if "direction_label" not in structured:
        direction = normalize_label_text(final_text)
        if direction in SENTIMENT3_LABELS:
            structured["direction_label"] = direction

    if "magnitude_bucket" not in structured:
        magnitude_match = re.search(r'"?magnitude_bucket"?\s*[:=]\s*"?(small|medium|large|unknown)"?', final_text, re.IGNORECASE)
        if magnitude_match:
            structured["magnitude_bucket"] = magnitude_match.group(1).lower()

    if "event_type" not in structured:
        event_match = re.search(r'"?event_type"?\s*[:=]\s*"?(ibes_[a-z_]+)"?', final_text, re.IGNORECASE)
        if event_match:
            structured["event_type"] = event_match.group(1)

    parse_success = "direction_label" in structured
    return {
        **think_info,
        "parsed_text": final_text,
        "parsed_json": parsed_json,
        "parsed_structured": structured,
        "parsed_label": structured.get("direction_label"),
        "parse_success": parse_success,
        "format_compliant": compliance,
    }


def parse_classification_prediction(text: str) -> dict[str, Any]:
    think_info = extract_think_block(text)
    final_text = think_info["stripped_text"]
    label = normalize_label_text(final_text)
    return {
        **think_info,
        "parsed_text": final_text,
        "parsed_json": None,
        "parsed_structured": None,
        "parsed_label": label,
        "parse_success": label is not None,
        "format_compliant": label is not None,
    }


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_classification_metrics(
    records: list[dict[str, Any]],
    label_key: str,
    known_labels: list[str],
) -> dict[str, Any]:
    total = len(records)
    valid_records = [row for row in records if row["parsed_label"] is not None]
    invalid_count = total - len(valid_records)
    correct = sum(1 for row in valid_records if row["parsed_label"] == row[label_key])
    accuracy = safe_div(correct, total)

    per_class: dict[str, Any] = {}
    f1_values = []
    confusion: Counter[tuple[str, str], int] = Counter()
    for row in records:
        gold = row[label_key]
        pred = row["parsed_label"] if row["parsed_label"] is not None else "__invalid__"
        confusion[(gold, pred)] += 1

    for label in known_labels:
        tp = confusion[(label, label)]
        fp = sum(confusion[(gold, label)] for gold in known_labels if gold != label)
        fn = sum(confusion[(label, pred)] for pred in [*known_labels, "__invalid__"] if pred != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall) if precision + recall else 0.0
        support = sum(confusion[(label, pred)] for pred in [*known_labels, "__invalid__"])
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)

    avg_output_chars = safe_div(sum(len(row["raw_output"]) for row in records), total)
    avg_output_tokens = safe_div(sum(row["generated_tokens"] for row in records), total)
    avg_latency = safe_div(sum(row["latency_seconds"] for row in records), total)
    contains_think_rate = safe_div(sum(1 for row in records if row["contains_think"]), total)
    malformed_think_rate = safe_div(sum(1 for row in records if row["malformed_think_tags"]), total)
    truncated_output_rate = safe_div(sum(1 for row in records if row["output_truncated"]), total)
    json_after_reasoning_text_rate = safe_div(sum(1 for row in records if row["json_after_reasoning_text"]), total)
    parseable_after_strip_rate = safe_div(
        sum(1 for row in records if row["parseable_after_strip"] and row["parse_success"]),
        total,
    )
    parse_failure_rate = safe_div(invalid_count, total)
    compliance_rate = safe_div(sum(1 for row in records if row["format_compliant"]), total)

    return {
        "sample_count": total,
        "accuracy": accuracy,
        "macro_f1": safe_div(sum(f1_values), len(f1_values)),
        "invalid_parse_count": invalid_count,
        "parse_failure_rate": parse_failure_rate,
        "format_compliance_rate": compliance_rate,
        "average_output_chars": avg_output_chars,
        "average_output_tokens": avg_output_tokens,
        "average_latency_seconds": avg_latency,
        "truncated_output_rate": truncated_output_rate,
        "contains_think_rate": contains_think_rate,
        "malformed_think_tag_rate": malformed_think_rate,
        "json_after_reasoning_text_rate": json_after_reasoning_text_rate,
        "parseable_after_strip_rate": parseable_after_strip_rate,
        "per_class": per_class,
        "confusion": {f"{gold}->{pred}": count for (gold, pred), count in sorted(confusion.items())},
    }


def compute_wrds_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    base = compute_classification_metrics(records, "target_label", list(SENTIMENT3_LABELS))
    exact_match_count = 0
    magnitude_correct = 0
    event_type_correct = 0
    for row in records:
        gold = row["gold_structured"] or {}
        pred = row["parsed_structured"] or {}
        exact = all(pred.get(key) == gold.get(key) for key in WRDS_OUTPUT_KEYS)
        exact_match_count += int(exact)
        magnitude_correct += int(pred.get("magnitude_bucket") == gold.get("magnitude_bucket"))
        event_type_correct += int(pred.get("event_type") == gold.get("event_type"))
        row["exact_json_match"] = exact
        row["magnitude_bucket_correct"] = pred.get("magnitude_bucket") == gold.get("magnitude_bucket")
        row["event_type_correct"] = pred.get("event_type") == gold.get("event_type")

    total = len(records)
    base["exact_json_match_rate"] = safe_div(exact_match_count, total)
    base["magnitude_bucket_accuracy"] = safe_div(magnitude_correct, total)
    base["event_type_accuracy"] = safe_div(event_type_correct, total)
    return base


def delta_metrics(base_metrics: dict[str, Any], adapter_metrics: dict[str, Any], keys: list[str]) -> dict[str, float]:
    return {
        key: adapter_metrics.get(key, 0.0) - base_metrics.get(key, 0.0)
        for key in keys
    }


def load_jsonl_examples(path: Path, dataset_name: str) -> list[Example]:
    rows: list[Example] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            raw = json.loads(line)
            gold_output = raw["output"]
            gold_structured = json.loads(gold_output) if gold_output.strip().startswith("{") else None
            target_label = gold_structured.get("direction_label") if gold_structured else normalize_label_text(gold_output)
            example_id = raw.get("event_id") or raw.get("id") or f"{dataset_name}-{idx}"
            rows.append(
                Example(
                    example_id=example_id,
                    dataset_name=dataset_name,
                    instruction=raw["instruction"],
                    input_text=raw.get("input", ""),
                    gold_output_text=gold_output,
                    target_label=target_label,
                    target_structured=gold_structured,
                    task_type="wrds_structured" if gold_structured else "classification",
                    metadata={k: v for k, v in raw.items() if k not in {"instruction", "input", "output"}},
                )
            )
    return rows


def benchmark_data_dir() -> Path:
    return REPO_ROOT / "external" / "FinGPT" / "fingpt" / "FinGPT_Benchmark" / "data"


def ensure_benchmark_dataset(name: str, local_files_only: bool) -> tuple[DatasetDict | Any | None, str | None]:
    specs: dict[str, tuple[list[tuple[str, str | None]], str]] = {
        "fpb": (
            [
                ("financial_phrasebank", "sentences_50agree"),
                ("atrost/financial_phrasebank", None),
                ("ArtGarfunkel/FinancialPhraseBank", None),
            ],
            "financial_phrasebank-sentences_50agree",
        ),
        "fiqa": ([("pauri32/fiqa-2018", None)], "fiqa-2018"),
        "tfns": ([("zeroshot/twitter-financial-news-sentiment", None)], "twitter-financial-news-sentiment"),
        "nwgi": ([("oliverwang15/news_with_gpt_instructions", None)], "news_with_gpt_instructions"),
        "headline": ([("FinGPT/fingpt-headline", None)], "fingpt-headline-instruct"),
    }
    sources, dest = specs[name]
    dest_path = benchmark_data_dir() / dest
    if dest_path.exists():
        return load_from_disk(str(dest_path)), None
    if local_files_only:
        return None, f"local benchmark cache missing for {name}: {dest_path}"

    errors: list[str] = []
    for source in sources:
        try:
            ds = load_dataset(*source)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(dest_path))
            return ds, None
        except Exception as exc:
            errors.append(f"{source[0]}: {type(exc).__name__}: {exc}")
    return None, f"benchmark auto-download failed for {name}: {' | '.join(errors)}"


def as_dataset_dict(ds: Any) -> DatasetDict:
    if isinstance(ds, DatasetDict):
        return ds
    if hasattr(ds, "keys"):
        return DatasetDict(ds)
    raise TypeError(f"Unsupported dataset container: {type(ds)}")


def load_benchmark_examples(name: str, local_files_only: bool, slice_size: int, seed: int) -> tuple[list[Example], str | None]:
    ds, error = ensure_benchmark_dataset(name, local_files_only)
    if error:
        return [], error
    dataset_dict = as_dataset_dict(ds)

    if name == "fpb":
        split = dataset_dict["train"].train_test_split(seed=seed)["test"]
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        examples = [
            Example(
                example_id=f"fpb-{i}",
                dataset_name="fpb",
                instruction="What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.",
                input_text=row["sentence"],
                gold_output_text=label_map[row["label"]],
                target_label=label_map[row["label"]],
                target_structured=None,
                task_type="classification",
                metadata={"source_split": "derived_test"},
            )
            for i, row in enumerate(split)
        ]
    elif name == "fiqa":
        combined = concatenate_datasets([dataset_dict["train"], dataset_dict["validation"], dataset_dict["test"]])
        split = combined.train_test_split(test_size=0.226, seed=seed)["test"]

        def fiqa_label(score: float) -> str:
            if score < -0.1:
                return "negative"
            if score < 0.1:
                return "neutral"
            return "positive"

        examples = []
        for i, row in enumerate(split):
            news_type = "tweet" if row.get("format") == "post" else "news"
            instruction = f"What is the sentiment of this {news_type}? Please choose an answer from {{negative/neutral/positive}}."
            label = fiqa_label(float(row["sentiment_score"]))
            examples.append(
                Example(
                    example_id=f"fiqa-{i}",
                    dataset_name="fiqa",
                    instruction=instruction,
                    input_text=row["sentence"],
                    gold_output_text=label,
                    target_label=label,
                    target_structured=None,
                    task_type="classification",
                    metadata={"source_split": "derived_test", "format": row.get("format")},
                )
            )
    elif name == "tfns":
        split = dataset_dict["validation"]
        label_map = {0: "negative", 1: "positive", 2: "neutral"}
        examples = [
            Example(
                example_id=f"tfns-{i}",
                dataset_name="tfns",
                instruction="What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.",
                input_text=row["text"],
                gold_output_text=label_map[row["label"]],
                target_label=label_map[row["label"]],
                target_structured=None,
                task_type="classification",
                metadata={"source_split": "validation"},
            )
            for i, row in enumerate(split)
        ]
    elif name == "nwgi":
        label_map = {
            "strong negative": "negative",
            "moderately negative": "negative",
            "mildly negative": "neutral",
            "neutral": "neutral",
            "mildly positive": "neutral",
            "moderately positive": "positive",
            "strong positive": "positive",
        }
        split_name = "test" if "test" in dataset_dict else next(iter(dataset_dict.keys()))
        split = dataset_dict[split_name]
        examples = [
            Example(
                example_id=f"nwgi-{i}",
                dataset_name="nwgi",
                instruction="What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.",
                input_text=row["news"],
                gold_output_text=label_map[row["label"]],
                target_label=label_map[row["label"]],
                target_structured=None,
                task_type="classification",
                metadata={"source_split": split_name, "original_label": row["label"]},
            )
            for i, row in enumerate(split)
        ]
    elif name == "headline":
        split_name = "test" if "test" in dataset_dict else next(iter(dataset_dict.keys()))
        split = dataset_dict[split_name]
        examples = []
        for i, row in enumerate(split):
            output = str(row["output"]).strip()
            label = "yes" if re.search(r"\byes\b", output.lower()) else "no"
            examples.append(
                Example(
                    example_id=f"headline-{i}",
                    dataset_name="headline",
                    instruction=row["instruction"],
                    input_text=row.get("input", ""),
                    gold_output_text=output,
                    target_label=label,
                    target_structured=None,
                    task_type="headline_binary",
                    metadata={"source_split": split_name},
                )
            )
    else:
        raise ValueError(f"Unsupported benchmark: {name}")

    if slice_size and len(examples) > slice_size:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(examples), generator=generator).tolist()
        examples = [examples[i] for i in perm[:slice_size]]
    return examples, None


def run_generation(
    args: argparse.Namespace,
    tokenizer,
    model,
    examples: list[Example],
    variant: str,
    prompt_mode: str,
) -> ModelRunOutput:
    records: list[dict[str, Any]] = []
    support_status = {"used_chat_template": False, "fallback_reasons": set()}

    parser_fn = parse_wrds_prediction if any(ex.task_type == "wrds_structured" for ex in examples) else parse_classification_prediction

    for start in range(0, len(examples), max(1, args.batch_size)):
        batch = examples[start : start + max(1, args.batch_size)]
        prompts = [build_prompt(args, tokenizer, ex, prompt_mode, support_status) for ex in batch]
        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
            return_token_type_ids=False,
        )
        tokenized = {key: value.to(model.device) for key, value in tokenized.items()}
        start_time = time.perf_counter()
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature
        with torch.no_grad():
            generated = model.generate(**tokenized, **generation_kwargs)
        batch_latency = time.perf_counter() - start_time

        attention_lengths = tokenized["attention_mask"].sum(dim=1).tolist()
        for idx, ex in enumerate(batch):
            continuation_ids = generated[idx][attention_lengths[idx] :]
            raw_output = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
            parsed = parser_fn(raw_output)
            record = {
                "example_id": ex.example_id,
                "dataset_name": ex.dataset_name,
                "task_type": ex.task_type,
                "variant": variant,
                "prompt_mode": prompt_mode,
                "instruction": ex.instruction,
                "effective_instruction": effective_instruction(args, ex, prompt_mode),
                "input": ex.input_text,
                "prompt_text": prompts[idx],
                "gold_output_text": ex.gold_output_text,
                "target_label": ex.target_label,
                "gold_structured": ex.target_structured,
                "raw_output": raw_output,
                "generated_tokens": int(continuation_ids.shape[0]),
                "max_new_tokens": args.max_new_tokens,
                "output_truncated": int(continuation_ids.shape[0]) >= args.max_new_tokens,
                "latency_seconds": batch_latency / len(batch),
                **parsed,
                "metadata": ex.metadata,
            }
            records.append(record)

        del tokenized, generated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = compute_wrds_metrics(records) if any(ex.task_type == "wrds_structured" for ex in examples) else compute_classification_metrics(
        records,
        "target_label",
        list(HEADLINE_LABELS) if examples and examples[0].task_type == "headline_binary" else list(SENTIMENT3_LABELS),
    )
    metrics["prompt_mode"] = prompt_mode
    metrics["variant"] = variant
    metrics["used_chat_template"] = support_status["used_chat_template"]
    metrics["fallback_reasons"] = sorted(support_status["fallback_reasons"])
    metrics["chat_template_effective"] = support_status.get("chat_template_effective", False)
    return ModelRunOutput(
        variant=variant,
        prompt_mode=prompt_mode,
        records=records,
        metrics=metrics,
        support_status=support_status,
    )


def unload_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def write_confusion_csv(path: Path, sections: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "variant", "prompt_mode", "gold_label", "predicted_label", "count"])
        writer.writeheader()
        for section in sections:
            for key, count in section["metrics"]["confusion"].items():
                gold, pred = key.split("->", 1)
                writer.writerow(
                    {
                        "dataset": section["dataset"],
                        "variant": section["variant"],
                        "prompt_mode": section["prompt_mode"],
                        "gold_label": gold,
                        "predicted_label": pred,
                        "count": count,
                    }
                )


def verdict(delta: dict[str, float]) -> str:
    if delta["accuracy"] > 0 and delta["macro_f1"] > 0 and delta["parse_failure_rate"] <= 0:
        return "improved"
    if delta["accuracy"] < 0 or delta["macro_f1"] < 0 or delta["parse_failure_rate"] > 0:
        return "regressed"
    return "inconclusive"


def build_regression_examples(
    base_records: list[dict[str, Any]],
    adapter_records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    base_by_id = {row["example_id"]: row for row in base_records}
    adapter_by_id = {row["example_id"]: row for row in adapter_records}
    wins: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    unchanged_wrong: list[dict[str, Any]] = []

    for example_id, adapter_row in adapter_by_id.items():
        base_row = base_by_id[example_id]
        gold = adapter_row["target_label"]
        base_correct = base_row["parsed_label"] == gold
        adapter_correct = adapter_row["parsed_label"] == gold
        merged = {
            "example_id": example_id,
            "prompt_mode": adapter_row["prompt_mode"],
            "instruction": adapter_row["instruction"],
            "input": adapter_row["input"],
            "gold": gold,
            "base_output": base_row["raw_output"],
            "adapter_output": adapter_row["raw_output"],
            "base_label": base_row["parsed_label"],
            "adapter_label": adapter_row["parsed_label"],
        }
        if adapter_correct and not base_correct and len(wins) < 10:
            wins.append(merged)
        elif base_correct and not adapter_correct and len(regressions) < 10:
            regressions.append(merged)
        elif not base_correct and not adapter_correct and len(unchanged_wrong) < 10:
            unchanged_wrong.append(merged)
    return {"wins": wins, "regressions": regressions, "unchanged_wrong": unchanged_wrong}


def write_regression_examples(path: Path, grouped: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    lines = ["# Regression Examples", ""]
    for section_name, buckets in grouped.items():
        lines.append(f"## {section_name}")
        lines.append("")
        for title, examples in buckets.items():
            lines.append(f"### {title.replace('_', ' ').title()}")
            lines.append("")
            if not examples:
                lines.append("None available.")
                lines.append("")
                continue
            for idx, ex in enumerate(examples, start=1):
                lines.append(f"{idx}. `{ex['example_id']}` mode=`{ex['prompt_mode']}` gold=`{ex['gold']}`")
                lines.append(f"   base=`{ex['base_label']}` adapter=`{ex['adapter_label']}`")
                lines.append(f"   instruction: {ex['instruction']}")
                lines.append(f"   input: {ex['input'][:300]}")
                lines.append(f"   base output: {ex['base_output'][:300]}")
                lines.append(f"   adapter output: {ex['adapter_output'][:300]}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_prompt_examples(path: Path, records: list[dict[str, Any]]) -> None:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in records:
        key = (row["prompt_mode"], row["example_id"])
        if key in seen:
            continue
        seen.add(key)
        by_mode[row["prompt_mode"]].append(row)

    lines = ["# Prompt Examples", ""]
    modes = sorted(by_mode.keys())
    if "thinking" in by_mode and "nonthinking" in by_mode:
        thinking_prompt = by_mode["thinking"][0]["prompt_text"]
        nonthinking_prompt = by_mode["nonthinking"][0]["prompt_text"]
        lines.append(f"- thinking_vs_nonthinking_rendered_prompts_differ: `{thinking_prompt != nonthinking_prompt}`")
        lines.append("")

    for mode in modes:
        lines.append(f"## {mode}")
        lines.append("")
        for row in by_mode[mode][:3]:
            lines.append(f"### `{row['example_id']}`")
            lines.append("")
            lines.append("Effective instruction:")
            lines.append("")
            lines.append("```text")
            lines.append(row["effective_instruction"])
            lines.append("```")
            lines.append("")
            lines.append("Rendered prompt:")
            lines.append("")
            lines.append("```text")
            lines.append(row["prompt_text"])
            lines.append("```")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_thinking_examples(path: Path, records: list[dict[str, Any]]) -> None:
    thinking_rows = [row for row in records if row["prompt_mode"] == "thinking"][:10]
    lines = ["# Raw Thinking Outputs", ""]
    if not thinking_rows:
        lines.append("No thinking-mode rows available.")
    else:
        for row in thinking_rows:
            lines.append(f"## `{row['variant']}` `{row['example_id']}`")
            lines.append("")
            lines.append(f"- generated_tokens: `{row['generated_tokens']}`")
            lines.append(f"- output_truncated: `{row['output_truncated']}`")
            lines.append(f"- parse_success: `{row['parse_success']}`")
            lines.append(f"- json_after_reasoning_text: `{row['json_after_reasoning_text']}`")
            lines.append("")
            lines.append("```text")
            lines.append(row["raw_output"][:4000])
            lines.append("```")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_eval_summary(path: Path, summary_rows: list[dict[str, Any]], skipped: list[str]) -> None:
    lines = ["# Eval Summary", ""]
    if skipped:
        lines.append("## Skipped")
        lines.append("")
        for item in skipped:
            lines.append(f"- {item}")
        lines.append("")

    datasets = defaultdict(list)
    for row in summary_rows:
        datasets[row["dataset"]].append(row)

    for dataset_name, rows in datasets.items():
        lines.append(f"## {dataset_name}")
        lines.append("")
        for row in rows:
            lines.append(f"### {row['variant']} / {row['prompt_mode']}")
            lines.append("")
            lines.append(f"- accuracy: `{row['metrics']['accuracy']:.4f}`")
            lines.append(f"- macro_f1: `{row['metrics']['macro_f1']:.4f}`")
            lines.append(f"- parse_failure_rate: `{row['metrics']['parse_failure_rate']:.4f}`")
            lines.append(f"- average_output_tokens: `{row['metrics']['average_output_tokens']:.2f}`")
            lines.append(f"- truncated_output_rate: `{row['metrics']['truncated_output_rate']:.4f}`")
            lines.append(f"- contains_think_rate: `{row['metrics']['contains_think_rate']:.4f}`")
            lines.append(f"- malformed_think_tag_rate: `{row['metrics']['malformed_think_tag_rate']:.4f}`")
            lines.append(f"- json_after_reasoning_text_rate: `{row['metrics']['json_after_reasoning_text_rate']:.4f}`")
            lines.append(f"- parseable_after_strip_rate: `{row['metrics']['parseable_after_strip_rate']:.4f}`")
            lines.append(f"- used_chat_template: `{row['metrics']['used_chat_template']}`")
            lines.append(f"- chat_template_effective: `{row['metrics']['chat_template_effective']}`")
            lines.append("")

        by_mode = defaultdict(dict)
        for row in rows:
            by_mode[row["prompt_mode"]][row["variant"]] = row["metrics"]
        if by_mode:
            lines.append("### Qwen Thinking Mode Comparison")
            lines.append("")
            for mode, variants in sorted(by_mode.items()):
                base_metrics = variants.get("base")
                adapter_metrics = variants.get("adapter")
                if not base_metrics or not adapter_metrics:
                    continue
                delta = delta_metrics(
                    base_metrics,
                    adapter_metrics,
                    ["accuracy", "macro_f1", "parse_failure_rate", "contains_think_rate", "parseable_after_strip_rate"],
                )
                lines.append(f"- `{mode}` base accuracy=`{base_metrics['accuracy']:.4f}` adapter accuracy=`{adapter_metrics['accuracy']:.4f}` delta=`{delta['accuracy']:+.4f}`")
                lines.append(f"- `{mode}` base macro_f1=`{base_metrics['macro_f1']:.4f}` adapter macro_f1=`{adapter_metrics['macro_f1']:.4f}` delta=`{delta['macro_f1']:+.4f}`")
                lines.append(f"- `{mode}` parse failure delta=`{delta['parse_failure_rate']:+.4f}` think-tag delta=`{delta['contains_think_rate']:+.4f}`")
                lines.append(f"- `{mode}` readiness verdict: `{verdict(delta)}`")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = make_tokenizer(args.model_id, args.local_files_only)
    tasks: list[tuple[str, list[Example]]] = []
    skipped: list[str] = []

    data_file = args.holdout_file or args.eval_file
    if data_file:
        dataset_name = "wrds_holdout" if args.holdout_file else "jsonl_eval"
        examples = load_jsonl_examples(Path(data_file), dataset_name)
        if args.max_examples:
            examples = examples[: args.max_examples]
        tasks.append((dataset_name, examples))

    if args.benchmark:
        benchmark_examples, error = load_benchmark_examples(
            args.benchmark,
            args.local_files_only,
            args.benchmark_split_size,
            args.benchmark_seed,
        )
        if error:
            skipped.append(error)
        else:
            if args.max_examples:
                benchmark_examples = benchmark_examples[: args.max_examples]
            tasks.append((f"benchmark_{args.benchmark}", benchmark_examples))

    summary_rows: list[dict[str, Any]] = []
    all_base_records: list[dict[str, Any]] = []
    all_adapter_records: list[dict[str, Any]] = []
    grouped_regressions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    metrics_payload: dict[str, Any] = {
        "model_id": args.model_id,
        "adapter_path": str(Path(args.adapter_path).resolve()),
        "qwen_thinking_mode": args.qwen_thinking_mode,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "local_files_only": args.local_files_only,
        "instruction_suffix": args.instruction_suffix,
        "thinking_instruction_suffix": args.thinking_instruction_suffix,
        "nonthinking_instruction_suffix": args.nonthinking_instruction_suffix,
        "run_label": args.run_label,
        "tasks": {},
        "skipped": skipped,
    }

    for dataset_name, examples in tasks:
        if not examples:
            skipped.append(f"{dataset_name}: no examples loaded")
            continue

        task_metrics: dict[str, Any] = {}
        for mode in prompt_modes(args.qwen_thinking_mode):
            base_model = make_model(args.model_id, args.local_files_only, adapter_path=None)
            base_result = run_generation(args, tokenizer, base_model, examples, "base", mode)
            unload_model(base_model)

            adapter_model = make_model(args.model_id, args.local_files_only, adapter_path=args.adapter_path)
            adapter_result = run_generation(args, tokenizer, adapter_model, examples, "adapter", mode)
            unload_model(adapter_model)

            for result in (base_result, adapter_result):
                if result.metrics["fallback_reasons"]:
                    print(
                        f"[warn] prompt-mode fallback for dataset={dataset_name} "
                        f"variant={result.variant} mode={mode}: "
                        f"{', '.join(result.metrics['fallback_reasons'])}"
                    )

            all_base_records.extend(base_result.records)
            all_adapter_records.extend(adapter_result.records)
            summary_rows.append({"dataset": dataset_name, "variant": "base", "prompt_mode": mode, "metrics": base_result.metrics})
            summary_rows.append({"dataset": dataset_name, "variant": "adapter", "prompt_mode": mode, "metrics": adapter_result.metrics})

            deltas = delta_metrics(
                base_result.metrics,
                adapter_result.metrics,
                ["accuracy", "macro_f1", "parse_failure_rate", "format_compliance_rate", "contains_think_rate", "parseable_after_strip_rate"],
            )
            task_metrics[mode] = {
                "base": base_result.metrics,
                "adapter": adapter_result.metrics,
                "delta": deltas,
                "verdict": verdict(deltas),
            }
            grouped_regressions[f"{dataset_name}:{mode}"] = build_regression_examples(base_result.records, adapter_result.records)

        metrics_payload["tasks"][dataset_name] = task_metrics

    write_jsonl(output_dir / "predictions_base.jsonl", all_base_records)
    write_jsonl(output_dir / "predictions_adapter.jsonl", all_adapter_records)
    write_json(output_dir / "metrics.json", metrics_payload)
    write_json(output_dir / "run_config.json", vars(args))
    write_confusion_csv(output_dir / "confusion_matrix.csv", summary_rows)
    write_eval_summary(output_dir / "eval_summary.md", summary_rows, skipped)
    write_regression_examples(output_dir / "regression_examples.md", grouped_regressions)
    write_prompt_examples(output_dir / "prompt_examples.md", all_base_records)
    write_thinking_examples(output_dir / "thinking_examples.md", all_base_records + all_adapter_records)

    print(json.dumps({"output_dir": str(output_dir), "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
