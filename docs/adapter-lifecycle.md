# Adapter Lifecycle

This repo uses PEFT LoRA adapters on top of `Qwen/Qwen3.6-27B`. The safe operating model is:

1. load the base model
2. attach one adapter for inference or training
3. verify which adapter is active
4. disable or delete the adapter only when you mean to change behavior
5. delete Python references and clear CUDA cache only when you are done with the whole model

This document is about operational safety. It does not merge weights and it does not start new modeling work.

## Core distinctions

These operations are not interchangeable:

- `disable_adapter`
  - Temporarily bypasses LoRA behavior.
  - The adapter modules still exist on the model object.
  - GPU memory is still held by the live model and adapter tensors.

- `delete_adapter`
  - Removes an adapter module from a live PEFT model if the installed PEFT version supports it.
  - This is different from disabling.
  - The base model object still exists and still occupies GPU memory.

- `del model`, `gc.collect()`, `torch.cuda.empty_cache()`
  - This is how you release model memory after the model object is no longer needed.
  - `torch.cuda.empty_cache()` only releases unused cached blocks. It does not free memory still referenced by live tensors.

- `resume_from_checkpoint`
  - Continues a Hugging Face Trainer run from a Trainer checkpoint directory such as `checkpoint-500`.
  - Requires Trainer state files in addition to adapter files.
  - This is not the same as simply loading an adapter for inference.

- `merge_and_unload`
  - Permanently folds LoRA weights into the base model for export/inference.
  - Do not use this when you want to continue LoRA training later.
  - This repo does not use merge for lifecycle switching unless explicitly requested.

## Load an adapter for inference

The normal inference path is:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", ...)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", ...)
model = PeftModel.from_pretrained(base, "outputs/qwen36-27b-ibes-baseline")
model.eval()
```

If you want a smoke-tested CLI path in this repo, use:

```bash
python scripts/adapter_lifecycle_check.py \
  --base-model Qwen/Qwen3.6-27B \
  --adapter-path outputs/qwen36-27b-ibes-baseline
```

## Disable an adapter temporarily

If your PEFT version supports `disable_adapter`, use it when you want to compare base-model behavior without destroying the loaded adapter:

```python
with model.disable_adapter():
    ...
```

Operationally:

- use this for quick A/B checks
- do not assume memory was freed
- do not assume the adapter is gone from the model

## Delete or unload an adapter from a live model

If your PEFT version supports `delete_adapter`, you can remove a named adapter from a live model:

```python
model.delete_adapter("lifecycle_adapter")
```

Operationally:

- this removes adapter modules from the live PEFT model
- the base model still exists
- you still need full model cleanup if you want GPU memory back

The smoke script attempts this automatically and reports whether the installed PEFT version supports it.

## Fully release model memory from GPU

To actually free memory for model switching:

```python
del model
del tokenizer
import gc
gc.collect()
import torch
torch.cuda.empty_cache()
```

Recommended order:

1. stop using the model
2. delete model references
3. run `gc.collect()`
4. run `torch.cuda.empty_cache()`
5. verify memory counters or `nvidia-smi`

Use `scripts/adapter_lifecycle_check.py` to print:

- `torch.cuda.memory_allocated()`
- `torch.cuda.memory_reserved()`
- `torch.cuda.max_memory_allocated()`

before load, after inference, after adapter deletion, and after final cleanup.

## Reload an adapter for continued training

There are two different cases:

### Case 1: continue inference-only use

Reload the base model and attach the adapter again:

```python
base = AutoModelForCausalLM.from_pretrained(...)
model = PeftModel.from_pretrained(base, "outputs/qwen36-27b-ibes-baseline")
```

### Case 2: resume a Trainer run

Use a Trainer checkpoint directory, not just the root adapter directory:

- valid example:
  - `outputs/qwen36-27b-ibes-10k-controlled/checkpoint-500`
- not the same thing:
  - `outputs/qwen36-27b-ibes-10k-controlled`

Before resuming, validate:

```bash
python scripts/check_resume_safety.py \
  --resume-from-checkpoint outputs/qwen36-27b-ibes-10k-controlled/checkpoint-500 \
  --output-dir outputs/qwen36-27b-ibes-10k-controlled
```

Then resume training:

```bash
python training/train_finance_lora.py \
  --model-id Qwen/Qwen3.6-27B \
  --train-file data/.../train.jsonl \
  --eval-file data/.../eval.jsonl \
  --output-dir outputs/qwen36-27b-ibes-10k-controlled \
  --resume-from-checkpoint outputs/qwen36-27b-ibes-10k-controlled/checkpoint-500
```

Important:

- `resume_from_checkpoint` is for Trainer checkpoints
- it expects `trainer_state.json`
- it should normally point at `checkpoint-*`, not only the adapter root

## Verify the correct adapter is active

At minimum, verify all of the following:

1. the adapter path you loaded is the one you intended
2. `active_adapters` or `active_adapter` on the model matches expectation
3. a smoke inference runs successfully
4. the training script prints the exact base model, checkpoint, and output directory before training begins

This repo now provides:

- `scripts/adapter_lifecycle_check.py` for inference-side inspection
- `scripts/check_resume_safety.py` for checkpoint integrity checks
- conservative training entrypoints that print:
  - base model
  - adapter path or new-run status
  - resume checkpoint
  - output directory

## Known local adapter paths

- baseline adapter root
  - `outputs/qwen36-27b-ibes-baseline`
- 10k controlled Trainer checkpoint
  - `outputs/qwen36-27b-ibes-10k-controlled/checkpoint-500`

For continued Trainer resume, prefer the checkpoint path. For inference-only adapter loading, the adapter root is sufficient if it contains:

- `adapter_config.json`
- `adapter_model.safetensors` or equivalent adapter weights
