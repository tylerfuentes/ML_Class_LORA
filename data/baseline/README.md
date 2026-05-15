# Baseline Dataset Split

Use this directory for the first real finance QLoRA baseline.

Target split:

- `train.jsonl`: `800` examples
- `eval.jsonl`: `100` examples
- `test.jsonl`: `100` examples

Rules:

- Keep total examples at or below `1,000` for the first real run.
- Prioritize clean, task-aligned examples over dataset size.
- Keep `data/samples/finance_train.jsonl` for smoke tests only.
- Do not scale up until the `r=16` adapter trains, saves, and reloads cleanly.

Expected JSONL format:

```json
{"instruction":"...","input":"...","output":"..."}
```
