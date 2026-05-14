# SenseAI Public Example

This repo includes a small public SenseAI example for demonstration purposes only.

## Important limitation

The public Hugging Face release of SenseAI does **not** currently expose the full `1,439+` row corpus described in the paper.

What is publicly accessible:

- the dataset card
- schema notes
- a screenshot preview containing a single visible example row

Sources:

- https://huggingface.co/datasets/SenseAI-RizqSpark/SenseAI
- https://arxiv.org/abs/2604.05135

## Files in this repo

- `data/public/senseai_snapshot_raw.jsonl`
  - a direct transcription of the visible screenshot row
  - preserves the fact that the reasoning text is truncated in the screenshot

- `data/public/senseai_qwen_thinking_example.jsonl`
  - a Qwen-style training example in the repo's `instruction` / `input` / `output` JSONL format
  - uses `<think>...</think>` in the assistant output to show a reasoning-aware target

## Why this is only an example

This public example is useful to show classmates:

- how a reasoning-aware finance row can be represented
- how Qwen-style thinking output might look in JSONL
- how sentiment labels can be paired with explicit reasoning

It is **not** enough to serve as the real baseline dataset by itself because:

- it is only one visible row
- one field is visibly truncated in the screenshot
- the full market outcome field is only described in the card, not released as a public corpus

## Recommended use

- Use `data/public/senseai_qwen_thinking_example.jsonl` as a formatting example.
- Use WRDS/8-K/TAQ/IBES-style data for the real supervised baseline.
- Treat the public SenseAI snapshot as a template for annotation quality, not as the main training corpus.
