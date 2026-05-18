# Evaluation Findings

## May 16, 2026 WRDS holdout result

Run:

```bash
python eval/evaluate_base_vs_adapter.py \
  --model-id Qwen/Qwen3.6-27B \
  --adapter-path outputs/qwen36-27b-ibes-baseline \
  --holdout-file data/processed/ibes_lora_baseline/jsonl/baseline_1k/holdout.jsonl \
  --output-dir outputs/evals/qwen36-27b-ibes-baseline-holdout \
  --qwen-thinking-mode both \
  --max-new-tokens 80 \
  --batch-size 1 \
  --local-files-only
```

Measured outcome:

- non-thinking mode was strong
- thinking mode was poor for both base and adapter
- the adapter dominated base in non-thinking mode
- the adapter did not preserve the measured thinking-mode behavior in this run

Non-thinking summary:

- base accuracy: `0.4200`
- adapter accuracy: `1.0000`
- base macro F1: `0.5183`
- adapter macro F1: `1.0000`
- base parse failure rate: `0.5600`
- adapter parse failure rate: `0.0000`

Thinking summary:

- base accuracy: `0.0100`
- adapter accuracy: `0.0000`
- base macro F1: `0.0202`
- adapter macro F1: `0.0000`
- base parse failure rate: `0.9900`
- adapter parse failure rate: `1.0000`

Interpretation rule:

- do claim strong non-thinking improvement on this WRDS structured task
- do not claim thinking-mode regression as a settled adapter property yet

Why that caution matters:

- `max_new_tokens=80` may be too low for a thinking-enabled path that emits extra text before the final JSON
- a chat-template or prompt mismatch may be steering the model toward prose instead of a machine-parseable answer
- parser and truncation diagnostics need to be checked on a small targeted rerun before concluding that the adapter itself regressed

Required next step:

- run a small 20 to 50 example diagnostic holdout evaluation with larger `max_new_tokens`
- compare base vs adapter in both thinking and non-thinking modes
- capture rendered prompts, truncation behavior, raw thinking outputs, and parse success after stripping reasoning text

Until those diagnostics are done:

- non-thinking adapter quality looks real
- thinking-mode preservation remains unproven

## May 16, 2026 diagnostic prompt and token-budget runs

Goal:

- test whether the poor thinking-mode result was caused by token budget, prompt shape, parsing, or a true adapter regression

Diagnostic runs:

```bash
python eval/evaluate_base_vs_adapter.py \
  --model-id Qwen/Qwen3.6-27B \
  --adapter-path outputs/qwen36-27b-ibes-baseline \
  --holdout-file data/processed/ibes_lora_baseline/jsonl/baseline_1k/holdout.jsonl \
  --output-dir outputs/evals/diag20-default-256 \
  --qwen-thinking-mode both \
  --max-examples 20 \
  --max-new-tokens 256 \
  --batch-size 1 \
  --local-files-only \
  --run-label diag20-default-256

python eval/evaluate_base_vs_adapter.py \
  --model-id Qwen/Qwen3.6-27B \
  --adapter-path outputs/qwen36-27b-ibes-baseline \
  --holdout-file data/processed/ibes_lora_baseline/jsonl/baseline_1k/holdout.jsonl \
  --output-dir outputs/evals/diag20-default-512 \
  --qwen-thinking-mode both \
  --max-examples 20 \
  --max-new-tokens 512 \
  --batch-size 1 \
  --local-files-only \
  --run-label diag20-default-512

python eval/evaluate_base_vs_adapter.py \
  --model-id Qwen/Qwen3.6-27B \
  --adapter-path outputs/qwen36-27b-ibes-baseline \
  --holdout-file data/processed/ibes_lora_baseline/jsonl/baseline_1k/holdout.jsonl \
  --output-dir outputs/evals/diag20-think-jsononly-256 \
  --qwen-thinking-mode both \
  --max-examples 20 \
  --max-new-tokens 256 \
  --batch-size 1 \
  --local-files-only \
  --thinking-instruction-suffix "Think internally if needed, but return only the final JSON object. Do not include reasoning text." \
  --run-label diag20-think-jsononly-256

python eval/evaluate_base_vs_adapter.py \
  --model-id Qwen/Qwen3.6-27B \
  --adapter-path outputs/qwen36-27b-ibes-baseline \
  --holdout-file data/processed/ibes_lora_baseline/jsonl/baseline_1k/holdout.jsonl \
  --output-dir outputs/evals/diag20-think-nonthinkhint-256 \
  --qwen-thinking-mode both \
  --max-examples 20 \
  --max-new-tokens 256 \
  --batch-size 1 \
  --local-files-only \
  --thinking-instruction-suffix "Use non-thinking mode for this structured classification task." \
  --run-label diag20-think-nonthinkhint-256
```

Thinking-mode comparison on the same 20 holdout examples:

- `diag20-default-256`: base accuracy `0.4000`, adapter accuracy `0.8000`, base macro F1 `0.5480`, adapter macro F1 `0.8001`
- `diag20-default-512`: base accuracy `0.7500`, adapter accuracy `0.7000`, base macro F1 `0.8410`, adapter macro F1 `0.6781`
- `diag20-think-jsononly-256`: base accuracy `0.6500`, adapter accuracy `0.8500`, base macro F1 `0.7116`, adapter macro F1 `0.8521`
- `diag20-think-nonthinkhint-256`: base accuracy `0.6000`, adapter accuracy `0.9000`, base macro F1 `0.6688`, adapter macro F1 `0.9011`

Non-thinking comparison on the same 20 holdout examples:

- `diag20-default-256`: base accuracy `0.9000`, adapter accuracy `1.0000`
- `diag20-default-512`: base accuracy `1.0000`, adapter accuracy `1.0000`
- `diag20-think-jsononly-256`: base accuracy `0.9000`, adapter accuracy `1.0000`
- `diag20-think-nonthinkhint-256`: base accuracy `0.9000`, adapter accuracy `1.0000`

What the diagnostics show:

- rendered thinking and non-thinking prompts do differ, and the thinking path opens an assistant `<think>` section in the chat template
- the models do not emit literal `<think>...</think>` blocks in output; `contains_think_rate` stayed `0.0000`
- default thinking-mode outputs are highly verbose and often machine-parseable only after stripping reasoning-style prose
- on the `512` run, thinking outputs still hit the token cap for both base and adapter on every example; `truncated_output_rate` stayed `1.0000`
- the adapter often does know the right final JSON in thinking mode, but it tends to place JSON after a long prose preamble; in the default `512` run the adapter `json_after_reasoning_text_rate` was `1.0000`
- the `80`-token full-holdout run was too small to support a stable conclusion about thinking compatibility

Interpretation:

- do not describe the original `80`-token thinking result as a clean adapter regression
- token budget and prompt shape clearly matter
- the strongest thinking-mode result in these diagnostics came from prompt steering, not retraining
- the best adapter thinking-mode result in this diagnostic set was the `Use non-thinking mode for this structured classification task.` hint at `256` tokens
- the non-thinking adapter result remains the strongest and most stable finding

Current working conclusion:

- the adapter is strong for this WRDS structured task in non-thinking mode
- thinking-mode readiness is partially recoverable at inference time with prompt changes
- thinking-mode behavior is still not stable enough to claim preservation or improvement as a general model property
- do not retrain for thinking mode yet
- a raw `1024` rerun is lower priority than better prompt control or a task-specific structured inference wrapper

## May 17, 2026 FinGPT benchmark slices

Goal:

- test whether the IBES adapter improves only the WRDS/IBES task or also helps on public finance classification benchmarks

Runs:

```bash
python eval/evaluate_base_vs_adapter.py \
  --model-id Qwen/Qwen3.6-27B \
  --adapter-path outputs/qwen36-27b-ibes-baseline \
  --benchmark fiqa \
  --benchmark-split-size 64 \
  --output-dir outputs/evals/qwen36-27b-fiqa-nonthinking \
  --qwen-thinking-mode nonthinking \
  --max-new-tokens 80 \
  --batch-size 1

python eval/evaluate_base_vs_adapter.py \
  --model-id Qwen/Qwen3.6-27B \
  --adapter-path outputs/qwen36-27b-ibes-baseline \
  --benchmark fpb \
  --benchmark-split-size 128 \
  --output-dir outputs/evals/qwen36-27b-fpb-nonthinking \
  --qwen-thinking-mode nonthinking \
  --max-new-tokens 80 \
  --batch-size 1
```

Benchmark results:

- `fiqa`:
  - base accuracy: `0.7812`
  - adapter accuracy: `0.8125`
  - base macro F1: `0.6491`
  - adapter macro F1: `0.6759`
  - base parse failure rate: `0.0781`
  - adapter parse failure rate: `0.0000`

- `fpb`:
  - base accuracy: `0.4688`
  - adapter accuracy: `0.7188`
  - base macro F1: `0.5229`
  - adapter macro F1: `0.7553`
  - base parse failure rate: `0.0156`
  - adapter parse failure rate: `0.0000`

Implementation note:

- the legacy `financial_phrasebank` dataset loader path no longer worked in the current `datasets` stack because dataset scripts are no longer supported
- the evaluator was updated to fall back to modern Hub mirrors for `fpb`

Interpretation:

- the current adapter is not just a brittle WRDS/IBES formatter
- on these first public benchmark slices, it improved over base Qwen in non-thinking mode
- this is still slice-level evidence, not full-benchmark final reporting
- the next reasonable benchmark expansion is `tfns` and `nwgi`, still in non-thinking mode first

## May 17, 2026 controlled 10k scale step

Goal:

- test the first controlled scale-up from the `800/100/100` baseline to `10000/1000/1000`
- keep non-thinking mode as the default for structured WRDS/IBES classification
- compare the new 10k adapter against base Qwen and the prior 1k adapter

Training:

- the `baseline_10k` split was materialized under `data/processed/ibes_lora_baseline/jsonl/baseline_10k`
- a controlled QLoRA run was launched with the verified HF/PEFT/bitsandbytes path against `Qwen/Qwen3.6-27B`
- the full epoch was intentionally not pushed to completion because validation at `checkpoint-500` had already improved over the 1k run while later full-holdout evaluation was the real gating signal
- the 10k adapter evaluated below is `outputs/qwen36-27b-ibes-10k-controlled/checkpoint-500`

Checkpoint note:

- `checkpoint-500` eval loss: `0.1778`
- prior 1k run final eval loss: `0.1848`
- lower eval loss alone was not treated as enough evidence to scale further

### 10k WRDS/IBES holdout

Run:

```bash
python eval/evaluate_base_vs_adapter.py \
  --model-id Qwen/Qwen3.6-27B \
  --adapter-path outputs/qwen36-27b-ibes-10k-controlled/checkpoint-500 \
  --holdout-file data/processed/ibes_lora_baseline/jsonl/baseline_10k/holdout.jsonl \
  --output-dir outputs/evals/qwen36-27b-ibes-10k-holdout \
  --qwen-thinking-mode nonthinking \
  --max-new-tokens 80 \
  --batch-size 1 \
  --local-files-only
```

Artifacts:

- summary: `outputs/evals/qwen36-27b-ibes-10k-holdout/eval_summary.md`
- metrics: `outputs/evals/qwen36-27b-ibes-10k-holdout/metrics.json`
- confusion: `outputs/evals/qwen36-27b-ibes-10k-holdout/confusion_matrix.csv`
- examples: `outputs/evals/qwen36-27b-ibes-10k-holdout/regression_examples.md`

Results vs base:

- base accuracy: `0.4490`
- 10k adapter accuracy: `1.0000`
- accuracy delta: `+0.5510`
- base macro F1: `0.5405`
- 10k adapter macro F1: `1.0000`
- macro F1 delta: `+0.4595`
- base parse failure rate: `0.5340`
- 10k adapter parse failure rate: `0.0000`
- parse failure delta: `-0.5340`
- 10k adapter exact JSON match rate: `0.9990`
- 10k adapter magnitude bucket accuracy: `0.9990`

Confusion summary:

- base confusion was dominated by invalid parses, especially neutral examples
- 10k adapter confusion was perfect on direction labels: `344` negative, `351` neutral, `305` positive all mapped correctly

Representative wins:

- `LSI|1996-02-26|O|1997-06-30`: base produced long prose and failed parsing; adapter returned valid JSON with `neutral` / `unknown`
- `LDW|1996-08-21|1|1996-08-31`: base emitted truncated reasoning text; adapter returned `positive` / `medium`
- `MARY|1997-01-08|2|1997-12-31`: base emitted prose despite correct underlying reasoning; adapter returned `negative` / `large`

There were no 10k-adapter regressions against base on this holdout.

### Prior 1k adapter on the same 10k holdout

To avoid paying another four-hour base pass, the 1k adapter was scored on the same 10k holdout while reusing the cached base predictions from the completed 10k-holdout run.

Artifacts:

- summary: `outputs/evals/qwen36-27b-ibes-1k-on-10k-holdout/eval_summary.md`
- metrics: `outputs/evals/qwen36-27b-ibes-1k-on-10k-holdout/metrics.json`
- confusion: `outputs/evals/qwen36-27b-ibes-1k-on-10k-holdout/confusion_matrix.csv`

1k adapter on 10k holdout:

- accuracy: `1.0000`
- macro F1: `1.0000`
- parse failure rate: `0.0000`
- exact JSON match rate: `0.9940`
- magnitude bucket accuracy: `0.9940`

10k minus 1k on the same 10k holdout:

- accuracy delta: `+0.0000`
- macro F1 delta: `+0.0000`
- parse failure delta: `+0.0000`
- exact JSON match delta: `+0.0050`
- magnitude bucket accuracy delta: `+0.0050`

Interpretation:

- scaling from 1k to 10k did not improve direction-label accuracy on the in-domain holdout because the 1k adapter was already saturated there
- scaling did slightly improve exact structured formatting fidelity and magnitude-bucket accuracy
- this is a real but small gain, not a breakthrough

### Public benchmark comparison

All public-task runs used non-thinking mode first.

Runs:

- `fiqa` 1k: `outputs/evals/qwen36-27b-fiqa-nonthinking`
- `fiqa` 10k: `outputs/evals/qwen36-27b-fiqa-10k-nonthinking`
- `fpb` 1k: `outputs/evals/qwen36-27b-fpb-nonthinking`
- `fpb` 10k: `outputs/evals/qwen36-27b-fpb-10k-nonthinking`
- `tfns` 1k: `outputs/evals/qwen36-27b-tfns-1k-nonthinking`
- `tfns` 10k: `outputs/evals/qwen36-27b-tfns-10k-nonthinking`
- `nwgi` 1k: `outputs/evals/qwen36-27b-nwgi-1k-nonthinking`
- `nwgi` 10k: `outputs/evals/qwen36-27b-nwgi-10k-nonthinking`

Results vs base:

- `fiqa`
  - base accuracy `0.7812`, 1k `0.8125`, 10k `0.7969`
  - base macro F1 `0.6491`, 1k `0.6759`, 10k `0.6597`
- `fpb`
  - base accuracy `0.4688`, 1k `0.7188`, 10k `0.6406`
  - base macro F1 `0.5229`, 1k `0.7553`, 10k `0.7125`
- `tfns`
  - base accuracy `0.5234`, 1k `0.7500`, 10k `0.6953`
  - base macro F1 `0.5581`, 1k `0.7419`, 10k `0.6970`
- `nwgi`
  - base accuracy `0.5234`, 1k `0.6094`, 10k `0.6250`
  - base macro F1 `0.5084`, 1k `0.6023`, 10k `0.6191`

Parse failure rate:

- every adapter run, both 1k and 10k, stayed at `0.0000`
- the gain here relative to base is stable, but scaling from 1k to 10k did not further improve parsing

10k minus 1k public-task deltas:

- `fiqa`: accuracy `-0.0156`, macro F1 `-0.0162`
- `fpb`: accuracy `-0.0781`, macro F1 `-0.0429`
- `tfns`: accuracy `-0.0547`, macro F1 `-0.0449`
- `nwgi`: accuracy `+0.0156`, macro F1 `+0.0169`

Representative 10k wins over 1k:

- `fpb-511`: gold `neutral`, 10k predicted `neutral`, 1k predicted `negative`
- `tfns-1957`: gold `positive`, 10k predicted `positive`, 1k predicted `neutral`
- `nwgi-3564`: gold `positive`, 10k predicted `positive`, 1k predicted `neutral`

Representative 10k regressions versus 1k:

- `fiqa-103`: gold `negative`, 10k predicted `neutral`, 1k predicted `negative`
- `fpb-563`: gold `neutral`, 10k predicted `positive`, 1k predicted `neutral`
- `tfns-1537`: gold `neutral`, 10k predicted `positive`, 1k predicted `neutral`
- `nwgi-624`: gold `neutral`, 10k predicted `positive`, 1k predicted `neutral`

Pattern in the regressions:

- the 10k adapter remained strong, but it became more willing to over-call weakly positive/transactional public headlines as `positive`
- this shows up most clearly in `fpb` and `tfns`
- `nwgi` was the only public task with a small 10k-over-1k improvement

## Controlled 10k conclusion

Did scaling improve the structured IBES task?

- yes, but only slightly beyond the already-saturated 1k adapter
- the 10k adapter matched the 1k adapter on direction-label accuracy and macro F1 on the 10k holdout
- it improved exact JSON match and magnitude-bucket accuracy by `0.0050`

Did scaling improve public finance benchmark generalization?

- mostly no
- relative to the 1k adapter, the 10k checkpoint regressed on `fiqa`, `fpb`, and `tfns`
- it improved only `nwgi`, and only by a small amount

Did scaling create regressions?

- yes, on public benchmarks
- no, on the structured 10k WRDS/IBES holdout
- the failure mode is narrower external generalization, not broken parsing or in-domain collapse

Should the next scale step be 50k?

- not yet
- the evidence does not justify a larger pure-IBES scaling push right now
- the 10k checkpoint already shows diminishing returns in-domain and mixed-to-worse public generalization
- the better next milestone is to stop scaling for the moment and shift effort toward the market-reaction layer once CRSP/link data is available

Recommended next step:

- do not start `50k` training yet
- keep the current non-thinking structured IBES adapter path
- prioritize CRSP daily returns plus the CRSP/Compustat link table so we can test whether these event labels correlate with realized market reactions
- optionally revisit `50k` later only if market-reaction work needs a better event-labeler and new evidence shows the current label quality is the bottleneck

## Classmate TODO

- CRSP daily returns
  - needed for event-window reaction labels and abnormal return calculations
- CRSP/Compustat link table
  - needed for reliable identifier joins from IBES events into return data
- Compustat fundamentals
  - optional, but useful for firm context once event/return joins work
- benchmark support
  - `tfns` and `nwgi` are now completed, so the remaining benchmark work is larger-slice or full-benchmark reruns rather than basic loader repair
