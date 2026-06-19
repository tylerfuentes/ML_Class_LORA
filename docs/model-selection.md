# Model Selection Summary

This is the current model-selection decision after the completed 10k IBES evaluation.

## Current best choices

- `1k` adapter: best general finance adapter so far
- `10k` adapter: best structured IBES JSON specialist so far
- `50k` training: not approved and not started

## Decision table

| Use case | Model | Why |
| --- | --- | --- |
| Public finance benchmarks such as `fiqa`, `fpb`, and `tfns` | `1k` adapter | best current accuracy and macro F1 across the broader public-task set |
| Narrow structured IBES classification with exact JSON and magnitude fidelity | `10k` adapter | matched perfect holdout direction accuracy and improved exact JSON plus magnitude fidelity versus the 1k adapter |
| Default adapter for broader research comparisons | `1k` adapter | better balance between structured finance competence and external generalization |
| Downstream market-reaction study | compare both `1k` and `10k` | the next stage should test whether narrow structured fidelity or broader generalization matters more for return-linked outcomes |

## Evidence summary

- On the `10k` IBES holdout, both adapters reached `1.0000` accuracy and `1.0000` macro F1.
- The `10k` adapter improved exact JSON match and magnitude-bucket accuracy slightly over the `1k` adapter.
- On public benchmarks, the `10k` adapter regressed versus the `1k` adapter on `fiqa`, `fpb`, and `tfns`.
- The `10k` adapter improved slightly on `nwgi`.

## Working conclusion

- Keep both adapters.
- Do not replace the `1k` adapter with the `10k` adapter as the repo default.
- Do not start `50k` pure-IBES training yet.
- Move the project to market-reaction measurement before any new training cycle.
