# Research Directions

This document records the research-backed direction for the first finance QLoRA baseline and the likely next-stage RL/reasoning work.

## 1. Reasoning-aware finance supervision

One useful source for reasoning-aware finance annotations is:

- Kabalisa, B. (2026). *SenseAI: A Human-in-the-Loop Dataset for RLHF-Aligned Financial Sentiment Reasoning*. arXiv:2604.05135.
  https://arxiv.org/abs/2604.05135

Why it matters for this project:

- SenseAI is explicitly framed as a financial sentiment and reasoning dataset.
- It uses human-in-the-loop correction rather than raw synthetic labeling alone.
- It is a good candidate for style/alignment data when we want the model to produce finance-specific reasoning that is more grounded than generic synthetic explanation traces.

Important limitation:

- SenseAI should be treated as a reasoning-style and annotation-quality reference, not as a substitute for market-truth labels from WRDS. The repo baseline still needs objective market outcome supervision.

## 2. WRDS-grounded event pipeline

The first serious dataset direction for this repo is:

- SEC 8-K triggering events for the event text and timestamp
- TAQ or other market data for post-event price response
- RavenPack, if Cornell access includes it, for structured event relevance and sentiment features
- IBES for analyst expectation and revision context

Project interpretation:

- The model should not only learn to summarize news or filings.
- It should learn to connect a filing or event to a market-relevant outcome label.
- That means the "truth" is not only textual explanation quality but also the historical market reaction and expectation context.

Recommended order:

1. Build `8-K -> event timestamp -> return label` first.
2. Add RavenPack-style structured sentiment features if available.
3. Add IBES surprise/revision context for earnings-sensitive examples.

## 3. Avoiding reasoning drift

For later-stage RL or reward-tuned reasoning work, useful references include:

- Hatamizadeh et al. (2026). *iGRPO: Self-Feedback-Driven LLM Reasoning*. arXiv:2602.09000.
  https://arxiv.org/abs/2602.09000
- Ahmadi et al. (2026). *Enhanced LLM Reasoning by Optimizing Reward Functions with Search-Driven Reinforcement Learning*. arXiv:2605.02073.
  https://arxiv.org/abs/2605.02073

What these papers support:

- GRPO-style optimization is a plausible next step for improving reasoning quality once the supervised QLoRA baseline is stable.
- Reward design matters a lot.
- A model can be improved further by tying updates to an explicit reward signal rather than only next-token imitation.

What is still our project inference:

- Using WRDS-derived price moves, abnormal returns, or earnings-surprise outcomes as the finance reward signal is our proposed application, not a claim directly established by the papers above.
- So the repo should treat this as a next-stage experimental plan, not as a proven result.

## 4. Repo implication

For this repository, the practical plan is:

- Keep the current `800/100/100` QLoRA baseline small and clean.
- Build the first real examples around event-driven finance tasks, ideally with 8-K-centered supervision.
- Use SenseAI-style reasoning quality as inspiration for annotation format.
- Use WRDS data as the factual grounding layer.
- Only consider GRPO-style post-training after the supervised adapter trains, saves, and reloads cleanly.
