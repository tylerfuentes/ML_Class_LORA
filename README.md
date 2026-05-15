# ML Class LoRA

This project focuses on fine-tuning a strong base model so it performs better on financial tasks that are easy to evaluate and explain.

The core idea is to improve how the model works with structured financial inputs, recognizes patterns, and produces more reliable outputs for realistic finance scenarios. We will compare the fine-tuned model against the base model using clear metrics and concrete examples, then translate the results into a business-facing story such as faster analysis, more consistent outputs, or stronger decision support.

We are running the project on a dedicated NVIDIA DGX Spark, which keeps the workflow local, avoids cloud cost overhead, and gives the team a shared environment for training, evaluation, and presentation work.

## Project goals

- Fine-tune a base model with LoRA for finance-oriented tasks.
- Measure improvement against the untuned baseline.
- Keep the workflow simple enough for a class project and clear enough for presentation.
- Split work cleanly across a four-person team using GitHub, SSH, and VS Code Remote SSH.

## Team

- Finn Kliewer
- Tyler Fuentes
- Richie Gray
- Om Patel
- Nathanael Gill

Suggested Linux usernames for DGX access:

- use each teammate's Cornell NetID as their Linux username

## Planned workflow

1. Set up teammate SSH access to the DGX.
2. Use VS Code Remote SSH for shared development on the box.
3. Prepare training, evaluation, and reporting code in this repo.
4. Fine-tune the selected base model with LoRA.
5. Evaluate base vs. tuned behavior on finance-focused tasks.
6. Turn results into a short technical and business presentation.

## Repository status

The repository currently contains:

- SSH onboarding and machine-access tooling
- a repo-local QLoRA training scaffold for `Qwen/Qwen3.6-27B`
- a pinned `FinGPT` submodule under `external/FinGPT/`
- small public sample data and export scripts for FinGPT / SEC demos

- Teammate getting-started guide: `docs/getting-started.md`
- Team SSH key intake file: `docs/team-ssh-keys.md`
- Admin guide: `docs/ssh-onboarding.md`
- Teammate intake template: `docs/teammate-key-request.md`
- Shared repo permissions: `scripts/configure_shared_repo.sh`
- Per-user provisioning: `scripts/provision_teammate.sh`

Training and data integration notes now live in:

- `docs/training-setup.md`
- `docs/fingpt-integration.md`
- `docs/fingpt-conversion.md`

The next project step is building the first clean finance baseline dataset and comparing base Qwen vs a small finance adapter.

## SSH onboarding

This repo is set up for separate Linux accounts per teammate rather than a shared login. That keeps access cleaner and works well with VS Code Remote SSH.

Machine-specific relay values are intentionally not tracked in this repo. Keep current host, port, and hardening notes in ignored local files under `admin/local/`.

Provisioning details live in:

- `docs/getting-started.md`
- `docs/team-ssh-keys.md`
- `docs/ssh-onboarding.md`
- `docs/teammate-key-request.md`
- `admin/LOCAL_OVERRIDES.md`

## Expected project structure

As the project is built out, this repo will likely hold:

- training scripts
- evaluation scripts
- experiment notes
- dataset preparation code
- presentation-ready results and examples

## DGX usage

The DGX Spark will act as the shared execution environment for:

- model access
- LoRA fine-tuning
- evaluation runs
- collaborative development through SSH and VS Code Remote SSH

To avoid collisions, the team should coordinate training windows and keep experiment outputs organized by run name and owner.
