# Contributing to Mohawk

This repository is used for research code and experiments around Mohawk distillation. Contributions are welcome when they improve correctness, reproducibility, or developer usability.

## Before You Start

Set up a local environment and confirm imports work:

```bash
git clone https://github.com/your-username/mohawk.git
cd mohawk
pip install -r requirements.txt
python3 -m compileall -q .
```

If your change touches model loading or Hugging Face assets, set `HF_TOKEN` in your shell instead of editing config files with credentials.

## What Makes a Good PR Here

- Solves a concrete problem (bug, missing behavior, poor ergonomics) with minimal unrelated refactoring.
- Includes updates to docs/config comments when behavior changes.
- Uses existing architecture/config patterns unless there is a clear reason to introduce a new one.
- States exactly what was tested and what was not tested.

## Code and Config Expectations

- Keep architecture code in `components/`, training flow in `distill/` or `training_wrapper/`, and utilities in `utils/`.
- Prefer extending existing YAML config composition (`LOAD`) over duplicating large config files.
- Avoid hardcoding machine-specific paths, tokens, or cluster assumptions.
- Add assertions/checks when shape or dtype mismatches can silently corrupt training.

## Testing Expectations

There is no single full CI pipeline in this repo yet, so include the strongest checks you can run locally:

1. Syntax/import check:
   ```bash
   python3 -m compileall -q .
   ```
2. A targeted functional run relevant to your change:
   - Training path change: run `python run.py --config <config>`
   - Benchmark path change: run `python evals/benchmark.py --dir <model_dir> --tasks <task_list>`
   - Generation/util change: run the corresponding script with a minimal input
3. Include command lines and outcomes in the PR description.

If you cannot run GPU-dependent validation, say that explicitly.

## Pull Request Template

Use this structure in your PR body:

- **Problem**: what was broken or missing.
- **Change**: what you implemented.
- **Risk**: likely failure modes or compatibility concerns.
- **Validation**: exact commands run and key output.

## Reporting Bugs

Open an issue with:

- Exact command/config used
- Full traceback
- Hardware/software context (GPU, CUDA, torch, transformers versions)
- Whether the failure is deterministic

Issues without reproducible steps are hard to act on and may be closed until more detail is provided.

## License

By contributing, you agree that contributions are licensed under MIT (same as this repo).
