# Contributing to Mohawk

Contributions are welcome when they improve correctness, reproducibility, or
developer usability.

## Setup

```bash
git clone https://github.com/goombalab/mohawk.git
cd mohawk
python -m pip install -r requirements.txt
```

For dependency-light validation only:

```bash
python -m pip install pytest PyYAML
```

Keep credentials such as `HF_TOKEN` and `WANDB_API_KEY` in environment
variables. Do not commit tokens or machine-specific paths.

## Change Guidelines

- Keep changes focused on one concrete problem.
- Follow the existing module and YAML composition patterns.
- Put model code in `components/`, distillation logic in `distill/`, wrapper
  behavior in `training_wrapper/`, and shared helpers in `utils/`.
- Avoid unrelated refactors and generated artifacts.
- Document user-visible behavior changes.
- State exactly what was tested and what remains untested.

## Validation

Run the dependency-light checks for every change:

```bash
python3 -m compileall -q .
pytest -q
```

Also run a focused command for the behavior you changed. Examples:

```bash
# Training
python run.py --config <config>

# lm-eval benchmark
python evals/benchmark.py --dir <checkpoint-or-model> --tasks <tasks>

# Generation
python generation/generate.py --model <checkpoint-or-model> --prompt <text>
```

GPU, distributed, remote-data, and optional-kernel changes require an
appropriate targeted run. If that environment is unavailable, say so in the
pull request instead of implying coverage.

## Pull Requests

Include:

- **Problem**: what was broken or missing.
- **Change**: what the patch does.
- **Risk**: compatibility concerns or likely failure modes.
- **Validation**: commands run and their outcomes.

Do not include unrelated formatting, metadata churn, local caches, logs,
checkpoints, or generated data.

## Bug Reports

Provide:

- The exact command and config.
- The complete traceback.
- Relevant Python, PyTorch, Transformers, CUDA, and GPU versions.
- Whether the failure reproduces consistently.

## License

Contributions are licensed under the repository's MIT license.
