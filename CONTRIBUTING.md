# Contributing to Mohawk

This repository is used for research code and experiments around Mohawk distillation. Contributions are welcome when they improve correctness, reproducibility, or developer usability.

## Before You Start

Set up a local environment and run the dependency-light CPU baseline:

```bash
git clone https://github.com/goombalab/mohawk.git
cd mohawk
python -m pip install pytest PyYAML
python3 -m compileall -q .
pytest -q
```

This is only the dependency-light CPU baseline. The default runtime and
evaluation manifest is now:

```bash
python -m pip install -r requirements.txt
```

That default manifest was validated in a fresh Python 3.12 venv on 2026-06-26:
it installed `torch 2.12.1+cu130`, `transformers 4.55.4`, and `lm_eval 0.4.11`,
then passed the full default pytest suite with `103 passed, 1 skipped`. It does
not install the CUDA-built SSM/Mamba kernels. For those, use
`requirements-ssm-cuda.txt` only after installing `requirements.txt`, and only
on a CUDA-devel host with a compatible driver, CUDA toolkit/nvcc, and compiler
toolchain:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -r requirements-ssm-cuda.txt
```

That optional path is still resource-gated in `PUBLIC_SUPPORT_MATRIX.md`.

See `PUBLIC_SUPPORT_MATRIX.md` for the strict scenario statuses:
`PASSED_END_TO_END`, `PASSED_REAL_SMOKE`, `FAILED_FIXED`, `FAILED_UNFIXED`,
`BLOCKED_RESOURCE`, `DOWNGRADED_DOCS`, and `REMOVED_OR_UNSUPPORTED`. In PR
validation notes, do not claim a scenario works unless the actual intended code
path ran with real dependencies, models/checkpoints, data, and required
hardware.

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

The repo has a CPU-safe CI workflow, but no full GPU/distributed CI gate yet;
include the strongest checks you can run locally:

1. Syntax check:
   ```bash
   python3 -m compileall -q .
   ```
2. A targeted functional run relevant to your change:
   - Training path change: run `python run.py --config <config>` only in
     a CUDA-capable environment with required datasets/checkpoints, or state
     that full training validation is resource-gated. For entrypoint-level
     regressions, the tiny CPU smoke may be used with
     `MOHAWK_ALLOW_CPU_TRAINING=1` after creating its local checkpoint with
     `python tools/create_tiny_smoke_checkpoint.py --config
     configs/smoke/tiny_cpu_supervised.yaml --output
     /tmp/mohawk_tiny_teacher_ckpt`; this does not validate CUDA, DDP, FSDP,
     SSM/Mamba components, representative configs, or full-size training.
     For changes to the Hugging Face teacher-loading path, run the tiny public
     teacher smoke with `MOHAWK_ALLOW_CPU_TRAINING=1 python run.py --config
     configs/smoke/tiny_cpu_hf_teacher_supervised.yaml`. The first run needs
     network access to populate the config's isolated `/tmp` HF cache; after
     that, rerun with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. This does not
     validate large or gated teachers, CUDA, DDP/FSDP, SSM/Mamba, or
     representative checkpoints.
     For changes to the public Hugging Face dataset training path, run
     `MOHAWK_ALLOW_CPU_TRAINING=1 python run.py --config
     configs/smoke/tiny_cpu_public_hfdata_supervised.yaml`. The first run needs
     network access to populate the config's isolated `/tmp` dataset and model
     caches for `NeelNanda/pile-10k` and `sshleifer/tiny-gpt2`; after that,
     rerun with `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
     TRANSFORMERS_OFFLINE=1`. This does not validate large or gated datasets,
     default-buffer streaming throughput, distributed sharding, CUDA,
     DDP/FSDP, or representative checkpoints.
     For launch-path changes, also run the single-process `torchrun` smoke with
     `MOHAWK_ALLOW_CPU_TRAINING=1 torchrun --standalone --nproc_per_node=1
     run.py --config configs/smoke/tiny_cpu_supervised.yaml` in an environment
     that permits local rendezvous sockets. Do not treat multi-process CPU
     `torchrun` as DDP/FSDP validation: the audited two-process CPU probe
     reached `run.py` and failed with `RuntimeError: Mohawk training requires
     CUDA` before DDP/FSDP setup.
     On a Slurm node with visible CUDA GPUs, create the tiny CUDA teacher with
     `python tools/create_tiny_smoke_checkpoint.py --config
     configs/smoke/tiny_cuda_supervised.yaml --output
     /home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_teacher_ckpt`, then run
     the one-process CUDA launch smoke with `srun -p batch -A
     nemotron_arch_dev --qos=interactive --gpus-per-node=4 --time=00:06:00
     --immediate=1 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
     WANDB_MODE=disabled
     /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun
     --standalone --nproc_per_node=1 run.py
     --config configs/smoke/tiny_cuda_supervised.yaml`.
     For centralized CUDA mixed-precision and resume changes, run
     `configs/smoke/tiny_cuda_mixed_precision.yaml`,
     `configs/smoke/tiny_cuda_resume.yaml`, and
     `configs/smoke/tiny_cuda_mixed_precision_resume.yaml` through the same
     Slurm allocation using
     `/home/abick/nemotron_abick/conda/envs/fla-raven/bin/python run.py`.
     For DDP/FSDP wrapper changes, run the audited tiny CUDA smokes with
     `configs/smoke/tiny_cuda_ddp_supervised.yaml`,
     `configs/smoke/tiny_cuda_fsdp_supervised.yaml`, and
     `configs/smoke/tiny_cuda_fsdp_activation_checkpointing.yaml` under
     `/home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun
     --standalone --nproc_per_node=2`. For mixed-precision wrapper changes,
     also run `configs/smoke/tiny_cuda_ddp_mixed_precision.yaml` and
     `configs/smoke/tiny_cuda_fsdp_mixed_precision.yaml`. For distributed
     resume changes, also run `configs/smoke/tiny_cuda_ddp_resume.yaml` and
     `configs/smoke/tiny_cuda_fsdp_resume.yaml`. The DDP config uses
     `configs/smoke/data_ddp` so each rank receives a JSON shard. These smokes
     validate only tiny one-node CUDA/DDP/FSDP paths; representative configs,
     multi-node training, SSM/Mamba kernels, large or gated data, mixed-precision
     distributed resume, and full-size checkpoints remain resource-gated.
     For sequential-entrypoint changes, run the tiny comma-separated smoke:
     `MOHAWK_ALLOW_CPU_TRAINING=1 python run.py --config
     configs/smoke/tiny_cpu_supervised.yaml,configs/smoke/tiny_cpu_supervised.yaml`.
     When Slurm CUDA resources are available, also run the tiny CUDA train+eval
     sequence: `/home/abick/nemotron_abick/conda/envs/fla-raven/bin/python run.py
     --config
     configs/smoke/tiny_cuda_comma_supervised.yaml,configs/smoke/tiny_cuda_comma_eval_ppl.yaml`
     inside the documented `srun` allocation.
     For Qwen2 model-family entrypoint changes, create
     `/tmp/mohawk_tiny_qwen2_teacher_ckpt` with
     `configs/smoke/tiny_cpu_qwen2_supervised.yaml`, then run
     `MOHAWK_ALLOW_CPU_TRAINING=1 python run.py --config
     configs/smoke/tiny_cpu_qwen2_supervised.yaml`; this does not validate
     full-size Qwen2 configs, CUDA, DDP/FSDP, SSM/Mamba, Phi/Falcon
     entrypoints, or representative checkpoints.
     For PhiAttention model-family entrypoint changes, create
     `/tmp/mohawk_tiny_phi_teacher_ckpt` with
     `configs/smoke/tiny_cpu_phi_supervised.yaml`, then run
     `MOHAWK_ALLOW_CPU_TRAINING=1 python run.py --config
     configs/smoke/tiny_cpu_phi_supervised.yaml`; this does not validate
     full-size Phi configs, CUDA, DDP/FSDP, SSM/Mamba package dependencies, or
     representative checkpoints.
     For FalconBlock model-family entrypoint changes, create
     `/tmp/mohawk_tiny_falcon_teacher_ckpt` with
     `configs/smoke/tiny_cpu_falcon_supervised.yaml`, then run
     `MOHAWK_ALLOW_CPU_TRAINING=1 python run.py --config
     configs/smoke/tiny_cpu_falcon_supervised.yaml`; this uses Transformers'
     sequential Mamba fallback and does not validate full-size Falcon configs,
     CUDA, DDP/FSDP, fast Mamba kernels, SSM/Mamba package dependencies, or
     representative checkpoints.
     For DoubleBlock block-family changes, create the matching tiny teacher
     checkpoint with `tools/create_tiny_smoke_checkpoint.py --config
     configs/smoke/tiny_cpu_doubleblock_<variant>_supervised.yaml --output
     /tmp/mohawk_tiny_doubleblock_<variant>_teacher_ckpt`, then run
     `MOHAWK_ALLOW_CPU_TRAINING=1 python run.py --config
     configs/smoke/tiny_cpu_doubleblock_<variant>_supervised.yaml` for
     `adapter`, `vanilla`, `hymba`, and `merger`. These validate only tiny CPU
     attention-only DoubleBlock entrypoint paths with `LlamaAttention` as both
     mixers; they do not validate SSM/Mamba mixers, production Qwen/Llama
     hybrid configs, CUDA, DDP/FSDP, or representative checkpoints.
     For distillation-objective changes, the hstates entrypoint smoke may be
     run after creating its checkpoint with `python
     tools/create_tiny_smoke_checkpoint.py --config
     configs/smoke/tiny_cpu_hstates.yaml --output
     /tmp/mohawk_tiny_hstates_teacher_ckpt`, then
     `MOHAWK_ALLOW_CPU_TRAINING=1 python run.py --config
     configs/smoke/tiny_cpu_hstates.yaml`; this does not validate the
     remaining objective types through `run.py`. For sequential-hstates changes,
     create `/tmp/mohawk_tiny_sequential_hstates_teacher_ckpt` with
     `configs/smoke/tiny_cpu_sequential_hstates.yaml`, then run
     `MOHAWK_ALLOW_CPU_TRAINING=1 python run.py --config
     configs/smoke/tiny_cpu_sequential_hstates.yaml`. For matrices-objective changes,
     create `/tmp/mohawk_tiny_matrices_teacher_ckpt` with
     `configs/smoke/tiny_cpu_matrices.yaml`, then run
     `MOHAWK_ALLOW_CPU_TRAINING=1 python run.py --config
     configs/smoke/tiny_cpu_matrices.yaml`; this still does not validate DPO or
     instruction distillation through `run.py`. For DPO changes, create
     `/tmp/mohawk_tiny_dpo_teacher_ckpt` with
     `configs/smoke/tiny_cpu_dpo.yaml`, then run
     `MOHAWK_ALLOW_CPU_TRAINING=1 python run.py --config
     configs/smoke/tiny_cpu_dpo.yaml`; this uses a tiny local preference JSON
     and does not validate representative preference datasets. For
     supervised-instruction changes, create
     `/tmp/mohawk_tiny_supervised_instruct_teacher_ckpt` with
     `configs/smoke/tiny_cpu_supervised_instruct.yaml`, then run
     `MOHAWK_ALLOW_CPU_TRAINING=1 python run.py --config
     configs/smoke/tiny_cpu_supervised_instruct.yaml`; this uses a tiny local
     instruction JSON and does not validate chat-template tokenizers or
     generation-supervision mode.
     For checkpoint-resume changes, first produce
     `/tmp/mohawk_tiny_cpu_run/save` with the supervised smoke, then run
     `MOHAWK_ALLOW_CPU_TRAINING=1 python run.py --config
     configs/smoke/tiny_cpu_resume.yaml`; for CUDA/DDP/FSDP resume behavior,
     use the Slurm smokes above. The CPU smoke does not validate CUDA,
     DDP/FSDP, mixed precision resume, or representative checkpoints.
     For lazy-init checkpoint-loading changes, first produce
     `/tmp/mohawk_tiny_cpu_run/save` with the supervised smoke, then run
     `MOHAWK_ALLOW_CPU_TRAINING=1 python run.py --config
     configs/smoke/tiny_cpu_lazy_load.yaml`; this validates centralized CPU
     lazy model initialization only, not optimizer/scheduler resume, CUDA,
     DDP/FSDP, mixed precision lazy init, or representative checkpoints.
   - Benchmark path change: run the tiny public HF smoke when possible:
     `python evals/benchmark.py --dir sshleifer/tiny-gpt2 --tasks wikitext
     --batch_size 1 --limit 1`. After both the model and task dataset are
     cached, rerun with `--local_files_only` if you need to validate offline
     behavior. A first local-only run fails when the lm-eval task dataset cache
     is absent.
   - Mohawk checkpoint benchmark change: run `python evals/benchmark.py
     --backend mohawk --dir <checkpoint_dir> --tasks <task_list>` only with a
     real checkpoint/tokenizer and compatible runtime/GPU resources, or state
     that it was not verified. For entrypoint-level checkpoint benchmark
     regressions, the tiny CPU checkpoint produced by
     `configs/smoke/tiny_cpu_supervised.yaml` may be used with
     `HF_DATASETS_CACHE=/tmp/mohawk_lm_eval_datasets python evals/benchmark.py
     --backend mohawk --dir /tmp/mohawk_tiny_cpu_run/save --tasks wikitext
     --batch_size 1 --limit 1 --device cpu`; this does not validate
     representative checkpoints, CUDA, DDP/FSDP, or gated models/data.
   - Perplexity change: run the tiny HF PPL command from the README, and for
     Mohawk checkpoint wrapper changes run `python evals/eval_ppl.py --backend
     mohawk --model /tmp/mohawk_tiny_cpu_run/save --local_files_only --device
     cpu --n_batches 1 --max_seq_len 16 --batch_size 1 --text "hello world
     from mohawk"` after producing the tiny checkpoint; this does not validate
     representative checkpoints or GPU/distributed PPL.
   - Generation/util change: run the corresponding script with a minimal input.
     For generation loading changes, run a tiny HF command such as
     `python generation/generate.py --model sshleifer/tiny-gpt2 --prompt hello
     --genlen 2 --repeats 1 --top_k 1`, and after producing the tiny CPU
     checkpoint run `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python
     generation/generate.py --model /tmp/mohawk_tiny_cpu_run/save
     --local_files_only --prompt hello --genlen 2 --repeats 1 --top_k 1`.
     For hybrid-weight-transfer changes, run
     `WANDB_MODE=disabled python tools/hybrid_weights_transfer.py --config
     configs/smoke/tiny_cpu_hybrid_transfer.yaml --heads 0:0 --device cpu
     --allow-unexpected-student-load`. The first run needs network access to
     populate the config's isolated `/tmp` HF cache for
     `hf-internal-testing/tiny-random-LlamaForCausalLM`; after that, rerun with
     `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. This only validates the tiny
     CPU public-teacher attention-transfer path, not SSM/Mamba transfer,
     production Qwen/Llama hybrid configs, CUDA, DDP/FSDP, or representative
     checkpoints.
     State any optional dependency, model asset, CUDA, or representative
     checkpoint blocker that prevents a broader run.
3. Include command lines and outcomes in the PR description.

If you cannot run GPU-dependent validation, say that explicitly and link the
relevant blocker or
[release-readiness gate](PUBLIC_SUPPORT_MATRIX.md#release-readiness-gates).

## Pull Request Template

Use this structure in your PR body:

- **Problem**: what was broken or missing.
- **Change**: what you implemented.
- **Risk**: likely failure modes or compatibility concerns.
- **Validation**: exact commands run, key output, and any scenarios not run or
  left resource-gated.

## Reporting Bugs

Open an issue with:

- Exact command/config used
- Full traceback
- Hardware/software context (GPU, CUDA, torch, transformers versions)
- Whether the failure is deterministic

Issues without reproducible steps are hard to act on and may be closed until more detail is provided.

## License

By contributing, you agree that contributions are licensed under MIT (same as this repo).
