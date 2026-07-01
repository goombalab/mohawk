# Knowledge Distillation for Hybrid Transformer-SSM Models

A PyTorch framework for knowledge distillation from large language models (LLMs) to hybrid architectures that combine Transformer attention mechanisms with State-Space Models (SSM), such as Mamba.

## What This Repository Contains

- Model building blocks for Llama, Qwen2, Falcon, and Phi-style hybrids.
- A composable YAML config system (`LOAD`-based inheritance).
- Distillation objectives: `supervised`, `hstates`, `matrices`, and `dpo`.
- Distributed training wrappers (DDP/FSDP/centralized).
- Evaluation utilities for perplexity and lm-eval-harness tasks.

These bullets describe code surfaces in the repository; they are not a claim
that every surface is fully validated in a fresh public checkout.

This is not a packaged library. You run scripts directly from the repo.

## Support Status

The current real-validation matrix uses strict scenario statuses:
`PASSED_END_TO_END`, `PASSED_REAL_SMOKE`, `FAILED_FIXED`, `FAILED_UNFIXED`,
`BLOCKED_RESOURCE`, `DOWNGRADED_DOCS`, and `REMOVED_OR_UNSUPPORTED`.

Only a small set of dependency-light, tiny-model, public-data, or tiny
CUDA/DDP/FSDP paths currently have real smoke evidence. Tiny Mamba2/SSM fast
kernels now have centralized, DDP, and FSDP CUDA smoke evidence on GB300. One
cached 424,303,808-parameter custom Transformers checkpoint now also has
bounded GB300 generation, perplexity, lm-eval, and hidden-state visualization
evidence. Full Mohawk training, representative distributed or SSM training,
large/gated remote datasets, gated models, and representative Mohawk hybrid
checkpoints are not currently proven. See
[PUBLIC_SUPPORT_MATRIX.md](PUBLIC_SUPPORT_MATRIX.md) for the evidence, latest
commands, and [release readiness gates](PUBLIC_SUPPORT_MATRIX.md#release-readiness-gates).

## Setup

### Requirements

- Python 3.8+

The row-scoped dependency-light CPU baseline validated by this audit needs only
Python plus `pytest` and `PyYAML`; it does not require CUDA, PyTorch, model
weights, or datasets. Full training/eval additionally requires:

- PyTorch 2.1+ with CUDA
- One or more NVIDIA GPUs for training/eval

### Install

For the row-scoped CPU-safe baseline validated by this audit:

```bash
git clone https://github.com/goombalab/mohawk.git
cd mohawk
python -m pip install pytest PyYAML
```

For the default Python runtime and evaluation stack validated by this audit:

```bash
python -m pip install -r requirements.txt
```

This default manifest deliberately excludes the CUDA-built SSM/Mamba kernels.
A fresh Python 3.12 venv on 2026-06-26 installed `requirements.txt`, imported
`torch 2.12.1+cu130`, `transformers 4.55.4`, and `lm_eval 0.4.11`, confirmed
that `mamba_ssm` and `causal_conv1d` were absent, then passed the full default
pytest suite with `103 passed, 1 skipped`. The local host still had no visible
CUDA device. A storage-backed default-runtime venv using the same default
manifest later ran `configs/smoke/tiny_cuda_default_runtime_supervised.yaml` on
a Slurm GB300 allocation and completed the tiny single-process CUDA
`Tiny-CUDA-Default-Runtime-Supervised-Smoke` training path. The same
storage-backed venv also ran one-node two-rank
`Tiny-CUDA-DDP-Default-Runtime-Supervised-Smoke` and
`Tiny-CUDA-FSDP-Default-Runtime-Supervised-Smoke` training paths, plus
two-node four-rank `Tiny-CUDA-DDP-Default-Runtime-Multinode-Gradient-Accumulation-Smoke`
and `Tiny-CUDA-FSDP-Default-Runtime-Multinode-Gradient-Accumulation-Smoke`
training paths, plus
one-node two-rank `Tiny-CUDA-DDP-Default-Runtime-Mixed-Precision-Smoke` and
`Tiny-CUDA-FSDP-Default-Runtime-Mixed-Precision-Smoke` training paths, and
one-node two-rank `Tiny-CUDA-DDP-Default-Runtime-BFloat16-Smoke` and
`Tiny-CUDA-FSDP-Default-Runtime-BFloat16-Smoke` training paths, plus
two-node four-rank
`Tiny-CUDA-DDP-Default-Runtime-Multinode-Mixed-Precision-Gradient-Accumulation-Smoke`,
`Tiny-CUDA-FSDP-Default-Runtime-Multinode-Mixed-Precision-Gradient-Accumulation-Smoke`,
`Tiny-CUDA-DDP-Default-Runtime-Multinode-BFloat16-Gradient-Accumulation-Smoke`,
and
`Tiny-CUDA-FSDP-Default-Runtime-Multinode-BFloat16-Gradient-Accumulation-Smoke`
training paths, and one-node
two-rank `Tiny-CUDA-DDP-Default-Runtime-Resume-Smoke` and
`Tiny-CUDA-FSDP-Default-Runtime-Resume-Smoke` checkpoint resume paths, plus
two-node four-rank `Tiny-CUDA-DDP-Default-Runtime-Multinode-Resume-Smoke` and
`Tiny-CUDA-FSDP-Default-Runtime-Multinode-Resume-Smoke` checkpoint resume
paths, plus
one-node two-rank DDP/FSDP `eval_hstates`, `eval_ppl`, and benchmark callback
paths, plus two-node four-rank DDP/FSDP `eval_hstates`, `eval_ppl`, and
benchmark callback paths. This proves only the default install/test surface plus
storage-backed tiny single-process, DDP/FSDP, one-node and multi-node DDP/FSDP
mixed-precision, one-node and multi-node DDP/FSDP bfloat16, one-node and
multi-node DDP/FSDP checkpoint-resume, and DDP/FSDP eval-callback CUDA smokes,
not fresh `/tmp` Slurm visibility, repo-local shared fresh installs, or
representative/full-scale SSM kernels.

For full SSM/Mamba research runs on a CUDA-devel host, install the optional
kernel layer after the default runtime:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -r requirements-ssm-cuda.txt
```

The optional manifest pins the validated Mamba 2.3 operator stack:
`mamba-ssm==2.3.2.post1`, `causal-conv1d==1.6.2.post1`, and the tested
`quack-kernels==0.5.3` dependency. Mamba 2.3 adds TileLang/Quack/CUTLASS
dependencies and previously exposed a top-level `utils` namespace collision.
Mohawk guards direct component imports by making the local `utils` package
explicit and moving the repository root ahead of site-packages before local
utility imports.

Source builds need a compatible CUDA toolkit and target architecture. For the
validated aarch64 GB300 environment, the build used CUDA 13.0 and:

```bash
export CUDA_HOME=/path/to/cuda-13.0
export TORCH_CUDA_ARCH_LIST=10.0
export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
python -m pip install --no-build-isolation -r requirements-ssm-cuda.txt
```

The earlier storage-backed Python 3.12 venv at
`/home/abick/storage/mohawk/venvs/ssm-cuda-20260626` established the CUDA 13.0
source-build path because the cluster does not expose a system `nvcc`.
Official compiler, CRT, NVVM, and CCCL wheels produced aarch64 `sm_100` wheels
for causal-conv1d and Mamba 2.2. A separate venv created from scratch at
`/home/abick/storage/mohawk/venvs/ssm-cuda-mamba23-20260630` installed the
default manifest followed by the Mamba `2.3.2.post1` wheel (SHA-256
`e3e6115fdb969130881fff5504c20b58398bf6a200a0d3359a5ae0ab9949c70e`),
causal-conv1d, and their declared dependencies. On a GB300, that fresh venv
imported Mohawk with the repository `utils` package, ran finite bfloat16 fast
forward/backward, completed a real two-step float32 `run.py` training smoke,
and generated from its saved checkpoint. This remains one architecture and
tiny-model evidence; it does not prove portable wheels, production hybrid
configs, representative Mamba 2.3 sequence lengths, multi-node Mamba 2.3, or
representative checkpoints.

Optional accelerators:

```bash
pip install flash-attn --no-build-isolation
```

This optional accelerator install was not part of the public validation suite;
run it only in a CUDA/build-compatible environment.

Some research utilities have additional optional dependencies that are not
needed for the CPU-safe validation suite. See
[PUBLIC_SUPPORT_MATRIX.md](PUBLIC_SUPPORT_MATRIX.md) before running GPU,
benchmark, generation, or visualization workflows.

### Environment Variables

Use environment variables for credentials instead of hardcoding:

- `HF_TOKEN` for private/gated Hugging Face models
- `WANDB_API_KEY` for online experiment tracking
- `CUDA_VISIBLE_DEVICES` to pin GPUs

Global runtime defaults live in `configs/management.yaml`. Public smoke runs
default to `WANDB_MODE=disabled`; set `WANDB_MODE=online` and provide
`WANDB_API_KEY` plus a real W&B entity for online logging.

## Quick Start

### Validate a Fresh Checkout

After installing the lightweight dependencies above, these commands are the
CPU-safe validation path and do not require downloaded model weights or
datasets:

```bash
python run.py --help
python evals/benchmark.py --help
python generation/generate.py --help
python3 -m compileall -q .
pytest -q
```

This is only a dependency-light checkout smoke. It does not verify full
training, CUDA execution, distributed wrappers, remote datasets, gated model
access, Mamba/SSM kernels, or the research utility scripts. See
[PUBLIC_SUPPORT_MATRIX.md](PUBLIC_SUPPORT_MATRIX.md) for the strict scenario
statuses, and use the
[release readiness gates](PUBLIC_SUPPORT_MATRIX.md#release-readiness-gates)
before making any broader support claim.

### Training Templates

Training requires NVIDIA CUDA, PyTorch with CUDA, Hugging Face access for the
teacher model/datasets, and a real checkpoint path wherever a config contains
`<path-to-checkpoint>`. Most configs are research templates rather than
turnkey public runs.

Single-process launch:

```bash
WANDB_MODE=disabled python run.py --config configs/Llama/1B/hybrid/mohawk_8.yaml
```

Multi-GPU launch:

```bash
WANDB_MODE=disabled torchrun --standalone --nproc_per_node=8 run.py \
  --config configs/Llama/1B/hybrid/mohawk_8.yaml
```

Before running either command, replace placeholder checkpoint paths in the
selected config or load chain. `--config` also accepts a comma-separated list;
the entrypoint's config-list parsing and per-config environment restoration are
smoke-tested, but full sequential training remains blocked by CUDA, data, and
checkpoint resources. A representative Slurm probe for
`configs/Llama/8B/llama/hstates.yaml` resolves the full Llama 8B hstates config
but currently fails before model initialization because `HuggingFaceFW/fineweb-edu`
is not available in offline/cache-only mode.

Tiny CPU `run.py` smoke for entrypoint regressions:

```bash
python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cpu_supervised.yaml \
  --output /tmp/mohawk_tiny_teacher_ckpt

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py --config configs/smoke/tiny_cpu_supervised.yaml
```

This scoped smoke runs the real supervised `run.py` path on a tiny local JSON
dataset and writes a checkpoint under `/tmp/mohawk_tiny_cpu_run/save`. It does
not validate CUDA training, DDP, FSDP, Mamba/SSM components, large or gated
datasets, representative research configs, or full-size checkpoints.

Tiny CPU gradient-accumulation smoke after creating the same local teacher
checkpoint:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_gradient_accumulation.yaml
```

This validates only centralized tiny CPU micro-batch accumulation through the
real supervised `run.py` path. The smoke uses `effective_batch_size: 2` with
`TorchDataLoader.batch_size: 1`, computes `accumulation_steps: 2` and
`n_batches: 4`, then writes `/tmp/mohawk_tiny_gradient_accumulation_cpu_run/save`.
Distributed no-sync coverage is separate; this CPU-only smoke does not validate
distributed no-sync, representative configs, or full-size checkpoints.

Tiny CUDA DDP/FSDP gradient-accumulation `no_sync` smokes on one Slurm node:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:10:00 --immediate=1 \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_ddp_gradient_accumulation.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:12:00 --immediate=1 \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_fsdp_gradient_accumulation.yaml
```

These validate only tiny Llama-style float32 distributed accumulation. Both runs
use `WORLD_SIZE=2`, `configs/smoke/data_ddp`, `TorchDataLoader.batch_size: 1`,
`effective_batch_size: 4`, and `n_tokens: 64`, which compute
`accumulation_steps: 2`, `n_batches: 4`, and `Total Grad Steps: 2`. The DDP run
logs `[TRAINING_WRAPPER] Using DDP no_sync for gradient accumulation.`; the FSDP
FULL_SHARD run logs `[TRAINING_WRAPPER] Using FSDP no_sync for gradient
accumulation.`. The final saves are
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_ddp_gradient_accumulation/save`
and
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_fsdp_gradient_accumulation/save`;
artifact verification found scheduler `_step_count == 2`, `num_steps == 2`, all
saved tensors `torch.float32`, and matching model hash
`b75cb407af8b36f88a630ae2f4a9ddcc240edbf31548899075c4f8136c729e29`. These
smokes do not validate mixed-precision or bfloat16 no-sync, which are covered
by separate tiny smokes below. Multi-node float32 DDP/FSDP training is covered
by a separate tiny smoke below. These smokes do not validate representative
configs, SSM/Mamba, large or gated data, or full-size checkpoints.

Tiny CUDA DDP/FSDP mixed-precision gradient-accumulation `no_sync` smokes on one
Slurm node:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:10:00 --immediate=1 \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_ddp_mixed_precision_gradient_accumulation.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:12:00 --immediate=1 \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_fsdp_mixed_precision_gradient_accumulation.yaml
```

These validate only tiny Llama-style float32 AMP plus distributed accumulation.
Both runs use `WORLD_SIZE=2`, `configs/smoke/data_ddp`,
`TorchDataLoader.batch_size: 1`, `mixed_precision: true`,
`effective_batch_size: 4`, and `n_tokens: 64`, which compute
`accumulation_steps: 2`, `n_batches: 4`, and `Total Grad Steps: 2`. The DDP run
logs `[TRAINING_WRAPPER] Using DDP no_sync for gradient accumulation.`; the FSDP
FULL_SHARD run logs `Mixed Precision: True` and `[TRAINING_WRAPPER] Using FSDP
no_sync for gradient accumulation.`. The final saves are
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_ddp_mixed_precision_gradient_accumulation/save`
and
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_fsdp_mixed_precision_gradient_accumulation/save`;
artifact verification found scheduler `_step_count == 2`, `num_steps == 2`, all
saved tensors `torch.float32`, DDP hash
`69a62d39bd3931ba5ccfcdaa3a5d08e285a29a9fa359a15e16cb54646c6d5203`, and FSDP
hash `a8eb5ba1843710879037f98bb659cccf4d477f885337907172c057e1da62a97e`.
These smokes do not validate bfloat16 no-sync, which is covered separately
below, nor representative configs, multi-node mixed-precision DDP/FSDP,
SSM/Mamba, large or gated data, or full-size checkpoints.

Tiny CUDA DDP/FSDP bfloat16 gradient-accumulation `no_sync` smokes on one Slurm
node:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:10:00 --immediate=1 \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_ddp_bfloat16_gradient_accumulation.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:12:00 --immediate=1 \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_fsdp_bfloat16_gradient_accumulation.yaml
```

These validate only tiny Llama-style bfloat16 distributed accumulation. Both
runs use `WORLD_SIZE=2`, `configs/smoke/data_ddp`,
`TorchDataLoader.batch_size: 1`, `model_dtype: bfloat16`,
`effective_batch_size: 4`, and `n_tokens: 64`, which compute
`accumulation_steps: 2`, `n_batches: 4`, and `Total Grad Steps: 2`. Both
wrappers log `Cannot use mixed precision with bfloat16. Setting
mixed_precision=False.`. The DDP run logs `[TRAINING_WRAPPER] Using DDP no_sync
for gradient accumulation.`; the FSDP FULL_SHARD run logs `Model dtype:
torch.bfloat16`, `Mixed Precision: False`, and `[TRAINING_WRAPPER] Using FSDP
no_sync for gradient accumulation.`. The final saves are
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_ddp_bfloat16_gradient_accumulation/save`
and
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_fsdp_bfloat16_gradient_accumulation/save`;
artifact verification found scheduler `_step_count == 2`, `num_steps == 2`, all
saved tensors `torch.bfloat16`, and matching DDP/FSDP hash
`84819bd613ccd234514e3b6aff202d5abdc69674963d7941328e80bf3ddd7282`. These
smokes do not validate representative configs; multi-node bfloat16 coverage is
recorded separately below. SSM/Mamba, large or gated data, and full-size
checkpoints remain unvalidated.

Tiny CUDA multi-node DDP/FSDP gradient-accumulation smokes across two Slurm
nodes:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive \
  --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
  --time=00:12:00 --immediate=1 \
  bash -lc 'nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); \
  export MASTER_ADDR=${nodes[0]}; export MASTER_PORT=29541; \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled; \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --nnodes=$SLURM_NNODES --nproc_per_node=2 \
  --node_rank=$SLURM_NODEID --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT run.py \
  --config configs/smoke/tiny_cuda_ddp_multinode_gradient_accumulation.yaml'

srun -p batch -A nemotron_arch_dev --qos=interactive \
  --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
  --time=00:14:00 --immediate=1 \
  bash -lc 'nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); \
  export MASTER_ADDR=${nodes[0]}; export MASTER_PORT=29543; \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled; \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --nnodes=$SLURM_NNODES --nproc_per_node=2 \
  --node_rank=$SLURM_NODEID --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT run.py \
  --config configs/smoke/tiny_cuda_fsdp_multinode_gradient_accumulation.yaml'
```

These validate only tiny float32 DDP/FSDP training across two Slurm nodes with
two local ranks per node. A two-node probe reported nodes `nvl72d091-T05` and
`nvl72d091-T07`, each with four `NVIDIA GB300` GPUs. Both runs use
`WORLD_SIZE=4`, `configs/smoke/data_ddp_multinode`,
`TorchDataLoader.batch_size: 1`, `effective_batch_size: 8`, and `n_tokens: 64`,
then log `Total Grad Steps: 1`. The FSDP run logs `FULL_SHARD`, `Model dtype:
torch.float32`, `Mixed Precision: False`, and `Activation Checkpointing:
False`. The final saves are
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_ddp_multinode_gradient_accumulation/save`
and
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_fsdp_multinode_gradient_accumulation/save`;
artifact verification found scheduler `_step_count == 1`, `num_steps == 1`,
all saved tensors `torch.float32`, and matching model hash
`01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55`. These
smokes do not validate multi-node mixed precision, which is covered separately
below, nor multi-node bfloat16, which is also covered separately below.
Multi-node resume, `eval_hstates`, `eval_ppl`, and benchmark are covered
separately below; representative configs, SSM/Mamba, large or gated data, and
full-size checkpoints remain unvalidated.

Tiny CUDA multi-node DDP/FSDP mixed-precision gradient-accumulation smokes
across two Slurm nodes:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive \
  --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
  --time=00:12:00 --immediate=1 \
  bash -lc 'nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); \
  export MASTER_ADDR=${nodes[0]}; export MASTER_PORT=29545; \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled; \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --nnodes=$SLURM_NNODES --nproc_per_node=2 \
  --node_rank=$SLURM_NODEID --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT run.py \
  --config configs/smoke/tiny_cuda_ddp_multinode_mixed_precision_gradient_accumulation.yaml'

srun -p batch -A nemotron_arch_dev --qos=interactive \
  --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
  --time=00:14:00 --immediate=1 \
  bash -lc 'nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); \
  export MASTER_ADDR=${nodes[0]}; export MASTER_PORT=29547; \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled; \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --nnodes=$SLURM_NNODES --nproc_per_node=2 \
  --node_rank=$SLURM_NODEID --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT run.py \
  --config configs/smoke/tiny_cuda_fsdp_multinode_mixed_precision_gradient_accumulation.yaml'
```

These validate only tiny float32 AMP DDP/FSDP training across two Slurm nodes
with two local ranks per node. The DDP run used nodes `nvl72d112-T11` and
`nvl72d112-T13`; the FSDP run used nodes `nvl72d037-T15` and
`nvl72d037-T18`. Both runs use `WORLD_SIZE=4`,
`configs/smoke/data_ddp_multinode`, `TorchDataLoader.batch_size: 1`,
`mixed_precision: true`, `effective_batch_size: 8`, and `n_tokens: 64`, then
log `Total Grad Steps: 1`. The FSDP run logs `FULL_SHARD`, `Model dtype:
torch.float32`, `Mixed Precision: True`, and `Activation Checkpointing: False`.
The final saves are
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_ddp_multinode_mixed_precision_gradient_accumulation/save`
and
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_fsdp_multinode_mixed_precision_gradient_accumulation/save`;
artifact verification found scheduler `_step_count == 1`, `num_steps == 1`,
all saved tensors `torch.float32`, and matching model hash
`01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55`. These
smokes do not validate multi-node bfloat16, which is covered separately below.
The same tiny float32 AMP path also passed through the storage-backed
default-runtime venv with
`configs/smoke/tiny_cuda_ddp_default_runtime_multinode_mixed_precision_gradient_accumulation.yaml`
and
`configs/smoke/tiny_cuda_fsdp_default_runtime_multinode_mixed_precision_gradient_accumulation.yaml`.
Those runs used Slurm jobs `692716` and `692810`, nodes
`nvl72d125-T07`/`nvl72d125-T08` and
`nvl72d105-T17`/`nvl72d105-T18`, `WORLD_SIZE=4`, `mixed_precision: true`,
`FULL_SHARD` for FSDP, scheduler `_step_count == 1`, `num_steps == 1`, and the
same model hash `01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55`.
Multi-node resume, `eval_hstates`, `eval_ppl`, and benchmark are covered
separately below; representative configs, SSM/Mamba, large or gated data, and
full-size checkpoints remain unvalidated.

Tiny CUDA multi-node DDP/FSDP bfloat16 gradient-accumulation smokes across two
Slurm nodes:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive \
  --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
  --time=00:12:00 --immediate=1 \
  bash -lc 'nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); \
  export MASTER_ADDR=${nodes[0]}; export MASTER_PORT=29549; \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled; \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --nnodes=$SLURM_NNODES --nproc_per_node=2 \
  --node_rank=$SLURM_NODEID --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT run.py \
  --config configs/smoke/tiny_cuda_ddp_multinode_bfloat16_gradient_accumulation.yaml'

srun -p batch -A nemotron_arch_dev --qos=interactive \
  --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
  --time=00:14:00 --immediate=1 \
  bash -lc 'nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); \
  export MASTER_ADDR=${nodes[0]}; export MASTER_PORT=29551; \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled; \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --nnodes=$SLURM_NNODES --nproc_per_node=2 \
  --node_rank=$SLURM_NODEID --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT run.py \
  --config configs/smoke/tiny_cuda_fsdp_multinode_bfloat16_gradient_accumulation.yaml'
```

These validate only tiny bfloat16 DDP/FSDP training across two Slurm nodes with
two local ranks per node. The DDP run used nodes `nvl72d112-T11` and
`nvl72d112-T13`; the FSDP run used nodes `nvl72d012-T01` and
`nvl72d012-T02`. Both runs use `WORLD_SIZE=4`,
`configs/smoke/data_ddp_multinode`, `TorchDataLoader.batch_size: 1`,
`model_dtype: bfloat16`, `mixed_precision: true`, `effective_batch_size: 8`,
and `n_tokens: 64`, then log `Total Grad Steps: 1` and the bfloat16 AMP
fallback warning. The FSDP run logs `FULL_SHARD`, `Model dtype:
torch.bfloat16`, `Mixed Precision: False`, and `Activation Checkpointing:
False`. The final saves are
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_ddp_multinode_bfloat16_gradient_accumulation/save`
and
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_fsdp_multinode_bfloat16_gradient_accumulation/save`;
artifact verification found scheduler `_step_count == 1`, `num_steps == 1`,
all saved tensors `torch.bfloat16`, and matching model hash
`0bc9e79b22d4fce8a926de7260a144c0a5a0ff09a86328af5f35777ab655af15`. These
smokes do not validate multi-node resume, `eval_hstates`, or `eval_ppl`, which
are covered separately below. The same tiny bfloat16 path also passed through
the storage-backed default-runtime venv with
`configs/smoke/tiny_cuda_ddp_default_runtime_multinode_bfloat16_gradient_accumulation.yaml`
and
`configs/smoke/tiny_cuda_fsdp_default_runtime_multinode_bfloat16_gradient_accumulation.yaml`.
Those runs used Slurm jobs `692872` and `692887`, nodes
`nvl72d054-T01`/`nvl72d054-T02` and
`nvl72d108-T15`/`nvl72d108-T18`, `WORLD_SIZE=4`, `model_dtype: bfloat16`,
the expected bfloat16 AMP fallback warning, `FULL_SHARD` for FSDP, scheduler
`_step_count == 1`, `num_steps == 1`, all saved tensors `torch.bfloat16`, and
the same model hash `0bc9e79b22d4fce8a926de7260a144c0a5a0ff09a86328af5f35777ab655af15`.
Multi-node benchmark, representative configs, SSM/Mamba, large or gated data,
and full-size checkpoints remain unvalidated.

Tiny CUDA multi-node DDP/FSDP checkpoint resume smokes across two Slurm nodes:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive \
  --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
  --time=00:12:00 --immediate=1 \
  bash -lc 'nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); \
  export MASTER_ADDR=${nodes[0]}; export MASTER_PORT=29553; \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled; \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --nnodes=$SLURM_NNODES --nproc_per_node=2 \
  --node_rank=$SLURM_NODEID --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT run.py \
  --config configs/smoke/tiny_cuda_ddp_multinode_resume.yaml'

srun -p batch -A nemotron_arch_dev --qos=interactive \
  --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
  --time=00:14:00 --immediate=1 \
  bash -lc 'nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); \
  export MASTER_ADDR=${nodes[0]}; export MASTER_PORT=29555; \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled; \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --nnodes=$SLURM_NNODES --nproc_per_node=2 \
  --node_rank=$SLURM_NODEID --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT run.py \
  --config configs/smoke/tiny_cuda_fsdp_multinode_resume.yaml'
```

These validate only tiny float32 DDP/FSDP checkpoint resume across two Slurm
nodes with two local ranks per node. The DDP run used nodes `nvl72d094-T12` and
`nvl72d094-T14`; the FSDP run used nodes `nvl72d016-T15` and
`nvl72d016-T18`. Both resume from the tiny multi-node float32 source
checkpoints, load model, optimizer, and scheduler paths from
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_ddp_multinode_gradient_accumulation/save`
or
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_fsdp_multinode_gradient_accumulation/save`,
run with `WORLD_SIZE=4`, `configs/smoke/data_ddp_multinode`,
`TorchDataLoader.batch_size: 1`, `effective_batch_size: 8`, and `n_tokens: 128`,
then advance scheduler state from `1/1` to `2/2`. Both runs used distributed
`no_sync` during resumed accumulation, saved
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_ddp_multinode_resume/save` and
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_fsdp_multinode_resume/save`,
and destroyed process groups cleanly. Artifact verification found source
optimizer state entries `0` and resumed optimizer state entries `12`, all saved
tensors `torch.float32`, source model hash
`01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55`, matching
resumed DDP/FSDP hash
`5faf34670d9ba7799cb14ed17fb29ec3679c01a003c1d9100783181ae2d6b2d9`, and
`hash_changed True`. These smokes do not validate multi-node `eval_hstates`,
which is covered separately below, nor multi-node `eval_ppl`, which is also
covered separately below. Multi-node benchmark is covered separately below.
The same tiny float32 path was also rerun through the storage-backed
default-runtime venv with
`configs/smoke/tiny_cuda_ddp_default_runtime_multinode_resume.yaml` and
`configs/smoke/tiny_cuda_fsdp_default_runtime_multinode_resume.yaml`. Those
default-runtime runs used Slurm jobs `692584` and `692601`, nodes
`nvl72d053-T10`/`nvl72d053-T13` and
`nvl72d053-T17`/`nvl72d053-T18`, the same `WORLD_SIZE=4` shape, source
checkpoints under
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_ddp_default_runtime_multinode_gradient_accumulation/save`
and
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_fsdp_default_runtime_multinode_gradient_accumulation/save`,
and saved
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_ddp_default_runtime_multinode_resume/save`
and
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_fsdp_default_runtime_multinode_resume/save`.
The default-runtime artifact verification matched the shared-runtime resume
hash `5faf34670d9ba7799cb14ed17fb29ec3679c01a003c1d9100783181ae2d6b2d9`,
advanced scheduler state from `1/1` to `2/2`, moved optimizer state entries
from `0` to `12`, and changed all 12 tensors. These smokes do not validate
mixed-precision or bfloat16 multi-node resume,
representative configs, SSM/Mamba, large or gated data, or full-size
checkpoints.

Tiny CUDA multi-node DDP/FSDP `eval_hstates` callback smokes across two Slurm
nodes:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive \
  --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
  --time=00:14:00 --immediate=1 \
  bash -lc 'nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); \
  export MASTER_ADDR=${nodes[0]}; export MASTER_PORT=29557; \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled; \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --nnodes=$SLURM_NNODES --nproc_per_node=2 \
  --node_rank=$SLURM_NODEID --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT run.py \
  --config configs/smoke/tiny_cuda_ddp_multinode_eval_hstates.yaml'

srun -p batch -A nemotron_arch_dev --qos=interactive \
  --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
  --time=00:16:00 --immediate=1 \
  bash -lc 'nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); \
  export MASTER_ADDR=${nodes[0]}; export MASTER_PORT=29559; \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled; \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --nnodes=$SLURM_NNODES --nproc_per_node=2 \
  --node_rank=$SLURM_NODEID --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT run.py \
  --config configs/smoke/tiny_cuda_fsdp_multinode_eval_hstates.yaml'
```

These validate only tiny float32 DDP/FSDP `EvalConfig -> eval_hstates`
callbacks across two Slurm nodes with two local ranks per node. The DDP run
used nodes `nvl72d026-T08` and `nvl72d026-T10`; the FSDP run used nodes
`nvl72d112-T11` and `nvl72d112-T13`. Both run with `WORLD_SIZE=4`, train and
eval data `configs/smoke/data_ddp_multinode`, `TorchDataLoader.batch_size: 1`,
`effective_batch_size: 8`, `n_tokens: 64`, `eval_at_start: true`,
`eval_at_end: true`, `save_latest: true`, and `n_batches: 1`. Both load 12/12
teacher keys, run `eval_hstates`, log `eval_score: 1.593651` and
`hstates_distance: 1.593651`, save `latest_eval_hstates`, and save final
checkpoints at
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_ddp_multinode_eval_hstates/save`
and
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_fsdp_multinode_eval_hstates/save`.
Artifact verification found final and latest checkpoint files, scheduler
`_step_count == 1`, `num_steps == 1`, latest score `{'eval_score':
1.5936508178710938, 'hstates_distance': 1.5936508178710938}`, all final/latest
tensors `torch.float32`, and matching final/latest DDP/FSDP hash
`01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55`. These
smokes do not validate multi-node `eval_ppl`, which is covered separately
below. Multi-node benchmark is covered separately below. These smokes do not
validate mixed-precision or bfloat16 multi-node eval callbacks, representative
eval data, SSM/Mamba, large or gated data, or full-size checkpoints.

Tiny CUDA multi-node DDP/FSDP `eval_ppl` callback smokes across two Slurm
nodes:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive \
  --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
  --time=00:14:00 --immediate=1 \
  bash -lc 'nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); \
  export MASTER_ADDR=${nodes[0]}; export MASTER_PORT=29561; \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled; \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --nnodes=$SLURM_NNODES --nproc_per_node=2 \
  --node_rank=$SLURM_NODEID --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT run.py \
  --config configs/smoke/tiny_cuda_ddp_multinode_eval_ppl.yaml'

srun -p batch -A nemotron_arch_dev --qos=interactive \
  --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
  --time=00:16:00 --immediate=1 \
  bash -lc 'nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); \
  export MASTER_ADDR=${nodes[0]}; export MASTER_PORT=29563; \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled; \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --nnodes=$SLURM_NNODES --nproc_per_node=2 \
  --node_rank=$SLURM_NODEID --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT run.py \
  --config configs/smoke/tiny_cuda_fsdp_multinode_eval_ppl.yaml'
```

These validate only tiny float32 DDP/FSDP `EvalConfig -> eval_ppl` callbacks
across two Slurm nodes with two local ranks per node. The DDP run used nodes
`nvl72d012-T03` and `nvl72d012-T04`; the FSDP run used nodes `nvl72d103-T01`
and `nvl72d103-T02`. Both run with `WORLD_SIZE=4`, train and eval data
`configs/smoke/data_ddp_multinode`, `TorchDataLoader.batch_size: 1`,
`effective_batch_size: 8`, `n_tokens: 64`, `eval_at_start: true`,
`eval_at_end: true`, `save_latest: true`, and `n_batches: 1`. Both load 12/12
teacher keys, run `eval_ppl`, log `eval_score: 44002.273438`, `perplexity:
44002.273438`, and `accuracy: 0.000000`, save `latest_Perplexity`, and save
final checkpoints at
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_ddp_multinode_eval_ppl/save`
and
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_fsdp_multinode_eval_ppl/save`.
Artifact verification found final and latest checkpoint files, scheduler
`_step_count == 1`, `num_steps == 1`, latest score `{'eval_score':
44002.2734375, 'perplexity': 44002.2734375, 'accuracy': 0.0}`, all final/latest
tensors `torch.float32`, and matching final/latest DDP/FSDP hash
`01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55`. These
Multi-node benchmark is covered separately below. These smokes do not validate
mixed-precision or bfloat16 multi-node eval callbacks, representative eval
data, SSM/Mamba, large or gated data, or full-size checkpoints.

Tiny CUDA multi-node DDP/FSDP benchmark callback smokes across two Slurm nodes:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive \
  --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
  --time=00:12:00 --immediate=1 \
  bash -lc 'nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); \
  export MASTER_ADDR=${nodes[0]}; export MASTER_PORT=29607; \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled; \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --nnodes=$SLURM_NNODES --nproc_per_node=2 \
  --node_rank=$SLURM_NODEID --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT run.py \
  --config configs/smoke/tiny_cuda_ddp_multinode_eval_benchmark.yaml'

srun -p batch -A nemotron_arch_dev --qos=interactive \
  --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
  --time=00:14:00 --immediate=1 \
  bash -lc 'nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); \
  export MASTER_ADDR=${nodes[0]}; export MASTER_PORT=29605; \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled; \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --nnodes=$SLURM_NNODES --nproc_per_node=2 \
  --node_rank=$SLURM_NODEID --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT run.py \
  --config configs/smoke/tiny_cuda_fsdp_multinode_eval_benchmark.yaml'
```

These validate only tiny float32 DDP/FSDP `EvalConfig -> benchmark` callbacks
across two Slurm nodes with two local ranks per node. Both run with
`WORLD_SIZE=4`, train data `configs/smoke/data_ddp_multinode`, cached public
`wikitext` from
`/home/abick/mohawk/artifacts/gpu_smoke/lm_eval_datasets`,
`TorchDataLoader.batch_size: 1`, `effective_batch_size: 8`, `n_tokens: 64`,
`eval_at_start: true`, `eval_at_end: false`, `save_latest: false`, and
benchmark `limit: 4`. The patched DDP run used Slurm job `688414`; the patched
FSDP `FULL_SHARD` run used Slurm job `688406`. Both logged `eval_score:
172057.421054`, `wikitext 172057.421054`, and `AVG 172057.421054`, saved final
checkpoints at
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_ddp_multinode_eval_benchmark/save`
and
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_fsdp_multinode_eval_benchmark/save`,
and destroyed process groups cleanly. Artifact verification found scheduler
`_step_count == 1`, `num_steps == 1`, all saved tensors `torch.float32`, and
matching DDP/FSDP hash
`01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55`. These
smokes do not validate representative benchmarks, uncached task downloads,
mixed-precision or bfloat16 distributed benchmark callbacks, SSM/Mamba, large
or gated data, or full-size checkpoints.

Tiny CPU WSD scheduler smoke after creating the same local teacher checkpoint:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_wsd_scheduler.yaml
```

This validates only centralized tiny CPU `OptimizerConfig.scheduler.name: wsd`
through the real supervised `run.py` path. The smoke uses `n_tokens: 32`,
fractional `warmup_steps: 0.25` and `decay_steps: 0.25`, which resolve to one
warmup step and one decay step out of four total scheduler steps. It writes
`/tmp/mohawk_tiny_wsd_scheduler_cpu_run/save`. It does not validate CUDA,
DDP/FSDP scheduler behavior, `cosine`, `wsd_train`, representative configs, or
full-size checkpoints.

Tiny CPU Adam optimizer smoke after creating the same local teacher checkpoint:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_adam_optimizer.yaml
```

This validates only centralized tiny CPU `OptimizerConfig.optimizer: Adam`
through the real supervised `run.py` path. The wrapper constructs
`torch.optim.Adam`, runs two optimizer/scheduler steps, and writes
`/tmp/mohawk_tiny_adam_optimizer_cpu_run/save`. It does not validate Adam under
CUDA, DDP/FSDP optimizer wrapping, representative configs, or full-size
checkpoints.

Tiny CPU selective-optimization smokes after creating the same local teacher
checkpoint:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_optimize_weights_whitelist.yaml

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_optimize_weights_blacklist.yaml
```

These validate only centralized tiny CPU `OptimizerConfig.optimize_weights`
filtering through the real supervised `run.py` path. The whitelist smoke trains
only `lm_head.weight`; the blacklist smoke freezes only `lm_head.weight`.
They write `/tmp/mohawk_tiny_optimize_weights_whitelist_cpu_run/save` and
`/tmp/mohawk_tiny_optimize_weights_blacklist_cpu_run/save`. They do not
validate CUDA, DDP/FSDP flattened/original-parameter behavior, representative
configs, or tied-embedding edge cases.

Tiny CPU partial `LoadConfig.model` checkpoint-load smokes after creating the
same local teacher checkpoint:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_load_model_whitelist.yaml

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_load_model_blacklist.yaml

python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cpu_supervised.yaml \
  --output /tmp/mohawk_tiny_renamed_teacher_ckpt \
  --rename-key lm_head.weight=renamed_lm_head.weight

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_load_model_rename.yaml

python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cpu_supervised.yaml \
  --output /tmp/mohawk_tiny_sequence_renamed_lm_head_ckpt \
  --rename-key lm_head.weight=renamed_lm_head.weight

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_load_model_sequence.yaml
```

These validate only centralized tiny CPU `LoadConfig.model` partial checkpoint
filtering through the real supervised `run.py` path. The whitelist smoke loads
only `lm_head.weight` from `/tmp/mohawk_tiny_teacher_ckpt` and freezes it; the
blacklist smoke loads the eleven non-`lm_head` tensors and trains only
`lm_head.weight`; the rename smoke loads `renamed_lm_head.weight` from
`/tmp/mohawk_tiny_renamed_teacher_ckpt` through `LoadConfig.model.rename` with
`renamed_lm_head: lm_head` and freezes the renamed `lm_head.weight`; the
sequence smoke proves two `LoadConfig.model` entries are applied in order by
loading eleven backbone tensors from `/tmp/mohawk_tiny_teacher_ckpt` and
loading a distinct renamed `lm_head.weight` from
`/tmp/mohawk_tiny_sequence_renamed_lm_head_ckpt`. They write
`/tmp/mohawk_tiny_load_model_whitelist_cpu_run/save` and
`/tmp/mohawk_tiny_load_model_blacklist_cpu_run/save`, and
`/tmp/mohawk_tiny_load_model_rename_cpu_run/save`, and
`/tmp/mohawk_tiny_load_model_sequence_cpu_run/save`. They do not validate CUDA,
DDP/FSDP sharded checkpoint loading, representative checkpoints,
`init_ssm_from_attention`, list ordering beyond this two-entry case, or
multi-key rename collision behavior.

Tiny CPU centralized mixed-precision smoke after creating the same local
teacher checkpoint:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_mixed_precision.yaml
```

This validates only the centralized tiny CPU path where both `TrainConfig` and
`TeacherConfig` set `mixed_precision: true`. It runs through the real
supervised `run.py` path and writes
`/tmp/mohawk_tiny_mixed_precision_cpu_run/save`. It does not validate CUDA
autocast/scaler behavior, DDP/FSDP mixed precision, bfloat16 model-dtype
fallback, representative configs, or full-size checkpoints.

Tiny CPU mixed-precision gradient-accumulation smoke after creating the same
local teacher checkpoint:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_mixed_precision_gradient_accumulation.yaml
```

This validates only the centralized tiny CPU path where AMP/scaler mode and
micro-batch accumulation are enabled together. The smoke uses
`mixed_precision: true`, `effective_batch_size: 2`, and
`TorchDataLoader.batch_size: 1`, computes `accumulation_steps: 2` and
`n_batches: 4`, then writes
`/tmp/mohawk_tiny_mixed_precision_gradient_accumulation_cpu_run/save`. It does
not validate CUDA autocast/scaler behavior, DDP/FSDP `no_sync` with mixed
precision, representative configs, or full-size checkpoints.

Tiny CPU bfloat16 mixed-precision fallback smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_bfloat16_mixed_precision_fallback.yaml
```

This validates only the tiny CPU Llama-style path where `model_dtype: bfloat16`
is combined with `mixed_precision: true`; the centralized wrapper logs that AMP
is disabled for bfloat16 and trains the bfloat16 model directly. It writes
`/tmp/mohawk_tiny_bfloat16_fallback_cpu_run/save`. It does not validate CUDA
bfloat16, DDP/FSDP, representative configs, or full-size checkpoints.

Tiny CUDA/DDP/FSDP bfloat16 supervised smokes:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:08:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  python run.py --config configs/smoke/tiny_cuda_bfloat16_supervised.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:08:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  torchrun --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_ddp_bfloat16_supervised.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:10:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  torchrun --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_fsdp_bfloat16_supervised.yaml
```

These validate tiny single-process CUDA, two-rank DDP, and two-rank FSDP
FULL_SHARD supervised training with `TrainConfig.model_dtype: bfloat16`,
`TeacherConfig.model_dtype: bfloat16`, and bfloat16 checkpoint tensors. The
wrappers log that AMP mixed precision is disabled for bfloat16 and train the
bfloat16 model directly. These are supervised-training smokes only;
representative configs, multi-node DDP/FSDP, SSM/Mamba, and full-size
checkpoints remain separate gates.

Tiny CPU bfloat16 gradient-accumulation fallback smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_bfloat16_gradient_accumulation_fallback.yaml
```

This validates only the tiny CPU Llama-style path where `model_dtype: bfloat16`
and `mixed_precision: true` are combined with micro-batch accumulation. The
centralized wrapper logs that AMP is disabled for bfloat16, computes
`accumulation_steps: 2`, and writes
`/tmp/mohawk_tiny_bfloat16_gradient_accumulation_cpu_run/save`.

Equivalent tiny CPU bfloat16 gradient-accumulation smokes exist for Qwen2, Phi,
and Falcon:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_qwen2_bfloat16_gradient_accumulation_fallback.yaml

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_phi_bfloat16_gradient_accumulation_fallback.yaml

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_falcon_bfloat16_gradient_accumulation_fallback.yaml
```

These write to `/tmp/mohawk_tiny_qwen2_bfloat16_gradient_accumulation_cpu_run/save`,
`/tmp/mohawk_tiny_phi_bfloat16_gradient_accumulation_cpu_run/save`, and
`/tmp/mohawk_tiny_falcon_bfloat16_gradient_accumulation_cpu_run/save`. They do
not validate DDP/FSDP `no_sync`, fast Mamba kernels, representative configs, or
full-size checkpoints.

Tiny CUDA Qwen2/Phi/Falcon bfloat16 eval callback smokes:

```bash
python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cpu_qwen2_supervised.yaml \
  --output /home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_qwen2_teacher_ckpt
python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cpu_phi_supervised.yaml \
  --output /home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_phi_teacher_ckpt
python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cpu_falcon_supervised.yaml \
  --output /home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_falcon_teacher_ckpt

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:10:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  python run.py --config configs/smoke/tiny_cuda_qwen2_bfloat16_eval_callbacks.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:10:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  python run.py --config configs/smoke/tiny_cuda_phi_bfloat16_eval_callbacks.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:10:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  python run.py --config configs/smoke/tiny_cuda_falcon_bfloat16_eval_callbacks.yaml
```

These validate tiny single-process CUDA bfloat16 eval callbacks for
`Qwen2Model -> Qwen2Block -> Qwen2Attention`, `LlamaModel -> MambaPhi ->
PhiAttention`, and `LlamaModel -> FalconBlock -> FalconMambaMixer`. Each config
runs both `EvalConfig -> eval_hstates` and `EvalConfig -> eval_ppl`, writes
`latest_eval_hstates` and `latest_Perplexity`, and saves final/latest tensors as
`torch.bfloat16`. Recorded eval scores are Qwen2 `1.531250` and
`59874.144531`, Phi `1.679688` and `40134.855469`, and Falcon `1.015625` and
`62943.968750`. Recorded model hashes are
`8414e24632d042ced981bc7f4457205e643195d41f9ff7d6b4a0f37da58d3a19`,
`e84464330dba07b4ef35e0f538d923fd4c188cae2cb1458b6cb18f93b60c66f0`, and
`a0297d015d195ccda5cb633c16624df99e256053a09f14d31a140073bc73d790`.
These smokes do not validate representative configs, fast Mamba kernels,
SSM/Mamba package dependencies, multi-node distributed eval, or full-size
checkpoints.

Tiny CUDA DDP/FSDP Qwen2/Phi/Falcon bfloat16 eval callback smokes:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:10:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  torchrun --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_ddp_qwen2_bfloat16_eval_callbacks.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:12:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  torchrun --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_fsdp_qwen2_bfloat16_eval_callbacks.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:10:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  torchrun --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_ddp_phi_bfloat16_eval_callbacks.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:12:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  torchrun --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_fsdp_phi_bfloat16_eval_callbacks.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:10:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  torchrun --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_ddp_falcon_bfloat16_eval_callbacks.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:12:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  torchrun --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_fsdp_falcon_bfloat16_eval_callbacks.yaml
```

These validate tiny two-rank DDP and two-rank FSDP FULL_SHARD bfloat16 eval
callbacks for the same Qwen2, Phi, and Falcon paths. The DDP configs inherit
the CPU model-family bfloat16 fallback bases directly, use the shared
Slurm-visible teacher checkpoints above, and use `configs/smoke/data_ddp` for
train and eval data so both ranks have local JSON shards. Each run executes
`EvalConfig -> eval_hstates` and `EvalConfig -> eval_ppl`, writes
`latest_eval_hstates` and `latest_Perplexity`, records scheduler
`_step_count == 1` and `num_steps == 1`, and saves final/latest tensors as
`torch.bfloat16`. Recorded DDP/FSDP scores and hashes are Qwen2 `1.4765625` and
`69068.7421875` with hash
`c95790cea5ec1fe21235972050d52293b981a21fb4f50bfd1acb3e9f308511ba`, Phi
`1.6484375` and `50082.56640625` with hash
`82d94929797662c9abc5b7ecb20962f82dfef0c41d252be6f5a4be8e31045e3b`, and
Falcon `0.99609375` and `53790.79296875` with hash
`ef48a53d4ba3941435096b197361619301b7560ef96322543ab0ece75b3c3dde`.
These smokes do not validate representative configs, fast Mamba kernels,
SSM/Mamba package dependencies, multi-node distributed eval, or full-size
checkpoints.

Tiny CPU Qwen2 bfloat16 mixed-precision fallback smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_qwen2_bfloat16_mixed_precision_fallback.yaml
```

This validates only the tiny CPU Qwen2 path where `model_dtype: bfloat16` is
combined with `mixed_precision: true`; the centralized wrapper logs that AMP is
disabled for bfloat16 and trains the bfloat16 Qwen2 model directly. It writes
`/tmp/mohawk_tiny_qwen2_bfloat16_fallback_cpu_run/save`. It does not validate
DDP/FSDP, representative configs, or full-size checkpoints.

Tiny CPU Phi bfloat16 mixed-precision fallback smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_phi_bfloat16_mixed_precision_fallback.yaml
```

This validates only the tiny CPU PhiAttention path where `model_dtype:
bfloat16` is combined with `mixed_precision: true`; the centralized wrapper
logs that AMP is disabled for bfloat16 and trains the bfloat16 Phi model
directly. It writes
`/tmp/mohawk_tiny_phi_bfloat16_fallback_cpu_run/save`. It does not validate
DDP/FSDP, representative configs, or full-size checkpoints.

Tiny CPU Falcon bfloat16 mixed-precision fallback smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_falcon_bfloat16_mixed_precision_fallback.yaml
```

This validates only the tiny CPU FalconBlock path where `model_dtype: bfloat16`
is combined with `mixed_precision: true`; the centralized wrapper logs that AMP
is disabled for bfloat16 and trains the bfloat16 Falcon model directly with
Transformers' sequential Mamba fallback. It writes
`/tmp/mohawk_tiny_falcon_bfloat16_fallback_cpu_run/save`. It does not validate
DDP/FSDP, fast Mamba kernels, SSM/Mamba package dependencies, representative
configs, or full-size checkpoints.

Tiny CPU centralized `torch.compile` smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_compile_model.yaml
```

This validates only the centralized tiny CPU path where the student wrapper
sets `TrainConfig.compile_model: true` and compiles with `torch.compile`
Inductor fullgraph mode before the real supervised `run.py` training loop. It
writes `/tmp/mohawk_tiny_compile_model_cpu_run/save` and verifies the saved
checkpoint has no `_orig_mod.` prefixes. It does not validate CUDA compile
behavior, DDP/FSDP compile paths, representative configs, or full-size
checkpoints.

Tiny CPU teacher `torch.compile` smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_teacher_compile_model.yaml
```

This validates only the centralized tiny CPU path where the teacher wrapper
sets `TeacherConfig.compile_model: true` and compiles in inference mode with
`torch.compile` Inductor fullgraph mode while the student remains uncompiled.
It writes `/tmp/mohawk_tiny_teacher_compile_model_cpu_run/save` and verifies
the saved student checkpoint has no `_orig_mod.` prefixes. It does not validate
CUDA compile behavior, DDP/FSDP compile paths, representative configs, or
full-size checkpoints.

Tiny CPU student+teacher `torch.compile` smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_student_teacher_compile_model.yaml
```

This validates only the centralized tiny CPU path where both the student and
teacher wrappers set `compile_model: true` and compile with `torch.compile`
Inductor fullgraph mode in the same real supervised `run.py` training loop. It
writes `/tmp/mohawk_tiny_student_teacher_compile_model_cpu_run/save` and
verifies the saved student checkpoint has no `_orig_mod.` prefixes.

Tiny CPU bfloat16 centralized `torch.compile` smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_bfloat16_compile_model.yaml
```

This validates only the centralized tiny CPU path where the compiled student
also uses `model_dtype: bfloat16` and the bfloat16 AMP fallback path. It writes
`/tmp/mohawk_tiny_bfloat16_compile_model_cpu_run/save`, verifies the saved
checkpoint has no `_orig_mod.` prefixes, and reloads the checkpoint into an
uncompiled bfloat16 local model. It does not validate CUDA compile behavior,
DDP/FSDP compile paths, representative configs, or full-size checkpoints.

Tiny CPU bfloat16 teacher `torch.compile` smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_bfloat16_teacher_compile_model.yaml
```

This validates only the centralized tiny CPU path where the teacher wrapper
compiles in inference mode under `model_dtype: bfloat16` and the bfloat16 AMP
fallback while the student remains uncompiled. It writes
`/tmp/mohawk_tiny_bfloat16_teacher_compile_model_cpu_run/save` and verifies the
saved student checkpoint has no `_orig_mod.` prefixes.

Tiny CPU bfloat16 student+teacher `torch.compile` smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_bfloat16_student_teacher_compile_model.yaml
```

This validates only the centralized tiny CPU path where both wrappers compile
under `model_dtype: bfloat16` and the bfloat16 AMP fallback. It writes
`/tmp/mohawk_tiny_bfloat16_student_teacher_compile_model_cpu_run/save`,
verifies the saved checkpoint has no `_orig_mod.` prefixes, and keeps CUDA,
DDP/FSDP, representative configs, and full-size checkpoints as separate gates.

Tiny CPU public Hugging Face teacher smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_hf_teacher_supervised.yaml
```

This validates only a tiny public teacher path with
`TeacherConfig.dir: sshleifer/tiny-gpt2`, local JSON data, and an isolated
`/tmp` Hugging Face cache. The first offline run fails if that isolated cache
is empty; after one network-backed run populates it, the same command can run
with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. This does not validate
large or gated teachers, CUDA, DDP/FSDP, SSM/Mamba, representative datasets, or
full-size checkpoints.

Tiny CPU public Hugging Face dataset smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_public_hfdata_supervised.yaml
```

This validates only a tiny public training-data path with
`NeelNanda/pile-10k` through
`HFDataset -> Tokenize -> PaddingDataLoader -> CycleDataLoader ->
TorchDataLoader -> run.py`, plus public `sshleifer/tiny-gpt2` teacher and
tokenizer. The first run needs network access to populate isolated `/tmp`
dataset/model caches; after that, the same command can run with
`HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`.
This does not validate large or gated datasets, default-buffer streaming
throughput, distributed sharding, CUDA, DDP/FSDP, or representative
checkpoints.

Tiny CPU public C4 streaming dataset smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_c4_streaming_supervised.yaml
```

This validates only a tiny public C4 streaming training path with
`allenai/c4`, `name: en.noclean`, `streaming: true`, and
`shuffle_buffer_size: 2` through
`HFDataset -> Tokenize -> PaddingDataLoader -> CycleDataLoader ->
TorchDataLoader -> run.py`. It uses the local tiny teacher checkpoint and a
cached `sshleifer/tiny-gpt2` tokenizer, streams public C4 data with network
access, completes two scheduler steps, and writes
`/tmp/mohawk_tiny_c4_streaming_cpu_run/save`. It does not validate the default
`10000` shuffle buffer, distributed sharding, CUDA, DDP/FSDP, gated datasets,
or full-size training.

Tiny CPU public FineWeb-Edu streaming dataset smoke has a validated
storage-backed default-runtime path:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 WANDB_MODE=disabled \
HF_DATASETS_CACHE=/home/abick/storage/mohawk/fineweb_edu_streaming_smoke_default_runtime/huggingface/datasets \
HF_HUB_CACHE=/home/abick/storage/mohawk/fineweb_edu_streaming_smoke_default_runtime/huggingface/hub \
/home/abick/storage/mohawk/venvs/ssm-cuda-20260626/bin/python run.py \
  --config configs/smoke/tiny_cpu_fineweb_edu_streaming_supervised_default_runtime.yaml
```

This config uses `HuggingFaceFW/fineweb-edu`, `name: sample-100BT`,
`streaming: true`, and `shuffle_buffer_size: 2` through the same
`HFDataset -> Tokenize -> PaddingDataLoader -> CycleDataLoader ->
TorchDataLoader -> run.py` path, with storage-backed outputs under
`/home/abick/storage/mohawk/fineweb_edu_streaming_smoke_default_runtime`.
The default-runtime run loads the local tiny teacher, completes two scheduler
steps, exits with status 0, and writes a complete checkpoint with model hash
`9e447f7af4d9d10fbd0bab04b7078d39bc6239f1681a73cdebc2c8c7762676d2`.
`Tokenize.local_files_only: false` is used only so the tiny
`sshleifer/tiny-gpt2` tokenizer can be fetched into the storage cache.

The older shared ML environment (`datasets 3.3.0`) still has documented
FineWeb-Edu shutdown failures: it writes the checkpoint, then exits with signal
139 during interpreter shutdown with `Fatal Python error:
PyGILState_Release`. A later rerun there also hit Hugging Face
`429 Too Many Requests` without an available `HF_TOKEN`. Treat the clean
default-runtime CPU result as a bounded tiny public-data smoke only.

A separate centralized CUDA smoke exercises the production loader settings:

```bash
python run.py \
  --config configs/smoke/tiny_cuda_fineweb_edu_production_loader_supervised.yaml
```

That config omits `HFDataset.shuffle_buffer_size`, taking the runtime default
`10000`, and uses `HFDataset -> Tokenize -> PackingDataLoader ->
TorchDataLoader` with `max_seq_len: 2048`. Slurm job `706699` ran on a GB300,
loaded 12/12 teacher keys, completed a real optimizer update, and saved 12
finite float32 tensors plus 12 optimizer states under
`/home/abick/storage/mohawk/fineweb_edu_production_loader_cuda`; scheduler
state was `2/2` and the model hash was
`55d4b4188067a61a6fcd6799b460e2059536b6b92f265c9e74e773ea96f600a1`.

DDP/FSDP variants are provided in
`configs/smoke/tiny_cuda_ddp_fineweb_edu_production_loader_supervised.yaml`
and
`configs/smoke/tiny_cuda_fsdp_fineweb_edu_production_loader_supervised.yaml`.
A two-rank shape/sharding pass reached DDP broadcast, FSDP `FULL_SHARD`, and
rank-aware FineWeb streams, but did not perform an optimizer update. The
corrected optimizer-update retry hit Hugging Face `429 Too Many Requests` on
both ranks without an available token.

The bounded distributed optimizer-update path now uses explicit public-shard
overlays:

```bash
torchrun --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_ddp_fineweb_edu_direct_shard_supervised.yaml
torchrun --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_fsdp_fineweb_edu_direct_shard_supervised.yaml
```

These configs keep the production `2048`-token packing path and default
`10000`-record shuffle buffer, but load one explicit public FineWeb-Edu parquet
shard through `HFDataset.path: parquet` and scoped `data_files`. Slurm jobs
`757646` (DDP) and `757737` (FSDP) each ran two ranks on a GB300 node, loaded
12/12 teacher keys, completed scheduler `2/2`, and saved 12 finite float32
tensors plus 12 optimizer states. Their model hashes match at
`4894e81ba6c093dcbe2e603f798803d8564a7d5d683b15c170fc07f5fb9dbd46`.
This proves a tiny two-rank optimizer update over one explicit public shard. It
does not prove repository-wide dataset discovery, multiple-shard distribution,
gated access, sustained streaming throughput, full-size models, or full-size
training.

Legacy `C4DataLoader` note: older placeholder configs previously referenced a
local MosaicML Streaming C4 path with `data_dir: <path-to-c4-dataset>` and
`loader: C4DataLoader`. Those references have been removed or replaced by
current public `HFDataset` C4 configs; the validated public C4 smoke above uses
that path instead of the legacy StreamingC4 loader. In the current validated
environments, `streaming` is not installed and `setup_dataloader` for
`C4DataLoader` raises
`ImportError: C4DataLoader requires an optional dependency that is not
installed: No module named 'streaming'`. Treat `C4DataLoader` as unsupported
until its dependency, local shard fixture, and a real `run.py` or evaluator
smoke are added.

Tiny CPU RandomDataLoader smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_random_data_supervised.yaml
```

This validates only a tiny synthetic random-token path with
`RandomDataLoader -> CycleDataLoader -> run.py`. It uses cached
`sshleifer/tiny-gpt2` tokenization metadata with `local_files_only: true`,
`max_seq_len: 8`, two generated samples, and 8-token batched `input_ids`.
`TorchDataLoader` is not in this loader chain because `RandomDataLoader`
already yields batched dictionaries; the config still includes
`TorchDataLoader.batch_size: 1` because the trainer reads it during batch-size
normalization. The run completes two scheduler steps and writes
`/tmp/mohawk_tiny_random_data_cpu_run/save`. It does not validate real data
loading, tokenization/collation/padding wrappers, TorchDataLoader wrapping,
representative datasets, CUDA, DDP/FSDP, distributed data sharding, or
full-size training.

Tiny CPU ShuffleLoader smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_shuffle_loader_supervised.yaml
```

This validates only a tiny local two-source shuffle path with
`ShuffleLoader -> CycleDataLoader -> run.py`. It uses
`configs/smoke/tiny_shuffle_source_a.yaml` and
`configs/smoke/tiny_shuffle_source_b.yaml`, which each build a
`JSONIterableDataset -> Tokenize(collate_type=text) -> PaddingDataLoader ->
TorchDataLoader` source over one local JSON sample. The smoke uses cached
`sshleifer/tiny-gpt2` tokenization, `max_seq_len: 8`, and 8-token
`input_ids` batches, completes two scheduler steps, and writes
`/tmp/mohawk_tiny_shuffle_loader_cpu_run/save`. It does not validate large
shuffle pools, source weighting, remote datasets, dataloader-state resume,
CUDA, DDP/FSDP, distributed data sharding, or full-size training.

Tiny CPU conversation-collate smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_conversation_collate.yaml
```

This validates only a tiny local conversation-data path with
`JSONIterableDataset -> Tokenize(collate_type=conversation) ->
PaddingDataLoader -> CycleDataLoader -> TorchDataLoader -> run.py`. It uses
`configs/smoke/data_conversation/tiny_conversation.json`, an explicit
`configs/smoke/tiny_chat_template.jinja` template for cached
`sshleifer/tiny-gpt2`, completes two scheduler steps, and writes
`/tmp/mohawk_tiny_conversation_collate_cpu_run/save`. It does not validate
large remote conversation datasets, chat-tokenizer default templates, CUDA,
DDP/FSDP, distributed data sharding, or full-size training.

Tiny CPU classic-collate smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_classic_collate.yaml
```

This validates only a tiny local pre-tokenized-data path with
`JSONIterableDataset -> Tokenize(collate_type=classic) ->
PaddingDataLoader -> CycleDataLoader -> TorchDataLoader -> run.py`. It uses
`configs/smoke/data_classic/tiny_classic.json` with explicit `input_ids`, a
cached `sshleifer/tiny-gpt2` tokenizer for padding metadata, completes two
scheduler steps, and writes
`/tmp/mohawk_tiny_classic_collate_cpu_run/save`. It does not validate the
remote `jondurbin/airoboros-2.2` dataset, nontrivial pre-tokenized dataset
schemas, CUDA, DDP/FSDP, distributed data sharding, or full-size training.

Tiny CPU KV/raw packing smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_kv_raw_packing.yaml
```

This validates only a tiny synthetic key-value retrieval path with
`RoundRobinLoader -> KVRetrieval -> Tokenize(collate_type=raw) ->
PackingDataLoader -> CycleDataLoader -> TorchDataLoader -> run.py`. It uses
`num_pairs: 1`, `seed: 0`, cached `sshleifer/tiny-gpt2` tokenization,
packed `input_ids`, `position_ids`, and `attention_mask`, completes two
scheduler steps, and writes `/tmp/mohawk_tiny_kv_raw_packing_cpu_run/save`.
It does not validate large KV retrieval runs, mixed real-data round-robin
ratios, long-context packing, CUDA, DDP/FSDP, distributed data sharding, or
full-size training.

Tiny CPU Needle-in-a-Haystack dataset smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_niah_dataset_supervised.yaml
```

This validates only a tiny synthetic `NeedleInHaystackDataset` path with
`NeedleInHaystackDataset -> CycleDataLoader -> TorchDataLoader -> run.py`. It
uses cached `sshleifer/tiny-gpt2` tokenization with `local_files_only: true`,
`type_haystack: none`, one key/value query, 96-token padded `input_ids`,
3-token `response_ids`, completes two scheduler steps, and writes
`/tmp/mohawk_tiny_niah_dataset_cpu_run/save`. It does not validate essay
haystacks, long-context needle retrieval, multi-query needles, CUDA,
DDP/FSDP, distributed data sharding, or full-size training.

Tiny CPU CopyingTaskDataset smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_copying_task_supervised.yaml
```

This validates only a tiny synthetic copying-task path with
`CopyingTaskDataset -> CycleDataLoader -> TorchDataLoader -> run.py`. It uses
cached `sshleifer/tiny-gpt2` tokenization with `local_files_only: true`,
`max_seq_len: 24`, padded 24-token `input_ids`, completes two scheduler
steps, and writes `/tmp/mohawk_tiny_copying_task_cpu_run/save`. It does not
validate long copying contexts, richer copying distributions, CUDA, DDP/FSDP,
distributed data sharding, or full-size training.

Tiny CPU SequentialLoader smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_sequential_loader_supervised.yaml
```

This validates only a tiny local two-source sequence path with
`SequentialLoader -> PaddingDataLoader -> CycleDataLoader -> TorchDataLoader ->
run.py`. It uses `configs/smoke/data_sequence_a/tiny_sequence_a.json` and
`configs/smoke/data_sequence_b/tiny_sequence_b.json`, cached
`sshleifer/tiny-gpt2` tokenization, `max_samples: 1`, and two distinct padded
batches before completing two scheduler steps and writing
`/tmp/mohawk_tiny_sequential_loader_cpu_run/save`. It does not validate large
multi-stage sequential curricula, remote datasets, long-context packing, CUDA,
DDP/FSDP, distributed data sharding, or full-size training.

Tiny CPU AggregationDataLoader smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_aggregation_loader_supervised.yaml
```

This validates only a tiny local two-source aggregation path with
`RoundRobinLoader -> PackingDataLoader -> AggregationDataLoader ->
CycleDataLoader -> TorchDataLoader -> run.py`. It uses
`configs/smoke/data_aggregation_a/tiny_aggregation_a.json` and
`configs/smoke/data_aggregation_b/tiny_aggregation_b.json`, cached
`sshleifer/tiny-gpt2` tokenization, `aggregation_size: 2`, packed
`input_ids`, `position_ids`, and `attention_mask`, completes two scheduler
steps, and writes `/tmp/mohawk_tiny_aggregation_loader_cpu_run/save`. It does
not validate large aggregation buffers, remote multi-source round-robin
datasets, long-context packing, CUDA, DDP/FSDP, distributed data sharding, or
full-size training.

Tiny CPU Qwen2 model-family `run.py` smoke:

```bash
python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cpu_qwen2_supervised.yaml \
  --output /tmp/mohawk_tiny_qwen2_teacher_ckpt

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py --config configs/smoke/tiny_cpu_qwen2_supervised.yaml
```

This validates only the tiny CPU
`LayeredMambaLM -> Qwen2Model -> Qwen2Block -> Qwen2Attention` entrypoint path.
It does not validate full-size Qwen2 configs, CUDA, DDP/FSDP, SSM/Mamba,
Phi/Falcon entrypoints, or representative checkpoints.

Tiny CPU PhiAttention model-family `run.py` smoke:

```bash
python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cpu_phi_supervised.yaml \
  --output /tmp/mohawk_tiny_phi_teacher_ckpt

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py --config configs/smoke/tiny_cpu_phi_supervised.yaml
```

This validates only the tiny CPU
`LayeredMambaLM -> LlamaModel -> MambaPhi -> PhiAttention` entrypoint path. It
does not validate full-size Phi configs, CUDA, DDP/FSDP, SSM/Mamba package
dependencies, or representative checkpoints.

Tiny CPU FalconBlock model-family `run.py` smoke:

```bash
python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cpu_falcon_supervised.yaml \
  --output /tmp/mohawk_tiny_falcon_teacher_ckpt

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py --config configs/smoke/tiny_cpu_falcon_supervised.yaml
```

This validates only the tiny CPU
`LayeredMambaLM -> LlamaModel -> FalconBlock -> FalconMambaMixer` entrypoint
path with Transformers' sequential Mamba fallback. It does not validate
full-size Falcon configs, CUDA, DDP/FSDP, fast Mamba kernels, SSM/Mamba package
dependencies, or representative checkpoints.

Tiny CPU DiscreteMamba2 reference `run.py` smoke:

```bash
python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cpu_discrete_mamba2_ref_supervised.yaml \
  --output /tmp/mohawk_tiny_discrete_mamba2_ref_teacher_ckpt

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_discrete_mamba2_ref_supervised.yaml
```

This validates only the tiny CPU pure-PyTorch reference path for
`LayeredMambaLM -> LlamaModel -> LlamaBlock -> DiscreteMamba2` with
`use_ref_impl: true`. The validated run completed two scheduler steps, saved
`/tmp/mohawk_tiny_discrete_mamba2_ref_cpu_run/save`, and produced model hash
`d0c1a65b9aa684fda86f08e60addbaaaf22e28fac02faaa32fd433d7d852c633`.
The dependency-light CPU environment still needs no `mamba_ssm` or
`causal_conv1d`. The optional storage-backed CUDA environment now validates the
fast path separately. A direct GB300 probe with `mamba-ssm==2.2.6.post3`,
`causal-conv1d==1.6.2.post1`, `use_ref_impl: false`, and bfloat16 input produced
finite output and input gradients plus a nonzero parameter-gradient sum.

Tiny fast-kernel `run.py` smokes:

```bash
python run.py \
  --config configs/smoke/tiny_cuda_doubleblock_discrete_mamba2_fast_supervised.yaml,configs/smoke/tiny_cuda_doubleblock_vanilla_discrete_mamba2_fast_supervised.yaml,configs/smoke/tiny_cuda_doubleblock_hymba_discrete_mamba2_fast_supervised.yaml,configs/smoke/tiny_cuda_doubleblock_merger_discrete_mamba2_fast_supervised.yaml

torchrun --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_ddp_doubleblock_discrete_mamba2_fast_supervised.yaml,configs/smoke/tiny_cuda_fsdp_doubleblock_discrete_mamba2_fast_supervised.yaml,configs/smoke/tiny_cuda_ddp_doubleblock_vanilla_discrete_mamba2_fast_supervised.yaml,configs/smoke/tiny_cuda_fsdp_doubleblock_vanilla_discrete_mamba2_fast_supervised.yaml,configs/smoke/tiny_cuda_ddp_doubleblock_hymba_discrete_mamba2_fast_supervised.yaml,configs/smoke/tiny_cuda_fsdp_doubleblock_hymba_discrete_mamba2_fast_supervised.yaml,configs/smoke/tiny_cuda_ddp_doubleblock_merger_discrete_mamba2_fast_supervised.yaml,configs/smoke/tiny_cuda_fsdp_doubleblock_merger_discrete_mamba2_fast_supervised.yaml
```

On Slurm GB300 nodes, current-code centralized smoke job `746718` and two-rank
DDP/FSDP smoke job `746732` completed two scheduler steps for
`DoubleBlockAdapter`, `DoubleBlockVanilla`, `DoubleBlockHymba`, and
`DoubleBlockMerger` with `DiscreteMamba2 use_ref_impl: false`. Artifact
verification found finite float32 tensors, scheduler `2/2`, optimizer state
counts `19`, `17`, `23`, and `21`, and matching DDP/FSDP hashes per variant:
Adapter `0041dc4ab3b24ccff2d5368840b72f6499ca1385d6e1c63dfa96c22bc8588f6f`,
Vanilla `c50c1bfa8e56246bff13e2e3b51d72e118d11f4a74b8f5d0bc0b7405f6ef0ce6`,
Hymba `afeb8c1bfccfa871751015e2c98635528ef45a2000e8ca13de806cba42b92bea`,
and Merger `29aa7cd4fbc0c734fc575b015b26bafaaa5c6f044cf0eb0bde179813682bba2a`.
This proves only tiny one-node DoubleBlock-family fast Mamba2 paths. It does
not validate production hybrid configs, long-context SSM, or representative
checkpoints. Tiny multi-node SSM evidence is recorded separately below.

Tiny two-node fast-kernel DDP/FSDP smokes:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive \
  --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
  --time=00:18:00 --immediate=120 \
  bash -lc 'nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); \
  export MASTER_ADDR=${nodes[0]} MASTER_PORT=29731; \
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled; \
  /home/abick/storage/mohawk/venvs/ssm-cuda-20260626/bin/torchrun \
  --nnodes=$SLURM_NNODES --nproc_per_node=2 \
  --node_rank=$SLURM_NODEID --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT run.py \
  --config configs/smoke/tiny_cuda_ddp_multinode_doubleblock_discrete_mamba2_fast_supervised.yaml'

# Repeat with a distinct rendezvous port and the FSDP config:
# configs/smoke/tiny_cuda_fsdp_multinode_doubleblock_discrete_mamba2_fast_supervised.yaml
```

Current-code Slurm jobs `747896` and `747933` ran the DDP and FSDP configs on
two GB300 nodes with two local ranks per node (`WORLD_SIZE=4`). Both loaded
19/19 teacher keys, used gradient accumulation with DDP/FSDP `no_sync`,
completed scheduler `2/2`, saved non-empty optimizer state for all 19
parameters, and destroyed the process group cleanly. FSDP reported
`FULL_SHARD`. The two saves contain 19 finite float32 tensors and have the same
model SHA-256,
`0fe968211b13f8e38a20391db9fe7aca14698f159f215848ed9dbedaae792272`.
This proves tiny two-node, four-rank fast-Mamba training, not production scale,
more than two nodes, representative sequence lengths, production Qwen/Llama
hybrids, or representative checkpoints.

Fresh Mamba 2.3 validation uses
`configs/smoke/tiny_cuda_doubleblock_discrete_mamba2_fast_mamba23_supervised.yaml`
and the isolated storage venv documented above. Slurm job `748528` imported
Mohawk in `17.742s` on an NVIDIA GB300 with `mamba-ssm 2.3.2.post1`, kept
`utils` resolved to the repository, and ran a finite bfloat16 fast
forward/backward with parameter-gradient sum `0.1579778864979744`. Job
`748589` then loaded 19/19 teacher keys, completed two optimizer steps, and
saved 19 finite float32 tensors with 19 optimizer states and SHA-256
`f7c108e9d676a1a4e81f1ffe0d95d5639e345cf8696359c009305940305224cf`.
Generation job `748805` loaded that checkpoint and produced
`hello anguish played` in `7ms`. This validates the tiny single-node Mamba 2.3
training and incremental-generation paths only.

Tiny CPU DoubleBlock DiscreteMamba2 reference `run.py` smokes:

```bash
python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cpu_doubleblock_discrete_mamba2_ref_supervised.yaml \
  --output /tmp/mohawk_tiny_doubleblock_discrete_mamba2_ref_teacher_ckpt

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_doubleblock_discrete_mamba2_ref_supervised.yaml
```

Equivalent configs exist for `vanilla`, `hymba`, and `merger`:
`configs/smoke/tiny_cpu_doubleblock_vanilla_discrete_mamba2_ref_supervised.yaml`,
`configs/smoke/tiny_cpu_doubleblock_hymba_discrete_mamba2_ref_supervised.yaml`,
and
`configs/smoke/tiny_cpu_doubleblock_merger_discrete_mamba2_ref_supervised.yaml`.
These validate only the tiny CPU
`LayeredMambaLM -> LlamaModel -> DoubleBlock*` paths with `mixer1` as
`DiscreteMamba2 use_ref_impl: true` and `mixer2` as `LlamaAttention`.
The validated runs loaded all teacher keys (`19/19`, `17/17`, `23/23`, and
`21/21`), completed two scheduler steps each, and saved
`/tmp/mohawk_tiny_doubleblock_discrete_mamba2_ref_cpu_run/save`,
`/tmp/mohawk_tiny_doubleblock_vanilla_discrete_mamba2_ref_cpu_run/save`,
`/tmp/mohawk_tiny_doubleblock_hymba_discrete_mamba2_ref_cpu_run/save`, and
`/tmp/mohawk_tiny_doubleblock_merger_discrete_mamba2_ref_cpu_run/save`.
The model hashes were
`fee4e7c14156ce0404efa63f631a89d6b1effb895f4f5c77e40696a4035ddc46`,
`1c779255a7d2b4920e3c09d87f1e51efc6f7a388ff44c5bd4f513d0bebb648a7`,
`36da178993d39692d8c1cba3163fdc78b0f6ad330c0bbd7fa6b5065297c8d449`,
and `56bb1d2f4226b9f1fa5ce9b945052204e4254a8bc9aeebdddae868e75525b3cc`.
The Hymba tiny run changed 17 of 23 tensors; the zero-initialized gate setup
left six tensors unchanged in this two-step smoke. These smokes do not validate
fast Mamba kernels, CUDA, DDP/FSDP, production Qwen/Llama hybrid configs,
transfer tooling, or representative checkpoints.

Tiny CUDA DoubleBlock DiscreteMamba2 reference `run.py` smokes:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:08:00 --immediate=120 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cuda_doubleblock_discrete_mamba2_ref_supervised.yaml
```

Equivalent CUDA overlays exist for
`configs/smoke/tiny_cuda_doubleblock_vanilla_discrete_mamba2_ref_supervised.yaml`,
`configs/smoke/tiny_cuda_doubleblock_hymba_discrete_mamba2_ref_supervised.yaml`,
and
`configs/smoke/tiny_cuda_doubleblock_merger_discrete_mamba2_ref_supervised.yaml`.
The four configs reuse the tiny `LayeredMambaLM -> LlamaModel ->
DoubleBlockAdapter/Vanilla/Hymba/Merger` paths with `DiscreteMamba2
use_ref_impl: true` and `LlamaAttention`, but use shared Slurm-visible paths
under `/home/abick/mohawk/artifacts/gpu_smoke`. The teacher checkpoints are the
verified shared checkpoints under
`/home/abick/mohawk/artifacts/gpu_smoke/doubleblock_ref_checkpoints`.
The validated Slurm jobs `689221`, `689237`, `689254`, and `689262` loaded
teacher keys `19/19`, `17/17`, `23/23`, and `21/21` on `cuda:0`, trained
1,612,596, 1,612,324, 1,612,660, and 1,612,628 parameters for two scheduler
steps, and saved
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_doubleblock_discrete_mamba2_ref/save`,
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_doubleblock_vanilla_discrete_mamba2_ref/save`,
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_doubleblock_hymba_discrete_mamba2_ref/save`,
and
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_doubleblock_merger_discrete_mamba2_ref/save`.
Artifact verification found tensor counts `19`, `17`, `23`, and `21`, all
`torch.float32`, optimizer state entries `19`, `17`, `23`, and `21`, and model
hashes `26410b59df6bd130e8b87ca6ce4f35a4040fb7dad89edc6e4392afc47c85eb0c`,
`688309cc0e531b8335568d7bcc5739087fcbf9ad82729bea0efa5711b1465053`,
`668075227edf9868305ccbbcfcd287f81a51c3325acc7bbc3eb25efe57aca8fd`, and
`282710e0946b6830ef1d3026e4a456d572e797c2742911053293df786dd85459`.
Adapter, vanilla, and merger changed all common tensors from their teacher
checkpoints; the Hymba tiny run changed 10 of 23 tensors in this two-step smoke.
These smokes do not validate fast Mamba kernels, multi-node distributed hybrid
training, production Qwen/Llama hybrid configs, transfer tooling, or
representative checkpoints.

Tiny CUDA DDP/FSDP DoubleBlock DiscreteMamba2 reference `run.py` smokes:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:10:00 --immediate=120 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled torchrun --standalone \
  --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_ddp_doubleblock_discrete_mamba2_ref_supervised.yaml
```

The matching FSDP adapter overlay is
`configs/smoke/tiny_cuda_fsdp_doubleblock_discrete_mamba2_ref_supervised.yaml`.
Equivalent DDP/FSDP overlays exist for vanilla, hymba, and merger:
`configs/smoke/tiny_cuda_{ddp,fsdp}_doubleblock_vanilla_discrete_mamba2_ref_supervised.yaml`,
`configs/smoke/tiny_cuda_{ddp,fsdp}_doubleblock_hymba_discrete_mamba2_ref_supervised.yaml`,
and
`configs/smoke/tiny_cuda_{ddp,fsdp}_doubleblock_merger_discrete_mamba2_ref_supervised.yaml`.
All eight configs use tiny `LayeredMambaLM -> LlamaModel ->
DoubleBlockAdapter/Vanilla/Hymba/Merger` paths with `DiscreteMamba2
use_ref_impl: true`, `LlamaAttention`, `configs/smoke/data_ddp`,
`effective_batch_size: 2`, `n_tokens: 32`, and the matching shared teacher
checkpoint. The DDP jobs `689318`, `689363`, `689378`, and `689404` used
`WORLD_SIZE=2`, wrapper `ddp`, `NO_SHARD`, broadcast weights from rank 0,
loaded teacher keys `19/19`, `17/17`, `23/23`, and `21/21`, completed two
scheduler steps, and saved the four
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_ddp_doubleblock*_discrete_mamba2_ref/save`
checkpoints. The FSDP jobs `689324`, `689371`, `689396`, and `689411` used
`WORLD_SIZE=2`, wrapper `fsdp`, `FULL_SHARD`, completed FSDP setup for student
and teacher, loaded the same teacher-key counts, completed two scheduler steps,
and saved the four
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_fsdp_doubleblock*_discrete_mamba2_ref/save`
checkpoints. Artifact verification found DDP/FSDP hashes matched per variant:
adapter `86be1ef437abce55cc58f7bffd6f3fdeecab0639a4014967314935dc30041bf4`,
vanilla `ebd61d3b2f1d789631a25612b27c51ca0ed8a5ef0ab516e32aa54a85fde2df8a`,
hymba `9ec928a4d5863363f05235ba63eb7a065f3152f4e5a0867f34feb85217614d98`,
and merger `12a16f0b147becab00aa2e6e3a69895368f215bb3c871b0e903c21e4faf2f6e7`.
Adapter, vanilla, and merger changed all common tensors from their teacher
checkpoints; Hymba changed 10 of 23 tensors in this two-step smoke. These
smokes do not validate fast Mamba kernels, multi-node distributed hybrid
training, production Qwen/Llama hybrid configs, transfer tooling, or
representative checkpoints.

Generated GPU smoke artifacts are stored outside the repo workspace at
`/home/abick/storage/mohawk/artifacts/gpu_smoke`; the repo path
`/home/abick/mohawk/artifacts/gpu_smoke` is a symlink to that storage location,
so the evidence paths above still resolve while keeping the workspace small.

Tiny CPU DoubleBlock attention-only `run.py` smokes:

```bash
python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cpu_doubleblock_adapter_supervised.yaml \
  --output /tmp/mohawk_tiny_doubleblock_adapter_teacher_ckpt

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_doubleblock_adapter_supervised.yaml
```

Equivalent configs exist for `vanilla`, `hymba`, and `merger`. These validate
only the tiny CPU `DoubleBlockAdapter`, `DoubleBlockVanilla`,
`DoubleBlockHymba`, and `DoubleBlockMerger` entrypoint paths with
`LlamaAttention` used as both tiny mixers. The DiscreteMamba2 reference hybrid
smokes above cover bounded SSM reference mixer paths, but these
attention-only smokes still do not validate fast SSM/Mamba mixers, production
Qwen/Llama hybrid configs, CUDA, DDP/FSDP, or representative checkpoints.

Tiny CPU `hstates` objective smoke after creating its local teacher checkpoint:

```bash
python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cpu_hstates.yaml \
  --output /tmp/mohawk_tiny_hstates_teacher_ckpt

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py --config configs/smoke/tiny_cpu_hstates.yaml
```

This validates only the tiny CPU hstates objective through the real `run.py`
path. The remaining objective types still need their own real entrypoint runs
before they can be claimed as end-to-end training paths.

Tiny CPU `eval_hstates` callback smoke after creating its local teacher
checkpoint:

```bash
python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cpu_eval_hstates.yaml \
  --output /tmp/mohawk_tiny_eval_hstates_teacher_ckpt

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_eval_hstates.yaml
```

This validates only the tiny CPU wrapper-driven `eval_hstates` callback through
the real `run.py` evaluator path. It does not validate representative hidden
state evaluation data, CUDA, DDP/FSDP, SSM/Mamba, or full-size checkpoints.

Tiny CPU bfloat16 wrapper-driven `eval_hstates` callback smoke using the same
local teacher checkpoint:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_bfloat16_eval_hstates.yaml
```

This validates only the tiny CPU path where bfloat16 student and teacher
wrappers run the `EvalConfig -> eval_hstates` callback and `save_latest`
checkpoint path. The centralized wrapper logs the bfloat16 AMP fallback, writes
`/tmp/mohawk_tiny_bfloat16_eval_hstates_cpu_run/save/latest_eval_hstates`, and
skips `dataloader_state_dict.pth` because the local JSON training loader is not
resumable. It does not validate CUDA bfloat16, distributed wrappers,
representative hidden-state eval data, resumable production dataloader
checkpoints, or full-size checkpoints.

Tiny CUDA/DDP/FSDP wrapper-driven `eval_hstates` callback smokes:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:08:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  python run.py --config configs/smoke/tiny_cuda_eval_hstates.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:08:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  torchrun --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_ddp_eval_hstates.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:10:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  torchrun --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_fsdp_eval_hstates.yaml
```

These validate tiny CUDA, two-rank DDP, and two-rank FSDP FULL_SHARD
`EvalConfig -> eval_hstates` paths with local JSON eval data and
`save_latest`. The FSDP run initially failed during full-state export because
`eval_hstates` used `torch.inference_mode()`; it passed after switching the
evaluator to `torch.no_grad()`. They do not validate representative
hidden-state eval data, multi-node distributed eval, SSM/Mamba, or full-size
checkpoints.

Tiny CUDA/DDP/FSDP bfloat16 wrapper-driven `eval_hstates` callback smokes:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:08:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  python run.py --config configs/smoke/tiny_cuda_bfloat16_eval_hstates.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:08:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  torchrun --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_ddp_bfloat16_eval_hstates.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:10:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  torchrun --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_fsdp_bfloat16_eval_hstates.yaml
```

These validate only tiny Llama-style CUDA bfloat16 wrapper callback paths on one
Slurm node. The single-process smoke writes
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_bfloat16_eval_hstates/save/latest_eval_hstates`
with `eval_score: 1.640625`; DDP and FSDP write the corresponding
`tiny_cuda_ddp_bfloat16_eval_hstates` and
`tiny_cuda_fsdp_bfloat16_eval_hstates` artifacts with
`eval_score: 1.664062`. Artifact verification recorded final/latest tensors as
`torch.bfloat16`, single-process hash
`52fe58e628de69a168b351d41c8d01a52017300497a0675770ae75653ccf1ef1`, and
DDP/FSDP hash `0bc9e79b22d4fce8a926de7260a144c0a5a0ff09a86328af5f35777ab655af15`.
These smokes do not validate representative hidden-state eval data, multi-node
DDP/FSDP eval, SSM/Mamba, or full-size checkpoints.

Tiny CPU bfloat16 `hstates` objective smoke using the same local hstates
teacher checkpoint:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_bfloat16_hstates.yaml
```

This validates only the tiny CPU hstates objective path while bfloat16 student
and teacher wrappers are configured. The centralized wrapper logs the bfloat16
AMP fallback and writes `/tmp/mohawk_tiny_bfloat16_hstates_cpu_run/save` with
saved tensors as `torch.bfloat16`. It does not validate CUDA bfloat16,
DDP/FSDP, SSM/Mamba, representative hidden-state data, or full-size
checkpoints.

Tiny CPU `sequential_hstates` objective smoke after creating its local teacher
checkpoint:

```bash
python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cpu_sequential_hstates.yaml \
  --output /tmp/mohawk_tiny_sequential_hstates_teacher_ckpt

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py --config configs/smoke/tiny_cpu_sequential_hstates.yaml
```

This validates only the tiny CPU sequential_hstates objective through the real
`run.py` path. It does not validate DPO, instruction distillation, CUDA,
distributed wrappers, or representative resources.

Tiny CPU bfloat16 `sequential_hstates` objective smoke using the same local
sequential_hstates teacher checkpoint:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_bfloat16_sequential_hstates.yaml
```

This validates only the tiny CPU sequential_hstates objective path while
bfloat16 student and teacher wrappers are configured. The centralized wrapper
logs the bfloat16 AMP fallback, requests teacher hidden states, and writes
`/tmp/mohawk_tiny_bfloat16_sequential_hstates_cpu_run/save` with saved tensors
as `torch.bfloat16`. It does not validate CUDA bfloat16, DDP/FSDP, SSM/Mamba,
representative hidden-state data, or full-size checkpoints.

Tiny CPU `matrices` objective smoke after creating its local teacher checkpoint:

```bash
python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cpu_matrices.yaml \
  --output /tmp/mohawk_tiny_matrices_teacher_ckpt

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py --config configs/smoke/tiny_cpu_matrices.yaml
```

This validates only the tiny CPU matrices objective through the real `run.py`
path, including teacher attention-matrix output. It does not validate DPO,
instruction distillation, CUDA, distributed wrappers, or representative
resources.

Tiny CPU bfloat16 `matrices` objective smoke using the same local matrices
teacher checkpoint:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_bfloat16_matrices.yaml
```

This validates only the tiny CPU matrices objective path while bfloat16 student
and teacher wrappers are configured. The centralized wrapper logs the bfloat16
AMP fallback, requests teacher attention matrices, and writes
`/tmp/mohawk_tiny_bfloat16_matrices_cpu_run/save` with saved tensors as
`torch.bfloat16`. It does not validate CUDA bfloat16, DDP/FSDP, SSM/Mamba,
representative attention-matrix data, or full-size checkpoints.

Tiny CPU DPO objective smoke after creating its local teacher checkpoint:

```bash
python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cpu_dpo.yaml \
  --output /tmp/mohawk_tiny_dpo_teacher_ckpt

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py --config configs/smoke/tiny_cpu_dpo.yaml
```

This validates only the tiny CPU DPO objective through the real `run.py` path
with a local one-sample preference JSON. It does not validate instruction
distillation, CUDA, distributed wrappers, or representative preference data.

Tiny CPU bfloat16 DPO objective smoke using the same local DPO teacher
checkpoint:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_bfloat16_dpo.yaml
```

This validates only the tiny CPU DPO objective path while bfloat16 student and
teacher wrappers are configured. The centralized wrapper logs the bfloat16 AMP
fallback, uses `Tokenize(collate_type=preference)` with `chosen_ids` and
`rejected_ids`, and writes `/tmp/mohawk_tiny_bfloat16_dpo_cpu_run/save` with
saved tensors as `torch.bfloat16`. It does not validate CUDA bfloat16,
DDP/FSDP, SSM/Mamba, representative preference datasets, or full-size
checkpoints.

Tiny CPU supervised-instruction objective smoke after creating its local teacher
checkpoint:

```bash
python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cpu_supervised_instruct.yaml \
  --output /tmp/mohawk_tiny_supervised_instruct_teacher_ckpt

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py --config configs/smoke/tiny_cpu_supervised_instruct.yaml
```

This validates only the tiny CPU supervised-instruction objective through the
real `run.py` path with a local one-sample instruction JSON. It does not
validate chat-template tokenizers, CUDA, distributed wrappers, or
representative instruction data.

Tiny CPU supervised-instruction teacher-logits supervision smoke using the same
local supervised-instruction teacher checkpoint:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_supervised_instruct_teacher_logits.yaml
```

This validates only the tiny CPU supervised-instruction branch where
`supervision: true` and `generate: false`, so the teacher supplies response-span
logit distributions through the real wrapper. It writes
`/tmp/mohawk_tiny_supervised_instruct_teacher_logits_cpu_run/save` and does not
validate chat-template tokenizers, CUDA, distributed wrappers, or
representative instruction data.

Tiny CPU supervised-instruction generate-supervision smoke using the same local
supervised-instruction teacher checkpoint:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_supervised_instruct_generate.yaml
```

This validates only the tiny CPU supervised-instruction branch where
`supervision: true` and `generate: true`, so the teacher wrapper generates a
bounded response with `generation_max_new_tokens: 2` before the student trains
on the response span. It writes
`/tmp/mohawk_tiny_supervised_instruct_generate_cpu_run/save` and does not
validate chat-template tokenizers, CUDA, distributed wrappers, or
representative instruction data.

Tiny CPU bfloat16 supervised-instruction objective smoke using the same local
supervised-instruction teacher checkpoint:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_bfloat16_supervised_instruct.yaml
```

This validates only the tiny CPU supervised-instruction objective path while
bfloat16 student and teacher wrappers are configured. The centralized wrapper
logs the bfloat16 AMP fallback, uses `Tokenize(collate_type=instruction,
return_dict=true)` with `input_ids` and `response_ids`, and writes
`/tmp/mohawk_tiny_bfloat16_supervised_instruct_cpu_run/save` with saved tensors
as `torch.bfloat16`. It does not validate CUDA bfloat16, DDP/FSDP, SSM/Mamba,
chat-template tokenizers, bfloat16 teacher-logits/generation-supervision modes,
representative instruction datasets, or full-size checkpoints.

Tiny CPU checkpoint resume smoke after running the supervised training smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py --config configs/smoke/tiny_cpu_resume.yaml
```

This validates only centralized CPU `run.py` resume from
`/tmp/mohawk_tiny_cpu_run/save` into `/tmp/mohawk_tiny_resume_cpu_run/save`,
including model, optimizer, and scheduler restore. Mixed-precision and
bfloat16 resume are covered by separate tiny CPU smokes below; this base smoke
does not validate CUDA, DDP/FSDP, or representative checkpoints.

Tiny CPU mixed-precision checkpoint resume smoke after running the
mixed-precision smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_mixed_precision_resume.yaml
```

This validates only centralized CPU `run.py` resume from
`/tmp/mohawk_tiny_mixed_precision_cpu_run/save` into
`/tmp/mohawk_tiny_mixed_precision_resume_cpu_run/save` while
`TrainConfig.mixed_precision: true` and `TeacherConfig.mixed_precision: true`
remain enabled. It restores model, optimizer, and scheduler state, continues
from scheduler step 2 to step 4, and does not validate CUDA or DDP/FSDP resume.

Tiny CPU bfloat16 checkpoint resume smoke after running the bfloat16 fallback
smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_bfloat16_resume.yaml
```

This validates only centralized CPU `run.py` resume from
`/tmp/mohawk_tiny_bfloat16_fallback_cpu_run/save` into
`/tmp/mohawk_tiny_bfloat16_resume_cpu_run/save` while
`TrainConfig.model_dtype: bfloat16`, `TeacherConfig.model_dtype: bfloat16`,
`TrainConfig.mixed_precision: true`, and `TeacherConfig.mixed_precision: true`
remain configured. The wrappers disable AMP for bfloat16, model/optimizer/
scheduler state is restored, scheduler step 2 advances to step 4, and the saved
checkpoint tensors remain `torch.bfloat16`. It does not validate CUDA bfloat16
or DDP/FSDP resume.

Tiny CPU public HFDataset dataloader-state resume after running the public
HFDataset latest-state smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_hfdata_dataloader_resume.yaml
```

This validates only `LoadConfig.dataloader` restore for the tiny public
`HFDataset` path. It starts from the saved `_index: 2` state in
`/tmp/mohawk_tiny_eval_ppl_hfdata_cpu_run/save/latest_Perplexity` and writes a
new latest checkpoint with `_index: 4`. It does not by itself validate
distributed sharding, CUDA, DDP/FSDP, model/optimizer/scheduler resume combined
with dataloader state, or representative datasets.

Tiny CPU public HFDataset full checkpoint/dataloader resume after running the
public HFDataset latest-state smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_hfdata_full_resume.yaml
```

This loads model, optimizer, scheduler, and `LoadConfig.dataloader` from
`/tmp/mohawk_tiny_eval_ppl_hfdata_cpu_run/save/latest_Perplexity`. The source
checkpoint has scheduler step 2 and saved `_index: 2`; the full-resume smoke
requests four total scheduler steps, saves scheduler step 4, and writes a new
latest checkpoint with `_index: 5`. It remains scoped to centralized tiny CPU
public `NeelNanda/pile-10k` data and does not validate distributed sharding,
CUDA, DDP/FSDP, SSM/Mamba, or representative checkpoints.

Tiny CPU bfloat16 public HFDataset full checkpoint/dataloader resume after
running the bfloat16 public HFDataset latest-state smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_bfloat16_hfdata_full_resume.yaml
```

This loads model, optimizer, scheduler, and `LoadConfig.dataloader` from
`/tmp/mohawk_tiny_bfloat16_eval_ppl_hfdata_cpu_run/save/latest_Perplexity`
while keeping the student and teacher wrappers configured for
`model_dtype: bfloat16`. The source checkpoint has scheduler step 2 and saved
`_index: 2`; the full-resume smoke requests four total scheduler steps, saves
scheduler step 4, writes a new latest checkpoint with `_index: 5`, and keeps
saved tensors as `torch.bfloat16`. It remains scoped to centralized tiny CPU
public `NeelNanda/pile-10k` data and does not validate CUDA bfloat16,
distributed sharding, DDP/FSDP, SSM/Mamba, or representative checkpoints.

Tiny CPU lazy-load checkpoint smoke after running the supervised training
smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py --config configs/smoke/tiny_cpu_lazy_load.yaml
```

This validates only centralized CPU `run.py` lazy student initialization from
`/tmp/mohawk_tiny_cpu_run/save` into `/tmp/mohawk_tiny_lazy_cpu_run/save`,
using `TrainConfig.init_fn: lazy` and `LoadConfig.model`. It does not validate
optimizer or scheduler resume, CUDA, DDP/FSDP, or representative checkpoints.

Tiny CPU mixed-precision lazy-load checkpoint smoke after running the
mixed-precision smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_mixed_precision_lazy_load.yaml
```

This validates only centralized CPU lazy student initialization from
`/tmp/mohawk_tiny_mixed_precision_cpu_run/save` into
`/tmp/mohawk_tiny_mixed_precision_lazy_cpu_run/save` while
`TrainConfig.mixed_precision: true` and `TeacherConfig.mixed_precision: true`
remain enabled. It does not validate optimizer or scheduler resume, CUDA,
DDP/FSDP, or representative checkpoints.

Tiny CPU bfloat16 lazy-load checkpoint smoke after running the bfloat16
fallback smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_bfloat16_lazy_load.yaml
```

This validates only centralized CPU lazy student initialization from
`/tmp/mohawk_tiny_bfloat16_fallback_cpu_run/save` into
`/tmp/mohawk_tiny_bfloat16_lazy_cpu_run/save` while
`TrainConfig.model_dtype: bfloat16`, `TeacherConfig.model_dtype: bfloat16`,
`TrainConfig.mixed_precision: true`, and `TeacherConfig.mixed_precision: true`
remain configured. It lazy-loads 12/12 checkpoint keys into a `meta`
initialized student, disables AMP for bfloat16, and saves `torch.bfloat16`
tensors after two scheduler steps. It does not validate optimizer or scheduler
resume, CUDA bfloat16, DDP/FSDP, or representative checkpoints.

Tiny CUDA lazy-load checkpoint smoke after running the tiny CUDA supervised
training smoke:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:06:00 --immediate=1 \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/python run.py \
  --config configs/smoke/tiny_cuda_lazy_load.yaml
```

This validates only centralized tiny CUDA lazy student initialization from
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_supervised/save` into
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_lazy_load/save`. The audited
run lazy-loaded 12/12 checkpoint keys into a `meta` initialized student, moved
the wrapper to `cuda:0`, completed two scheduler steps, and saved `torch.float32`
tensors. It does not validate optimizer or scheduler resume from a lazy
checkpoint, SSM/Mamba, representative configs, or full-size checkpoints.

Tiny CUDA DDP/FSDP lazy-load checkpoint smoke after running the tiny CUDA
DDP/FSDP supervised training smokes:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:08:00 --immediate=1 \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_ddp_lazy_load.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:10:00 --immediate=1 \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_fsdp_lazy_load.yaml
```

This validates only tiny one-node two-rank DDP/FSDP lazy student initialization
from `/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_ddp_supervised/save` and
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_fsdp_supervised/save` into the
matching `tiny_cuda_ddp_lazy_load/save` and `tiny_cuda_fsdp_lazy_load/save`
directories. The audited runs used `WORLD_SIZE=2`, loaded 12/12 checkpoint keys
on rank 0, exercised DDP rank-0 broadcast and FSDP `FULL_SHARD`, completed two
scheduler steps, and saved `torch.float32` tensors with hashes changed from the
source checkpoints. It does not validate optimizer or scheduler resume from a
lazy checkpoint, multi-node distributed lazy init, mixed-precision distributed
lazy init, SSM/Mamba, representative configs, or full-size checkpoints.

Tiny CPU `torchrun` single-process smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled torchrun --standalone --nproc_per_node=1 run.py \
  --config configs/smoke/tiny_cpu_supervised.yaml
```

This validates the real single-process `torchrun` launch path only when local
socket creation is allowed. It does not validate CUDA, multi-process DDP/FSDP,
representative configs, or full-size training.

Do not use the CPU opt-in as a substitute for distributed validation. A
two-process CPU `torchrun --standalone --nproc_per_node=2` probe reaches
`run.py` only when local rendezvous sockets are allowed, then fails on both
ranks with `RuntimeError: Mohawk training requires CUDA` before DDP/FSDP setup,
optimizer stepping, or checkpoint saving.

Tiny CUDA supervised smoke on a Slurm GPU allocation:

```bash
python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cuda_supervised.yaml \
  --output /home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_teacher_ckpt

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:06:00 --immediate=1 \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --standalone --nproc_per_node=1 run.py \
  --config configs/smoke/tiny_cuda_supervised.yaml
```

The audited run logged `WORLD_SIZE=1`, `Device: cuda:0`, completed two
scheduler steps, and saved
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_supervised/save` with model
hash `287b32e35195cb73a9a0f35493802b98f510bff5269543d6ca721750a8a2bb46`.
This validates only the tiny CUDA supervised path with local JSON data and a
cached public tokenizer.

Tiny CUDA mixed-precision and centralized resume smokes:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:06:00 --immediate=1 \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/python run.py \
  --config configs/smoke/tiny_cuda_mixed_precision.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:06:00 --immediate=1 \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/python run.py \
  --config configs/smoke/tiny_cuda_resume.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:06:00 --immediate=1 \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/python run.py \
  --config configs/smoke/tiny_cuda_mixed_precision_resume.yaml
```

The audited mixed-precision run kept `TrainConfig.mixed_precision: true` and
`TeacherConfig.mixed_precision: true`, completed two scheduler steps, and saved
hash `bd20786dba84dfb0975db85ea79cebeb1929e1658318fdbfa6e0d18ec91cb42f`.
The centralized CUDA resume smokes restored model, optimizer, and scheduler
state from scheduler step 2 to 4. The plain CUDA resume hash was
`17e40ae4debe2dac0f42d643bc662fc4618f921b106a72c80d542a30bcad046f`; the
mixed-precision CUDA resume hash was
`6e23c54372d768965deab8fc8442d8e842938a52f7c858075d7411d25fd82f4e`.
These validate only tiny centralized CUDA mixed precision and resume.

Tiny CUDA DDP/FSDP smokes on one Slurm node:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:08:00 --immediate=1 \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_ddp_supervised.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:08:00 --immediate=1 \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_fsdp_supervised.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:08:00 --immediate=1 \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_fsdp_activation_checkpointing.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:08:00 --immediate=1 \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_ddp_mixed_precision.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:08:00 --immediate=1 \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_fsdp_mixed_precision.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:08:00 --immediate=1 \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_ddp_resume.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:08:00 --immediate=1 \
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  /home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun \
  --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_fsdp_resume.yaml
```

`configs/smoke/data_ddp` supplies one tiny JSON shard per rank; the first DDP
attempt with the single-file smoke dataset hung because rank 1 received an
empty shard. The audited DDP run logged `WORLD_SIZE=2`, `wrapper_type: ddp`,
and teacher weight broadcast, then saved a checkpoint with hash
`01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55`.
The FSDP runs validated `FULL_SHARD` with activation checkpointing off and on.
The DDP wrapper was fixed to honor `mixed_precision` instead of unconditionally
entering AMP; the audited non-mixed DDP rerun has `mixed_precision False
False`, and the DDP/FSDP mixed-precision smokes both have `mixed_precision True
True`.
The DDP/FSDP resume smokes loaded model, optimizer, and scheduler checkpoint
files from their one-step source checkpoints, restored scheduler progress from
`1/1` to target scheduler `2/2`, and saved resumed checkpoints with hash
`14d447df4d4d3c6bfff71734e78cb36c71909b38ffb41be77c8b698739e5401b`. The
source distributed optimizer states had zero entries, so non-empty distributed
optimizer-state restore still needs a richer source checkpoint.
These smokes do not validate representative configs, multi-node training,
SSM/Mamba kernels, large or gated data, mixed-precision distributed resume, or
full-size checkpoints.

Tiny CPU comma-separated sequential smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_supervised.yaml,configs/smoke/tiny_cpu_supervised.yaml
```

This validates that the entrypoint can execute two real training configs in
sequence. It does not validate representative sequential train/eval chains,
CUDA, or full-size resources.

Tiny CPU comma-separated train+eval callback smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_supervised.yaml,configs/smoke/tiny_cpu_eval_ppl.yaml
```

This validates that the entrypoint can execute a real training config followed
by a second real training config with an `EvalConfig -> eval_ppl` callback in
the same comma-separated sequence. It writes `/tmp/mohawk_tiny_cpu_run/save`,
`/tmp/mohawk_tiny_eval_ppl_cpu_run/save`, and
`/tmp/mohawk_tiny_eval_ppl_cpu_run/save/latest_Perplexity`. It does not
validate representative sequential chains, CUDA, distributed wrappers, external
evaluation commands, or full-size resources.

Tiny CUDA comma-separated train+eval callback smoke:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:08:00 --immediate=120 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cuda_comma_supervised.yaml,configs/smoke/tiny_cuda_comma_eval_ppl.yaml
```

This validates only a tiny single-process CUDA sequence: one real supervised
training config followed by one real training config with an
`EvalConfig -> eval_ppl` callback. It writes storage-backed artifacts under
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_comma_supervised/save`,
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_comma_eval_ppl/save`, and
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_comma_eval_ppl/save/latest_Perplexity`.
It does not validate representative sequential chains, distributed wrappers,
external evaluation commands, large or gated datasets, or full-size resources.

A representative comma-separated Slurm probe using
`configs/Llama/8B/llama/hstates.yaml,configs/Llama/8B/llama/hstates.yaml`
parses and loads both config entries, but currently fails during the first
config's `RoundRobinLoader -> HFDataset` setup because
`HuggingFaceFW/fineweb-edu` is not available in offline/cache-only mode. It
does not reach model initialization, training, or the second config.

## How Configuration Works

Every run is driven by YAML. The important top-level sections are:

- `ComponentsConfig`: architecture definition (block sequence and layer types)
- `TrainConfig`: optimization schedule and training length
- `DistillConfig`: objective selection and logging run name
- `TeacherConfig`: teacher checkpoint/path and tokenizer context
- `TrainDataConfig`: dataset source and loader strategy
- `LoadConfig`: initialization and checkpoint loading rules
- `ManagementConfig`: cache paths, W&B config, environment defaults

Useful research templates:

- `configs/Qwen2/1.5B/hybrid/adapter.yaml`
- `configs/Llama/1B/hybrid/mohawk_8.yaml`
- `configs/Llama/8B/bases/_supervised.yaml`
- `configs/Llama/8B/llama/supervised.yaml`
- `configs/Llama/8B/llama/hstates.yaml`
- `configs/Llama/8B/llama/matrices.yaml`

Use wrapper configs for hybrid runs. Raw `architecture_*.yaml` files under
`hybrid/` are include fragments that rely on wrapper-provided constants such as
`BLOCK_NAME` and `ROPE_SCALING`; they are not standalone public entrypoint
configs.

The Llama 8B llama templates above load through the YAML system when a Slurm
constant is supplied, for example `CONSTANTS={'slurm_job_id': '0'}`. That only
validates config loadability; these templates still require the hardware,
credentials, datasets, and checkpoint paths listed above before they count as
real training or evaluation runs.

## Evaluation and Analysis

### Representative custom Transformers checkpoint

The model-facing CLIs accept repeatable `--model-registration-module MODULE`
arguments. Each named Python module is imported before Transformers resolves
the model config, allowing an installed optional package to register custom
`AutoConfig` and `AutoModelForCausalLM` types without making that package a
default Mohawk dependency.

The cached `AvivBick/raven-nslots256-topk64` checkpoint exercises this path. Its
`model.safetensors` is 1,697,254,264 bytes; the loaded Raven model has
424,303,808 parameters, 24 layers, and hidden size 1024. With the compatible
FLA source package on `PYTHONPATH`, the command shapes are:

```bash
python generation/generate.py \
  --model AvivBick/raven-nslots256-topk64 \
  --model-registration-module fla.models.raven \
  --local_files_only --model_dtype bfloat16 \
  --prompt "The future of language models is" --genlen 4 --repeats 1 --top_k 1

python evals/eval_ppl.py \
  --model AvivBick/raven-nslots256-topk64 --backend hf \
  --model-registration-module fla.models.raven \
  --local_files_only --device cuda --n_batches 1 --max_seq_len 64 --batch_size 1

python evals/benchmark.py \
  --dir /path/to/raven-nslots256-topk64-snapshot \
  --model-registration-module fla.models.raven \
  --tasks wikitext --batch_size 1 --num_fewshot 0 --limit 1 \
  --local_files_only --backend hf --device cuda

python tools/visualize_attention.py \
  --model_name AvivBick/raven-nslots256-topk64 \
  --model-registration-module fla.models.raven \
  --mode hidden_similarity --layers 0,12,24 \
  --output_dir /path/to/storage-backed-output --local_files_only --device cuda
```

Slurm jobs `751419`, `751508`, `751635`, and `752591` respectively generated
`The future of language models is a topic of much`, reported standalone PPL
`97.02066040039062` with accuracy `0.2916666567325592`, reported one-sample
Wikitext `61.629803`, and saved a visually inspected `512 x 906` RGB PNG with
SHA-256 `afaa1b151a6434f7bf24424241b236959062a085e15de242d30fda239a12e50f`.
The visualization uses token-to-token cosine similarity from hidden states
because Raven deliberately does not return attention tensors.

This is real non-tiny checkpoint evidence for the four public tool paths, not a
quality benchmark, a portable FLA installation claim, a representative Mohawk
hybrid checkpoint, full-dataset evaluation, distributed inference, or
attention-head visualization. The optional registration module and checkpoint
must already be installed or cached.

### Perplexity

`evals/eval_ppl.py` implements the evaluator class used by training/eval
wrappers. A tiny public Hugging Face smoke runs through the real evaluator,
real JSON/tokenizer/padding/Torch dataloader stack, and a cached model:

```bash
python evals/eval_ppl.py --model sshleifer/tiny-gpt2 \
  --local_files_only --device cpu --n_batches 1 --max_seq_len 16 \
  --batch_size 1 --text "hello world from mohawk"
```

This is not a Mohawk checkpoint or GPU wrapper-integrated perplexity run; those
remain release-readiness gates in [PUBLIC_SUPPORT_MATRIX.md](PUBLIC_SUPPORT_MATRIX.md).
The standalone PPL JSON includes `model_dtype` for the loaded model. For
`--backend mohawk`, `--model` must be a local Mohawk checkpoint directory with
`config.json`; passing a Hugging Face model ID is not a representative Mohawk
checkpoint PPL run.

Tiny CPU Mohawk checkpoint PPL after running the training smoke above:

```bash
python evals/eval_ppl.py --backend mohawk \
  --model /tmp/mohawk_tiny_cpu_run/save --local_files_only --device cpu \
  --n_batches 1 --max_seq_len 16 --batch_size 1 \
  --text "hello world from mohawk"
```

This only validates the tiny CPU Mohawk checkpoint path; it does not validate
representative checkpoints, CUDA, distributed wrappers, or full-size data.

Tiny CPU bfloat16 Mohawk checkpoint PPL after running the bfloat16 fallback
smoke above:

```bash
python evals/eval_ppl.py --backend mohawk \
  --model /tmp/mohawk_tiny_bfloat16_fallback_cpu_run/save \
  --local_files_only --device cpu --n_batches 1 --max_seq_len 16 \
  --batch_size 1 --text "hello world from mohawk"
```

This validates only the tiny CPU bfloat16 Mohawk checkpoint path. The standalone
PPL JSON includes `model_dtype` so this smoke records that the loaded checkpoint
ran as `torch.bfloat16`. The default `--backend auto` also routes this local
Mohawk checkpoint directory through the Mohawk backend. This does not validate
CUDA bfloat16, representative checkpoints, distributed wrappers, or full-size
data.

Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint PPL after running the
DoubleBlock reference training smokes:

```bash
python evals/eval_ppl.py --backend mohawk \
  --model /tmp/mohawk_tiny_doubleblock_discrete_mamba2_ref_cpu_run/save \
  --local_files_only --device cpu --n_batches 1 --max_seq_len 16 \
  --batch_size 1 --text "hello world from mohawk"
```

Equivalent commands also ran for
`/tmp/mohawk_tiny_doubleblock_vanilla_discrete_mamba2_ref_cpu_run/save`,
`/tmp/mohawk_tiny_doubleblock_hymba_discrete_mamba2_ref_cpu_run/save`, and
`/tmp/mohawk_tiny_doubleblock_merger_discrete_mamba2_ref_cpu_run/save`. The
four JSON outputs were `66080.9453125`, `42200.625`, `64897.3828125`, and
`70790.328125` perplexity respectively, all with
`"model_dtype": "torch.float32"`. This validates only tiny CPU DoubleBlock-family
checkpoints using the pure-PyTorch DiscreteMamba2 reference path; it does not
validate fast Mamba kernels, CUDA eval, distributed wrappers, production
Qwen/Llama hybrid checkpoints, representative checkpoints, or full-size data.

Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint PPL uses shared
checkpoint paths because Slurm compute nodes cannot see login-host `/tmp`
checkpoints. After copying the four reference checkpoints into
`/home/abick/mohawk/artifacts/gpu_smoke/doubleblock_ref_checkpoints`, the CUDA
Adapter command is:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:05:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  python evals/eval_ppl.py --backend mohawk \
  --model /home/abick/mohawk/artifacts/gpu_smoke/doubleblock_ref_checkpoints/adapter \
  --local_files_only --device cuda --n_batches 1 --max_seq_len 16 \
  --batch_size 1 --text "hello world from mohawk"
```

Equivalent CUDA commands also ran for `vanilla`, `hymba`, and `merger` under
the same shared directory. The four CUDA JSON outputs were `66080.9453125`,
`42200.62890625`, `64897.38671875`, and `70790.3203125` perplexity
respectively, all with `"model_dtype": "torch.float32"`. This validates only
tiny single-process CUDA DoubleBlock-family checkpoint PPL using the
pure-PyTorch DiscreteMamba2 reference path; it does not validate fast Mamba
kernels, distributed wrappers, production Qwen/Llama hybrid checkpoints,
representative checkpoints, or full-size data.

Tiny CUDA bfloat16 Mohawk checkpoint PPL after running the CUDA bfloat16
supervised smoke:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:05:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  python evals/eval_ppl.py --backend mohawk \
  --model /home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_bfloat16_supervised/save \
  --local_files_only --device cuda --n_batches 1 --max_seq_len 16 \
  --batch_size 1 --text "hello world from mohawk"
```

This validates only the tiny single-process CUDA bfloat16 Mohawk checkpoint PPL
path. The run printed `{"accuracy": 0.0, "eval_score": 49020.80859375,
"model_dtype": "torch.bfloat16", "perplexity": 49020.80859375}` from the
verified checkpoint whose saved tensors are all `torch.bfloat16` with hash
`52fe58e628de69a168b351d41c8d01a52017300497a0675770ae75653ccf1ef1`. It does
not validate representative checkpoints, distributed wrappers, or full-size
data.

Tiny CPU wrapper-driven `eval_ppl` callback smoke after creating its local
teacher checkpoint:

```bash
python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cpu_eval_ppl.yaml \
  --output /tmp/mohawk_tiny_eval_ppl_teacher_ckpt

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py --config configs/smoke/tiny_cpu_eval_ppl.yaml
```

This validates only the tiny CPU `EvalConfig -> eval_ppl` callback and
`save_latest` checkpoint path. The local JSON training loader used by this
smoke does not implement resumable dataloader state, so the latest checkpoint
skips `dataloader_state_dict.pth`. Representative data, CUDA, DDP/FSDP, and
resumable production dataloader checkpoints remain separate gates.

Tiny CPU bfloat16 wrapper-driven `eval_ppl` callback smoke using the same
local teacher checkpoint:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_bfloat16_eval_ppl.yaml
```

This validates only the tiny CPU path where bfloat16 student and teacher
wrappers run the `EvalConfig -> eval_ppl` callback and `save_latest`
checkpoint path. The centralized wrapper logs the bfloat16 AMP fallback, writes
`/tmp/mohawk_tiny_bfloat16_eval_ppl_cpu_run/save/latest_Perplexity`, and skips
`dataloader_state_dict.pth` because the local JSON training loader is not
resumable. It does not validate CUDA bfloat16, distributed wrappers,
representative eval data, resumable production dataloader checkpoints, or
full-size checkpoints.

Tiny CUDA/DDP/FSDP bfloat16 wrapper-driven `eval_ppl` callback smokes:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:08:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  python run.py --config configs/smoke/tiny_cuda_bfloat16_eval_ppl.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:08:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  torchrun --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_ddp_bfloat16_eval_ppl.yaml

srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:10:00 --immediate=1 env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  torchrun --standalone --nproc_per_node=2 run.py \
  --config configs/smoke/tiny_cuda_fsdp_bfloat16_eval_ppl.yaml
```

These validate only tiny Llama-style CUDA bfloat16 wrapper callback paths on one
Slurm node. The single-process smoke writes
`/home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_bfloat16_eval_ppl/save/latest_Perplexity`
with `eval_score: 49020.808594`; DDP and FSDP write the corresponding
`tiny_cuda_ddp_bfloat16_eval_ppl` and `tiny_cuda_fsdp_bfloat16_eval_ppl`
artifacts with `eval_score: 55746.601562`. Artifact verification recorded
final/latest tensors as `torch.bfloat16`, single-process hash
`52fe58e628de69a168b351d41c8d01a52017300497a0675770ae75653ccf1ef1`, and
DDP/FSDP hash `0bc9e79b22d4fce8a926de7260a144c0a5a0ff09a86328af5f35777ab655af15`.
These smokes do not validate representative eval data, multi-node DDP/FSDP
eval, SSM/Mamba, or full-size checkpoints.

Tiny CPU frequency-triggered eval_ppl callback using the same local teacher
checkpoint:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_eval_ppl_frequency.yaml
```

This validates only the tiny CPU periodic evaluator path with `frequency: 1`,
`eval_at_start: false`, and `eval_at_end: false`. It writes
`latest_Perplexity` from the frequency branch of the real training loop.

Tiny CPU `save_best` checkpoint smoke using the same eval_ppl callback after
creating its local teacher checkpoint:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_eval_ppl_save_best.yaml
```

This validates only the tiny CPU `EvalConfig -> eval_ppl` callback and
`save_best` checkpoint path. It writes `best_Perplexity` on the first real
evaluation and, like the `save_latest` smoke above, skips
`dataloader_state_dict.pth` because the local JSON training loader is not
resumable.

Tiny CPU bfloat16 `save_best` checkpoint smoke using the same eval_ppl callback
and local teacher checkpoint:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_bfloat16_eval_ppl_save_best.yaml
```

This validates only the tiny CPU path where bfloat16 student and teacher
wrappers run the `EvalConfig -> eval_ppl` callback and `save_best` checkpoint
path. It writes
`/tmp/mohawk_tiny_bfloat16_eval_ppl_save_best_cpu_run/save/best_Perplexity`,
logs the bfloat16 AMP fallback, and skips `dataloader_state_dict.pth` because
the local JSON training loader is not resumable. It does not validate CUDA
bfloat16, distributed wrappers, representative eval data, resumable production
dataloader checkpoints, or full-size checkpoints.

Tiny CPU multi-evaluator callback smoke using the same local teacher checkpoint
and cached public `wikitext` task data:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_eval_multi.yaml
```

This validates only the tiny CPU path where one `run.py` training execution has
multiple `EvalConfig` entries. The smoke runs `eval_ppl`, saves
`latest_Perplexity`, then runs the benchmark callback on cached `wikitext` with
the same live Mohawk model. It does not validate representative benchmarks,
uncached task downloads, CUDA, DDP/FSDP, or full-size checkpoints.

Tiny CPU public HFDataset `save_latest` state smoke using the populated caches
from the public HFDataset training smoke:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_eval_ppl_hfdata_save_latest.yaml
```

This validates that a tiny public `HFDataset` training loader writes
`dataloader_state_dict.pth` into the evaluator latest checkpoint. It remains
scoped to public `NeelNanda/pile-10k` cached data and does not validate
representative datasets, distributed sharding, CUDA, DDP/FSDP, or full-size
checkpoints.

Tiny CPU bfloat16 public HFDataset `save_latest` state smoke using the same
populated caches:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_bfloat16_eval_ppl_hfdata_save_latest.yaml
```

This validates that the tiny public `HFDataset` training loader writes
`dataloader_state_dict.pth` into the evaluator latest checkpoint while the
student and teacher wrappers use `model_dtype: bfloat16` and the bfloat16 AMP
fallback path. It writes
`/tmp/mohawk_tiny_bfloat16_eval_ppl_hfdata_cpu_run/save/latest_Perplexity`.
It remains scoped to cached public `NeelNanda/pile-10k` data and does not
validate CUDA bfloat16, representative datasets, large or gated datasets,
distributed sharding, DDP/FSDP, or full-size checkpoints.

Tiny CPU bfloat16 public HFDataset `save_best` state smoke using the same
populated caches:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_bfloat16_eval_ppl_hfdata_save_best.yaml
```

This validates that the tiny public `HFDataset` training loader writes
`dataloader_state_dict.pth` into the evaluator best checkpoint while the
student and teacher wrappers use `model_dtype: bfloat16` and the bfloat16 AMP
fallback path. It writes
`/tmp/mohawk_tiny_bfloat16_eval_ppl_hfdata_save_best_cpu_run/save/best_Perplexity`.
It remains scoped to cached public `NeelNanda/pile-10k` data and does not
validate CUDA bfloat16, representative datasets, large or gated datasets,
distributed sharding, DDP/FSDP, or full-size checkpoints.

Tiny CPU public HFDataset `save_best` state smoke using the same populated
caches:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_eval_ppl_hfdata_save_best.yaml
```

This validates that a tiny public `HFDataset` training loader writes
`dataloader_state_dict.pth` into the evaluator best checkpoint. It remains
scoped to public `NeelNanda/pile-10k` cached data and does not validate
representative datasets, distributed sharding, CUDA, DDP/FSDP, or full-size
checkpoints.

### lm-eval-harness Benchmarks

```bash
python evals/benchmark.py --dir <checkpoint_or_hf_model_dir> --tasks mmlu
```

This is a usage template, not a representative public checkpoint run. The
current public audit validates the row-scoped tiny Hugging Face and tiny Mohawk
checkpoint smokes below; representative Mohawk checkpoint benchmarking still
requires a real checkpoint, tokenizer, compatible runtime dependencies, task
data, and suitable GPU resources.
The default `--backend auto` can route loadable Hugging Face models through the
row-scoped tiny public HF path validated by this audit; use `--backend mohawk`
only for a real Mohawk checkpoint run with the resources listed above.

`--tasks` is a comma-separated list, for example:
`arc_challenge,arc_easy,piqa,winogrande,hellaswag,mmlu`.

Tiny public Hugging Face smoke with network access for task data:

```bash
python evals/benchmark.py --dir sshleifer/tiny-gpt2 \
  --tasks wikitext --batch_size 1 --limit 1
```

After both the tiny model and lm-eval task dataset are already cached, the same
command can be rerun with `--local_files_only`. A first local-only run fails if
the task dataset cache is absent. Full Mohawk checkpoint benchmarking
additionally needs the Mohawk runtime dependencies, checkpoint, tokenizer, and
suitable GPU resources.

Tiny CPU Mohawk checkpoint smoke after running the training smoke above:

```bash
HF_DATASETS_CACHE=/tmp/mohawk_lm_eval_datasets \
  python evals/benchmark.py --backend mohawk \
  --dir /tmp/mohawk_tiny_cpu_run/save \
  --tasks wikitext --batch_size 1 --limit 1 --device cpu
```

After `wikitext` task data is cached in that writable `/tmp` cache, the same
command can be rerun with `--local_files_only` plus `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. This only validates the tiny CPU Mohawk checkpoint
path; it does not validate representative checkpoints, CUDA, DDP/FSDP, or gated
models/data.

Tiny CPU bfloat16 Mohawk checkpoint lm-eval after running the bfloat16 fallback
smoke above and after `wikitext` task data is cached:

```bash
HF_DATASETS_CACHE=/tmp/mohawk_lm_eval_datasets \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python evals/benchmark.py --backend mohawk \
  --dir /tmp/mohawk_tiny_bfloat16_fallback_cpu_run/save \
  --tasks wikitext --batch_size 1 --limit 1 --local_files_only --device cpu
```

The benchmark CLI infers the saved `TrainConfig.model_dtype` for local Mohawk
checkpoints and prints the effective loaded dtype. This validates only the tiny
CPU bfloat16 checkpoint with cached public task data; it does not validate CUDA
bfloat16, representative checkpoints, uncached task downloads, distributed
evaluation, or full-size data.

Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint lm-eval after running
the DoubleBlock reference training smokes and after `wikitext` task data is
cached:

```bash
HF_DATASETS_CACHE=/tmp/mohawk_lm_eval_datasets \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python evals/benchmark.py --backend mohawk \
  --dir /tmp/mohawk_tiny_doubleblock_discrete_mamba2_ref_cpu_run/save \
  --tasks wikitext --batch_size 1 --limit 1 --local_files_only --device cpu
```

Equivalent commands also ran for
`/tmp/mohawk_tiny_doubleblock_vanilla_discrete_mamba2_ref_cpu_run/save`,
`/tmp/mohawk_tiny_doubleblock_hymba_discrete_mamba2_ref_cpu_run/save`, and
`/tmp/mohawk_tiny_doubleblock_merger_discrete_mamba2_ref_cpu_run/save`.
Adapter, Vanilla, and Hymba printed `wikitext 206463.186451` plus
`AVG 206463.186451`; Merger printed `wikitext 218913.474906` plus
`AVG 218913.474906`. All four loaded as `torch.float32` through the Mohawk
backend. This validates only tiny CPU DoubleBlock-family checkpoints using the
pure-PyTorch DiscreteMamba2 reference path and cached public `wikitext`; it does
not validate fast Mamba kernels, CUDA lm-eval, distributed evaluation,
production Qwen/Llama hybrid checkpoints, representative checkpoints, uncached
task downloads, or full-size data.

Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval uses the
shared checkpoint and dataset-cache paths because Slurm compute nodes cannot see
login-host `/tmp`. The adapter path can be checked with:

```bash
srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4 \
  --time=00:08:00 --immediate=120 env \
  HF_DATASETS_CACHE=/home/abick/mohawk/artifacts/gpu_smoke/lm_eval_datasets \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python evals/benchmark.py --backend mohawk \
  --dir /home/abick/mohawk/artifacts/gpu_smoke/doubleblock_ref_checkpoints/adapter \
  --tasks wikitext --batch_size 1 --limit 1 --local_files_only --device cuda
```

Equivalent commands also ran for
`/home/abick/mohawk/artifacts/gpu_smoke/doubleblock_ref_checkpoints/vanilla`,
`/home/abick/mohawk/artifacts/gpu_smoke/doubleblock_ref_checkpoints/hymba`,
and
`/home/abick/mohawk/artifacts/gpu_smoke/doubleblock_ref_checkpoints/merger`.
All four loaded on `cuda:0` through the Mohawk centralized inference wrapper,
printed `Model dtype: torch.float32`, and used cached public `wikitext`.
Adapter, Vanilla, and Hymba printed `wikitext 206463.186451` plus
`AVG 206463.186451`; Merger printed `wikitext 218913.474906` plus
`AVG 218913.474906`. This validates only tiny single-process CUDA
DoubleBlock-family checkpoint lm-eval through the pure-PyTorch reference path;
it does not validate fast Mamba kernels, distributed evaluation, production
Qwen/Llama hybrid checkpoints, representative checkpoints, uncached task
downloads, or full-size data.

Tiny CPU wrapper-driven benchmark callback after creating its local teacher
checkpoint:

```bash
python tools/create_tiny_smoke_checkpoint.py \
  --config configs/smoke/tiny_cpu_eval_benchmark.yaml \
  --output /tmp/mohawk_tiny_eval_benchmark_teacher_ckpt

MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_eval_benchmark.yaml
```

This validates only the tiny CPU `EvalConfig -> benchmark` callback with cached
`wikitext` task data in `/tmp/mohawk_lm_eval_datasets`. It does not validate
representative benchmarks, uncached task downloads, CUDA, DDP/FSDP, or full-size
checkpoints.

Tiny CPU bfloat16 wrapper-driven benchmark callback using the same local teacher
checkpoint and cached public `wikitext` task data:

```bash
MOHAWK_ALLOW_CPU_TRAINING=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled python run.py \
  --config configs/smoke/tiny_cpu_bfloat16_eval_benchmark.yaml
```

This validates only the tiny CPU `EvalConfig -> benchmark` callback while the
student and teacher wrappers use `model_dtype: bfloat16` and the bfloat16 AMP
fallback path. It does not validate CUDA bfloat16, representative benchmarks,
uncached task downloads, DDP/FSDP, or full-size checkpoints.

The tiny public benchmark smoke used `lm_eval 0.4.11`, matching the pinned
default `requirements.txt` dependency that passed in a fresh Python 3.12 venv.
That default dependency proof still does not validate representative benchmark
runs, uncached task downloads, CUDA execution, distributed evaluation, or the
optional CUDA/SSM dependency layer tracked in
[PUBLIC_SUPPORT_MATRIX.md](PUBLIC_SUPPORT_MATRIX.md).

## Utility Scripts

- `tools/hybrid_weights_transfer.py`
  Copies selected attention heads from a teacher to a hybrid student. A tiny CPU
  public-teacher smoke runs:
  `WANDB_MODE=disabled python tools/hybrid_weights_transfer.py --config
  configs/smoke/tiny_cpu_hybrid_transfer.yaml --heads 0:0 --device cpu
  --allow-unexpected-student-load`. The first run needs network access to
  populate the isolated `/tmp` cache for
  `hf-internal-testing/tiny-random-LlamaForCausalLM`; after that, rerun with
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. This only validates the tiny
  public-teacher attention-transfer path.

  A production Qwen2.5-1.5B ARCH40 transfer has also run on a GB300 with the
  optional Mamba 2.3 environment and the public bfloat16 teacher checkpoint:
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled python
  tools/hybrid_weights_transfer.py --config
  configs/smoke/production_qwen2_1_5b_hybrid_arch40_transfer.yaml --device
  cuda`. The run loaded all 339 teacher tensors, copied the selected 40 heads,
  and saved a 1,799,668,560-parameter bfloat16 hybrid checkpoint. The paired
  `production_qwen2_1_5b_hybrid_arch40_supervised.yaml` config then completed
  one verified optimizer update over a bounded local 128-token batch. This is
  production architecture/checkpoint-path evidence, not a quality or sustained
  training result; gated Llama weights/tokenizers and distributed production
  training remain open.

  The independent
  `configs/smoke/production_llama3_2_3b_hybrid_arch20_probe.yaml` config keeps
  that gated-asset boundary explicit while testing the production Llama hybrid
  architecture. Slurm job `772052` constructed 4,066,033,536 random bfloat16
  parameters with 20 retained eager-attention heads, completed fast-Mamba
  forward/backward with 354 finite gradient tensors, and peaked at `19.493GB`.
  This is architecture integration evidence only; it does not substitute for
  the unavailable gated Llama checkpoint or tokenizer.

- `tools/benchmark_throughput.py`
  CUDA-graph throughput microbenchmark. Timing uses synchronized CUDA events;
  earlier `time.time()` results measured asynchronous graph submission and are
  not valid throughput evidence. A tiny public CUDA smoke works on a Slurm GPU
  allocation after the public model is cached:
  `srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4
  --time=00:05:00 --immediate=1 env HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1 python tools/benchmark_throughput.py --model hf
  --hf-model-name hf-internal-testing/tiny-random-LlamaForCausalLM
  --local_files_only --batch-sizes 1 --prompt-len 128 --max-gen-len 128`. This
  loads the cached tiny Llama-family model on CUDA and runs the same CUDA graph
  benchmark path with Hugging Face `StaticCache`.

  Production-size architecture benchmarks also run from cached public configs:

  ```bash
  python tools/benchmark_throughput.py --model codestral --local_files_only \
    --batch-sizes 1,2,4 --prompt-len 128 --max-gen-len 128
  python tools/benchmark_throughput.py --model falcon --local_files_only \
    --batch-sizes 1,2,4 --prompt-len 128 --max-gen-len 128
  PYTHONPATH=/path/to/edge/cartesia-pytorch \
  python tools/benchmark_throughput.py --model llamba --llamba-config-only \
    --local_files_only --batch-sizes 1,2,4 --prompt-len 128 --max-gen-len 128
  ```

  On a GB300 with the Mamba 2.3 runtime, the final 128-token rates for batches
  `1,2,4` were Codestral `277.43,531.49,938.83`, Falcon
  `255.19,475.21,952.45`, and Llamba `203.95,354.15,688.33` tokens/s. The
  Codestral and Falcon branches construct production-config-shaped random
  weights and load their production tokenizers. `--llamba-config-only` uses
  Cartesia's public source plus the Llamba-8B production config, constructs
  random weights, and deliberately skips pretrained weights and the gated
  Llama tokenizer. These are single-GPU architecture-throughput results, not
  checkpoint-quality, pretrained-weight, multi-GPU, or gated Llama evidence.

  Long-context generation prefill computes only the final prompt logit while
  retaining the full recurrent state, and reports synchronized prefill timing
  separately from decode. On the same GB300, Codestral processed a `32768`-
  token prompt in `10.724348s` (`3055.48` tokens/s) at `14.54GB`; Falcon-Mamba
  processed it in `3.520817s` (`9306.93` tokens/s) at `14.07GB`. Their token-128
  decode rates were `277.98` and `253.76` tokens/s. Llamba-8B's first 8192-token
  prefill included shape-specific compilation and took `718.051062s`; an
  identical warm-cache rerun took `7.905415s` (`1036.25` tokens/s), peaked at
  `20.95GB`, and decoded at `201.47` tokens/s. Llamba 32K prefill did not finish
  within either a 30-minute cold allocation or a 20-minute retry, so 32K
  Llamba throughput remains unproven for this source-overlay implementation.

- `tools/visualize_attention.py`
  Produces attention heatmaps for selected heads on a fixed example. A tiny
  public CPU smoke works with cached `sshleifer/tiny-gpt2` and writes a PNG,
  falling back to PIL when `matplotlib` is unavailable:
  `python tools/visualize_attention.py --model_name sshleifer/tiny-gpt2
  --local_files_only --device cpu --output_dir /tmp/mohawk_viz`. A broader
  public Llama-family CPU smoke also works after populating the cache:
  `python tools/visualize_attention.py --model_name
  hf-internal-testing/tiny-random-LlamaForCausalLM --device cpu --output_dir
  /tmp/mohawk_viz_llama`, followed by an offline rerun with
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and `--local_files_only`. This
  wrote a `256 x 622` RGB PNG with SHA-256
  `593bb14787d14db3ca6f631649aaf8cf3858cd9a455a8115e7253210a1328806`.
  A tiny public CUDA visualization also works on Slurm with storage-backed
  output:
  `srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4
  --time=00:04:00 --immediate=120 env HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1 python tools/visualize_attention.py --model_name
  hf-internal-testing/tiny-random-LlamaForCausalLM --output_dir
  /home/abick/storage/mohawk/visualize_attention_cuda --local_files_only
  --device cuda`. That run wrote a `256 x 622` RGB PNG with SHA-256
  `db3c010fae59d7418ba0cd40ed91eb5c5a483e58d00d3682e991c46e5edd18ef`.
  A representative default-model CUDA probe with
  `meta-llama/Llama-3.2-3B-Instruct` reached model loading but failed because
  the gated model files were not in the local cache while outgoing Hugging Face
  traffic was disabled. The representative Raven hidden-state visualization
  above covers a recurrent model that does not expose attention matrices;
  larger or gated attention-head visualizations still need a compatible cached
  model plus the relevant runtime resources.

- `generation/generate.py`
  Inference/sampling script with timing output. The tiny HF smoke runs in the
  ML runtime with cached or network-accessible model/tokenizer files. After
  running the tiny CPU training smoke above, the tiny Mohawk checkpoint
  generation path can be checked with:
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python generation/generate.py
  --model /tmp/mohawk_tiny_cpu_run/save --local_files_only --model_dtype auto
  --prompt hello --genlen 2 --repeats 1 --top_k 1`. A tiny bfloat16 checkpoint
  generation path can be checked with the same command using
  `--model /tmp/mohawk_tiny_bfloat16_fallback_cpu_run/save`; `--model_dtype
  auto` preserves the checkpoint's saved `TrainConfig.model_dtype` when the
  local config records one. A tiny CUDA bfloat16 checkpoint generation smoke
  also runs on a Slurm GPU allocation:
  `srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4
  --time=00:04:00 --immediate=1 env HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled python generation/generate.py
  --model /home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_bfloat16_supervised/save
  --local_files_only --model_dtype auto --prompt hello --genlen 2 --repeats 1
  --top_k 1`, which prints `Model dtype: torch.bfloat16` and generates
  `hello EverestMist`. These commands validate tiny Mohawk checkpoint paths;
  the representative Raven command above separately validates one custom
  Transformers checkpoint.

  The trained production Qwen hybrid can be loaded offline with:
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python generation/generate.py
  --model
  /home/abick/storage/mohawk/artifacts/production_qwen2_1_5b_arch40_supervised/save
  --local_files_only --model_dtype auto --prompt 'Explain state space models in
  one phrase:' --genlen 4 --repeats 1 --top_k 1`. Slurm job `771456` reported
  all 1,799,668,560 bfloat16 parameters and completed prompt processing plus
  decoding in `203ms`. This checkpoint received only one bounded optimizer
  update; the generated text is execution evidence, not a quality claim.

  Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoints can also be checked
  with the same local command shape after running the DoubleBlock reference
  training smokes above. For example:
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled python
  generation/generate.py --model
  /tmp/mohawk_tiny_doubleblock_discrete_mamba2_ref_cpu_run/save
  --local_files_only --model_dtype auto --prompt hello --genlen 2 --repeats 1
  --top_k 1`. The first adapter attempt failed during cached decoding with
  `DiscreteMamba2 autoregressive step requires mamba_ssm selective_state_update`;
  after wiring the `use_ref_impl: true` path to the pure-PyTorch reference
  autoregressive recurrence, adapter, vanilla, hymba, and merger checkpoints
  generated `hello anguish played`, `hello Bian Jian`, `hello anguish played`,
  and `hello anguish played`. This remains scoped to tiny CPU reference
  checkpoints and does not validate fast Mamba kernels, CUDA generation,
  distributed inference, production Qwen/Llama hybrid checkpoints, or
  representative checkpoints.

  Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation uses
  shared artifact paths because Slurm compute nodes cannot see login-host
  `/tmp`. The adapter path can be checked with:
  `srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4
  --time=00:05:00 --immediate=1 env HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled python generation/generate.py
  --model /home/abick/mohawk/artifacts/gpu_smoke/doubleblock_ref_checkpoints/adapter
  --local_files_only --model_dtype auto --prompt hello --genlen 2 --repeats 1
  --top_k 1`. Equivalent runs for
  `/home/abick/mohawk/artifacts/gpu_smoke/doubleblock_ref_checkpoints/vanilla`,
  `/home/abick/mohawk/artifacts/gpu_smoke/doubleblock_ref_checkpoints/hymba`,
  and
  `/home/abick/mohawk/artifacts/gpu_smoke/doubleblock_ref_checkpoints/merger`
  reported `Model dtype: torch.float32`, parameter counts `1612596`,
  `1612324`, `1612660`, and `1612628`, generated `hello anguish played`,
  `hello Bian Jian`, `hello anguish played`, and `hello anguish played`, and
  printed prompt processing plus decoding times `8ms`, `6ms`, `7ms`, and
  `7ms`. This remains scoped to tiny single-process CUDA DoubleBlock-family
  checkpoint generation through the pure-PyTorch reference autoregressive path;
  it does not validate fast Mamba kernels, `mamba_ssm`, `causal_conv1d`,
  distributed inference, production Qwen/Llama hybrid checkpoints, or
  representative checkpoints.

  Tiny CUDA DoubleBlock DiscreteMamba2 fast checkpoint generation uses the
  storage-backed fast training saves. The final current-code Slurm probe ran:
  `srun -p batch -A nemotron_arch_dev --qos=interactive --gpus-per-node=4
  --time=00:18:00 --immediate=120 env HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled python generation/generate.py
  --model /home/abick/storage/mohawk/artifacts/gpu_smoke/tiny_cuda_doubleblock_discrete_mamba2_fast/save
  --local_files_only --model_dtype auto --prompt hello --genlen 2 --repeats 1
  --top_k 1`, with equivalent commands for
  `tiny_cuda_doubleblock_vanilla_discrete_mamba2_fast/save`,
  `tiny_cuda_doubleblock_hymba_discrete_mamba2_fast/save`, and
  `tiny_cuda_doubleblock_merger_discrete_mamba2_fast/save`. Adapter, Vanilla,
  Hymba, and Merger reported `Model dtype: torch.float32`, parameter counts
  `1612596`, `1612324`, `1612660`, and `1612628`, generated
  `hello anguish played`, `hello Bian Jian`, `hello anguish played`, and
  `hello anguish played`, and printed prompt processing plus decoding times
  `7ms`, `6ms`, `7ms`, and `7ms`. The first Vanilla fast-generation probes
  exposed and fixed DoubleBlock nested-cache handling, an over-broad CUDA graph
  heuristic for hybrid attention checkpoints, and a `causal_conv1d` one-token
  stride-layout fallback. Slurm job `748021` also loaded the DDP checkpoint
  produced by the two-node fast-Mamba run and generated `hello anguish played`
  in `7ms`. That probe exposed a distributed-checkpoint metadata edge case:
  DDP wrote a component-only `config.json` and the tokenizer setting in the
  full `config.yaml`. Tokenizer discovery now examines each available local
  config instead of stopping at the first parseable file. This remains scoped
  to tiny single-process CUDA generation from fast DoubleBlock-family
  checkpoints; it does not validate distributed inference, production
  Qwen/Llama hybrid checkpoints, gated model/tokenizer access, representative
  checkpoints, or long-context generation.

## Repository Layout

```text
mohawk/
├── components/          # Blocks, mixers, LM heads
├── configs/             # Train/eval architecture recipes
├── dataloaders/         # Dataset generators and wrappers
├── distill/             # Run orchestration and objective steps
├── evals/               # Evaluation entrypoints and adapters
├── external_models/     # External model definitions integrated here
├── generation/          # Text generation utilities
├── training_wrapper/    # DDP/FSDP/centralized wrappers
├── utils/               # Config, logging, init, distributed helpers
└── run.py               # Main training entrypoint
```

## Publications

This codebase was used in the following research publications:

### Retrieval-Aware Distillation for Transformer-SSM Hybrids
```bibtex
@misc{bick2026retrieval,
      title={Retrieval-Aware Distillation for Transformer-SSM Hybrids}, 
      author={Aviv Bick and Eric P. Xing and Albert Gu},
      year={2026},
      eprint={2602.11374},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.11374}, 
}
```

### Llamba: Scaling Distilled Recurrent Models for Efficient Language Processing
```bibtex
@article{bick2025llamba,
  title={Llamba: Scaling distilled recurrent models for efficient language processing},
  author={Bick, Aviv and Katsch, Tobias and Sohoni, Nimit and Desai, Arjun and Gu, Albert},
  journal={arXiv preprint arXiv:2502.14458},
  year={2025}
}
```

### Thinking Slow, Fast: Scaling Inference Compute with Distilled Reasoners
```bibtex
@misc{paliotta2025thinking,
      title={Thinking Slow, Fast: Scaling Inference Compute with Distilled Reasoners}, 
      author={Daniele Paliotta and Junxiong Wang and Matteo Pagliardini and Kevin Y. Li and Aviv Bick and J. Zico Kolter and Albert Gu and François Fleuret and Tri Dao},
      year={2025},
      eprint={2502.20339},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.20339}, 
}
```

### Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models (Mohawk)
```bibtex
@misc{mohawk,
      title={Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models}, 
      author={Aviv Bick and Kevin Y. Li and Eric P. Xing and J. Zico Kolter and Albert Gu},
      year={2025},
      eprint={2408.10189},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.10189}, 
}
```

## Citation

If this repository is useful in your work, cite:

```bibtex
@software{mohawk,
  title = {Knowledge Distillation for Hybrid Transformer-SSM Models},
  author = {Aviv Bick},
  year = {2024},
  url = {https://github.com/goombalab/mohawk}
}
```

## License

MIT. See [LICENSE](LICENSE).

## Contributing

Contribution workflow and expectations are documented in [CONTRIBUTING.md](CONTRIBUTING.md).
