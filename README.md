# Knowledge Distillation for Hybrid Transformer-SSM Models

A PyTorch framework for knowledge distillation from large language models (LLMs) to hybrid architectures that combine Transformer attention mechanisms with State-Space Models (SSM), such as Mamba.

## What This Repository Contains

- Model building blocks for Llama, Qwen2, Falcon, and Phi-style hybrids.
- A composable YAML config system (`LOAD`-based inheritance).
- Distillation objectives: `supervised`, `hstates`, `matrices`, and `dpo`.
- Distributed training wrappers (DDP/FSDP/centralized).
- Evaluation utilities for perplexity and lm-eval-harness tasks.

This is not a packaged library. You run scripts directly from the repo.

## Setup

### Requirements

- Python 3.8+
- PyTorch 2.1+ with CUDA
- One or more NVIDIA GPUs for training/eval

### Install

```bash
git clone https://github.com/avivbick/mohawk.git
cd mohawk
pip install -r requirements.txt
```

Optional accelerators:

```bash
pip install flash-attn --no-build-isolation
```

### Environment Variables

Use environment variables for credentials instead of hardcoding:

- `HF_TOKEN` for private/gated Hugging Face models
- `WANDB_API_KEY` for experiment tracking
- `CUDA_VISIBLE_DEVICES` to pin GPUs

Global runtime defaults live in `configs/management.yaml`.

## Quick Start

### Single-GPU Training

```bash
python run.py --config configs/Qwen2/1.5B/hybrid/adapter.yaml
```

### Multi-GPU Training

```bash
torchrun --standalone --nproc_per_node=8 run.py \
  --config configs/Qwen2/1.5B/hybrid/adapter.yaml
```

`--config` also accepts a comma-separated list; configs are loaded and run sequentially.

## How Configuration Works

Every run is driven by YAML. The important top-level sections are:

- `ComponentsConfig`: architecture definition (block sequence and layer types)
- `TrainConfig`: optimization schedule and training length
- `DistillConfig`: objective selection and logging run name
- `TeacherConfig`: teacher checkpoint/path and tokenizer context
- `TrainDataConfig`: dataset source and loader strategy
- `LoadConfig`: initialization and checkpoint loading rules
- `ManagementConfig`: cache paths, W&B config, environment defaults

Useful starting points:

- `configs/Qwen2/1.5B/hybrid/adapter.yaml`
- `configs/Llama/1B/hybrid/mohawk_8.yaml`
- `configs/Llama/8B/bases/_supervised.yaml`

## Evaluation and Analysis

### Perplexity

Perplexity is integrated through training/eval wrappers and `evals/eval_ppl.py` implements the evaluator class used by those wrappers.

### lm-eval-harness Benchmarks

```bash
python evals/benchmark.py --dir <checkpoint_or_hf_model_dir> --tasks mmlu
```

`--tasks` is a comma-separated list, for example:
`arc_challenge,arc_easy,piqa,winogrande,hellaswag,mmlu`.

## Utility Scripts

- `tools/hybrid_weights_transfer.py`
  Copies selected attention heads from a teacher to a hybrid student. Uses `--config` and expects a supported `TeacherConfig.dir`.

- `tools/benchmark_throughput.py`
  CUDA-graph throughput microbenchmark. This script is research-oriented and currently contains model-specific assumptions and hardcoded defaults.

- `tools/visualize_attention.py`
  Produces attention heatmaps for manually selected heads on a fixed example. Useful for qualitative inspection, not automated evaluation.

- `generation/generate.py`
  Inference/sampling script with timing output.

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
  url = {https://github.com/avivbick/mohawk}
}
```

## License

MIT. See [LICENSE](LICENSE).

## Contributing

Contribution workflow and expectations are documented in [CONTRIBUTING.md](CONTRIBUTING.md).
