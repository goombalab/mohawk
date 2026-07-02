# Knowledge Distillation for Hybrid Transformer-SSM Models

Mohawk is a PyTorch research framework for distilling language models into
hybrid architectures that combine Transformer attention with state-space
models such as Mamba.

## Features

- Llama, Qwen2, Falcon, and Phi-style hybrid model components.
- YAML configuration with `LOAD`-based composition.
- Supervised, hidden-state, matrix, and DPO distillation objectives.
- Centralized, DDP, and FSDP training wrappers.
- Perplexity, lm-eval-harness, generation, and analysis utilities.

This repository is run directly from source; it is not a packaged library.

## Setup

```bash
git clone https://github.com/goombalab/mohawk.git
cd mohawk
python -m pip install -r requirements.txt
```

The default requirements cover the Python training and evaluation runtime.
CUDA-built SSM kernels are optional:

```bash
python -m pip install --no-build-isolation -r requirements-ssm-cuda.txt
```

Those kernels require a compatible CUDA toolkit and compiler. Flash Attention
is also optional:

```bash
python -m pip install flash-attn --no-build-isolation
```

For a dependency-light checkout validation:

```bash
python -m pip install pytest PyYAML
python run.py --help
python evals/benchmark.py --help
python generation/generate.py --help
python3 -m compileall -q .
pytest -q
```

Use environment variables for credentials and runtime selection:

- `HF_TOKEN` for private or gated Hugging Face assets.
- `WANDB_API_KEY` for online experiment tracking.
- `WANDB_MODE=disabled` to disable W&B logging.
- `CUDA_VISIBLE_DEVICES` to select GPUs.

## Training

Single process:

```bash
WANDB_MODE=disabled python run.py \
  --config configs/Qwen2/1.5B/hybrid/adapter.yaml
```

Distributed:

```bash
WANDB_MODE=disabled torchrun --standalone --nproc_per_node=8 run.py \
  --config configs/Qwen2/1.5B/hybrid/adapter.yaml
```

Research configs may contain placeholder checkpoint or dataset paths. Review
the selected config and its `LOAD` chain before launching a run.

The main configuration sections are:

- `ComponentsConfig`: model architecture and layer types.
- `TrainConfig`: optimizer, precision, schedule, and training length.
- `DistillConfig`: objective and run metadata.
- `TeacherConfig`: teacher model and tokenizer.
- `TrainDataConfig`: dataset and loader pipeline.
- `LoadConfig`: checkpoint initialization rules.
- `ManagementConfig`: paths, caches, and logging.

## Evaluation

Run lm-eval-harness tasks against a Mohawk checkpoint or Hugging Face model:

```bash
python evals/benchmark.py \
  --dir <checkpoint-or-model> \
  --tasks arc_easy,piqa \
  --batch_size 1
```

Run a perplexity smoke through the repository's evaluator:

```bash
python evals/eval_ppl.py \
  --model <checkpoint-or-model> \
  --backend auto \
  --n_batches 1
```

Both commands support local-cache-only operation with `--local_files_only`.
Custom Transformers implementations can be registered with repeated
`--model-registration-module <module>` options.

## Generation

```bash
python generation/generate.py \
  --model <checkpoint-or-model> \
  --prompt "The future of language models is" \
  --genlen 32
```

## Utilities

- `tools/hybrid_weights_transfer.py`: transfer selected teacher weights into a
  hybrid student.
- `tools/benchmark_throughput.py`: measure model prefill and decode throughput.
- `tools/visualize_attention.py`: render attention heatmaps.

## Repository Layout

```text
components/          Model blocks, mixers, and heads
configs/             Architecture and training recipes
dataloaders/         Dataset and batching pipelines
distill/             Distillation objectives and orchestration
evals/               Evaluation entry points and adapters
external_models/     Integrated external model definitions
generation/          Text generation tools
training_wrapper/    Centralized, DDP, and FSDP wrappers
utils/               Configuration, initialization, and runtime helpers
run.py               Training entry point
```

## Publications

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

```bibtex
@software{mohawk,
  title = {Knowledge Distillation for Hybrid Transformer-SSM Models},
  author = {Aviv Bick},
  year = {2024},
  url = {https://github.com/goombalab/mohawk}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
