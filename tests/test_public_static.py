from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ALLOWED_STATUSES = {
    "PASSED_END_TO_END",
    "PASSED_REAL_SMOKE",
    "FAILED_FIXED",
    "FAILED_UNFIXED",
    "BLOCKED_RESOURCE",
    "DOWNGRADED_DOCS",
    "REMOVED_OR_UNSUPPORTED",
}

OLD_STATUS_WORDING = [
    "Fully executed and passed",
    "Smoke-tested only",
    "Not verified",
    "Skipped due to missing",
    "Broken and fixed",
    "Broken and not fixed",
]


def _split_row(line: str) -> list[str]:
    return [cell.strip().replace("`", "") for cell in line.strip().strip("|").split("|")]


def _matrix_rows(section_name: str) -> list[list[str]]:
    matrix = (ROOT / "PUBLIC_SUPPORT_MATRIX.md").read_text()
    section = matrix.split(f"## {section_name}", 1)[1]
    next_heading = section.find("\n## ")
    if next_heading != -1:
        section = section[:next_heading]
    rows = []
    for line in section.splitlines():
        if line.startswith("| ") and not line.startswith("| ---"):
            cells = _split_row(line)
            if cells[0].lower() in {"scenario", "probe", "command"}:
                continue
            rows.append(cells)
    return rows


def test_matrix_uses_only_allowed_real_validation_statuses():
    matrix = (ROOT / "PUBLIC_SUPPORT_MATRIX.md").read_text()

    for status in ALLOWED_STATUSES:
        assert status in matrix
    for old_status in OLD_STATUS_WORDING:
        assert old_status not in matrix
    assert "Tiny Mamba2/SSM fast kernels now have centralized" in matrix
    assert "representative distributed or SSM training" in matrix
    assert "full research dependency install" not in matrix
    assert "full-requirements install" not in matrix
    assert "Validate requirements-pinned lm-eval" not in matrix

    scenario_rows = _matrix_rows("Scenario Matrix")
    assert scenario_rows
    for row in scenario_rows:
        assert len(row) == 5
        scenario, public_claim, evidence, status, next_evidence = row
        assert scenario
        assert public_claim
        assert evidence
        assert next_evidence
        assert status in ALLOWED_STATUSES


def test_hybrid_attention_backend_and_doubleblock_factory_placement_contracts():
    for path in [
        ROOT / "components/cores/Qwen2Attention.py",
        ROOT / "components/cores/LlamaAttention.py",
    ]:
        source = path.read_text()
        assert 'attn_implementation != "flash_attention_2"' in source
        assert 'self.model_cfg["_attn_implementation"] = attn_implementation' in source

    doubleblock_source = (ROOT / "components/blocks/DoubleBlock.py").read_text()
    assert "def _to_factory_dtype_device" in doubleblock_source
    assert "self.mlp = _to_factory_dtype_device" in doubleblock_source
    assert "self.adapter = Adapter(d_model, config, factory_kwargs)" in doubleblock_source
    assert "factory_kwargs=factory_kwargs" in doubleblock_source


def test_phase_1_fresh_environment_evidence_is_recorded():
    matrix = (ROOT / "PUBLIC_SUPPORT_MATRIX.md").read_text()
    phase_rows = _matrix_rows("Phase 1 Fresh Environment Evidence")
    phase_statuses = {row[0]: row[3] for row in phase_rows}

    assert phase_statuses["System Python"] == "PASSED_REAL_SMOKE"
    assert phase_statuses["System pip"] == "PASSED_REAL_SMOKE"
    assert phase_statuses["Local NVIDIA driver"] == "BLOCKED_RESOURCE"
    assert phase_statuses["System Torch probe"] == "BLOCKED_RESOURCE"
    assert phase_statuses["Interactive Slurm GPU probe"] == "PASSED_REAL_SMOKE"
    assert phase_statuses["Interactive Slurm Torch CUDA probe"] == "PASSED_REAL_SMOKE"
    assert phase_statuses["Fresh default runtime venv creation"] == "PASSED_REAL_SMOKE"
    assert phase_statuses["Fresh CPU manifest venv creation"] == "PASSED_REAL_SMOKE"
    assert phase_statuses["Fresh CPU manifest install, sandbox"] == "BLOCKED_RESOURCE"
    assert phase_statuses["Fresh CPU manifest install, network allowed"] == "PASSED_REAL_SMOKE"
    assert phase_statuses["Fresh CPU manifest validation commands"] == "PASSED_REAL_SMOKE"
    assert phase_statuses["Fresh default requirements install, sandbox"] == "BLOCKED_RESOURCE"
    assert phase_statuses["Fresh default requirements install, network allowed"] == "PASSED_REAL_SMOKE"
    assert phase_statuses["Fresh default runtime validation commands"] == "PASSED_REAL_SMOKE"
    assert phase_statuses["Current fresh default runtime venv creation"] == "PASSED_REAL_SMOKE"
    assert phase_statuses["Current fresh default requirements install, sandbox"] == "BLOCKED_RESOURCE"
    assert phase_statuses["Current fresh default requirements install, network allowed"] == "PASSED_REAL_SMOKE"
    assert phase_statuses["Current fresh default runtime validation commands"] == "PASSED_REAL_SMOKE"
    assert phase_statuses["Fresh default venv Slurm CUDA probe, /tmp path"] == "BLOCKED_RESOURCE"
    assert phase_statuses["Shared fresh default venv install for Slurm probe"] == "BLOCKED_RESOURCE"
    assert phase_statuses["Optional CUDA/SSM dependency manifest"] == "FAILED_FIXED"

    assert "Python 3.12.12" in matrix
    assert "pip 25.3" in matrix
    assert "NVIDIA-SMI has failed" in matrix
    assert "ModuleNotFoundError: No module named 'torch'" in matrix
    assert "batch* up 4:00:00 1276 gpu:4(S:0-1)" in matrix
    assert "batch_long up 7-00:00:00 1276 gpu:4(S:0-1)" in matrix
    assert "NVIDIA GB300" in matrix
    assert "cuda available True" in matrix
    assert "device count `4`" in matrix
    assert "torch 2.12.1+cu130" in matrix
    assert "transformers 4.55.4" in matrix
    assert "lm_eval 0.4.11" in matrix
    assert "mamba_ssm_spec None" in matrix
    assert "causal_conv1d_spec None" in matrix
    assert "nvcc None" in matrix
    assert "nvidia-cuda-nvcc 13.0.88" in matrix
    assert "nvidia-cuda-crt 13.0.88" in matrix
    assert "nvidia-nvvm 13.0.88" in matrix
    assert "nvidia-cuda-cccl 13.0.85" in matrix
    assert "python storage ssm-cuda venv" in matrix
    assert "/home/abick/storage/mohawk/venvs/ssm-cuda-20260626/bin/python" in matrix
    assert "configs/smoke/tiny_cuda_default_runtime_supervised.yaml" in matrix
    assert "Tiny-CUDA-Default-Runtime-Supervised-Smoke" in matrix
    assert "configs/smoke/tiny_cuda_ddp_default_runtime_bfloat16.yaml" in matrix
    assert "configs/smoke/tiny_cuda_fsdp_default_runtime_bfloat16.yaml" in matrix
    assert "configs/smoke/tiny_cuda_ddp_default_runtime_eval_hstates.yaml" in matrix
    assert "configs/smoke/tiny_cuda_fsdp_default_runtime_eval_hstates.yaml" in matrix
    assert "configs/smoke/tiny_cuda_ddp_default_runtime_eval_ppl.yaml" in matrix
    assert "configs/smoke/tiny_cuda_fsdp_default_runtime_eval_ppl.yaml" in matrix
    assert "configs/smoke/tiny_cuda_ddp_default_runtime_eval_benchmark.yaml" in matrix
    assert "configs/smoke/tiny_cuda_fsdp_default_runtime_eval_benchmark.yaml" in matrix
    assert "Tiny-CUDA-DDP-Default-Runtime-BFloat16-Smoke" in matrix
    assert "Tiny-CUDA-FSDP-Default-Runtime-BFloat16-Smoke" in matrix
    assert "Tiny-CUDA-DDP-Default-Runtime-Eval-HStates-Smoke" in matrix
    assert "Tiny-CUDA-FSDP-Default-Runtime-Eval-HStates-Smoke" in matrix
    assert "BaseDataWrapper.close()" in matrix
    assert "BaseDataGenerator" in matrix
    assert "13 passed in 20.07s" in matrix
    assert "Slurm job `690381`" in matrix
    assert "optimizer_state_entries `12`" in matrix
    assert "287b32e35195cb73a9a0f35493802b98f510bff5269543d6ca721750a8a2bb46" in matrix
    assert "nvcc --list-gpu-code" in matrix
    assert "sm_100" in matrix
    assert "mohawk-phase1-default-venv-20260626" in matrix
    assert "103 passed, 1 skipped, 6 warnings in 234.48s" in matrix
    assert "execve()" in matrix
    assert "Disk quota exceeded" in matrix
    assert "requirements-ssm-cuda.txt" in matrix
    assert "requirements-cpu.txt" in matrix
    assert "19 passed, 8 skipped" in matrix
    assert "103 passed, 1 skipped" in matrix


def test_core_public_scenarios_are_conservatively_classified():
    rows = {row[0]: row for row in _matrix_rows("Scenario Matrix")}

    assert rows["Default runtime dependency install"][3] == "FAILED_FIXED"
    assert rows["Optional CUDA/SSM dependency install"][3] == "FAILED_FIXED"
    assert rows["Dependency-light CPU checkout"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py supervised training smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU run.py gradient accumulation smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA DDP/FSDP run.py mixed-precision gradient accumulation no_sync smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA DDP/FSDP run.py bfloat16 gradient accumulation no_sync smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py WSD scheduler smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py Adam optimizer smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py optimize_weights whitelist/blacklist smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py LoadConfig.model whitelist/blacklist smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py LoadConfig.model rename smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py LoadConfig.model sequential-list smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py LoadConfig.model strict failure guards"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py centralized mixed-precision smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py mixed-precision gradient accumulation smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py bfloat16 mixed-precision fallback smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CUDA/DDP/FSDP run.py bfloat16 supervised smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py bfloat16 gradient accumulation fallback smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py Qwen2/Phi/Falcon bfloat16 gradient accumulation fallback smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py Qwen2 bfloat16 mixed-precision fallback smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CUDA run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py compile_model smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU run.py bfloat16 compile_model smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py teacher compile_model smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py student+teacher compile_model smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py bfloat16 teacher compile_model smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py bfloat16 student+teacher compile_model smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py public Hugging Face teacher smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py hstates training smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU run.py bfloat16 hstates training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py sequential_hstates training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py bfloat16 sequential_hstates training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py matrices training smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU run.py bfloat16 matrices training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py DPO training smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU run.py bfloat16 DPO training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py supervised_instruct training smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU run.py supervised_instruct teacher-logits supervision smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py supervised_instruct generate-supervision smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py bfloat16 supervised_instruct training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU torchrun single-process training smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU run.py checkpoint resume smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU run.py mixed-precision checkpoint resume smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py bfloat16 checkpoint resume smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA run.py checkpoint resume smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA run.py mixed-precision checkpoint resume smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py public HFDataset dataloader resume smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py public HFDataset full-resume smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py bfloat16 public HFDataset full-resume smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py lazy-load checkpoint training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py mixed-precision lazy-load checkpoint smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py bfloat16 lazy-load checkpoint smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA run.py lazy-load checkpoint smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Lazy/eager initialization"][3] == "BLOCKED_RESOURCE"
    assert rows["Single-process CUDA training"][3] == "PASSED_REAL_SMOKE"
    assert rows["Single-process CUDA torchrun training"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU comma-separated run.py sequential training smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU comma-separated run.py train+eval callback chain smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA comma-separated run.py train+eval callback chain smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Representative comma-separated config execution"][3] == "BLOCKED_RESOURCE"
    assert rows["Llama 8B representative config loadability"][3] == "FAILED_FIXED"
    assert rows["Phi 1.5B representative config loadability"][3] == "FAILED_FIXED"
    assert rows["Additional Llama/Qwen2/Falcon representative config loadability"][3] == "FAILED_FIXED"
    assert rows["Generated hybrid architecture fragments"][3] == "REMOVED_OR_UNSUPPORTED"
    assert rows["DDP multi-GPU training"][3] == "FAILED_FIXED"
    assert rows["FSDP multi-GPU training"][3] == "PASSED_REAL_SMOKE"
    assert rows["Centralized CUDA mixed precision"][3] == "PASSED_REAL_SMOKE"
    assert rows["DDP/FSDP mixed precision"][3] == "FAILED_FIXED"
    assert rows["Tiny CUDA DDP checkpoint resume smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA FSDP checkpoint resume smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Real FSDP activation checkpointing"][3] == "PASSED_REAL_SMOKE"
    assert rows["Wrapper resume"][3] == "PASSED_REAL_SMOKE"
    assert rows["Mamba/SSM fast CUDA kernel components"][3] == "FAILED_FIXED"
    assert rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 fast training smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA multi-node DDP/FSDP run.py DoubleBlock DiscreteMamba2 fast training smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA DoubleBlock DiscreteMamba2 fast checkpoint generation"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU run.py DiscreteMamba2 reference training smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA DDP/FSDP run.py DoubleBlock DiscreteMamba2 reference training smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Public remote HFDataset + Tokenize smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py public HFDataset training smoke"][3] == "FAILED_FIXED"
    assert rows["Public remote streaming HFDataset + Tokenize smoke"][3] == "FAILED_FIXED"
    assert rows["Public C4 streaming HFDataset + Tokenize smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py public C4 streaming training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CUDA run.py FineWeb-Edu production-loader training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA DDP/FSDP FineWeb-Edu production-loader training smokes"][3] == "FAILED_FIXED"
    assert rows["Legacy local C4DataLoader"][3] == "REMOVED_OR_UNSUPPORTED"
    assert rows["Tiny CPU run.py RandomDataLoader training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py ShuffleLoader training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py conversation-collate training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py classic-collate training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py KV raw-packing training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py CopyingTaskDataset training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py SequentialLoader training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py AggregationDataLoader training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Large/gated remote training datasets"][3] == "BLOCKED_RESOURCE"
    assert rows["Tiny CPU run.py Qwen2 supervised training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py Phi supervised training smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU run.py Falcon supervised training smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU run.py DiscreteMamba2 reference training smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA DDP/FSDP run.py DoubleBlock DiscreteMamba2 reference training smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py DoubleBlock attention-only training smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py eval_hstates callback smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU run.py bfloat16 eval_hstates callback smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA run.py eval_hstates callback smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA DDP run.py eval_hstates callback smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA FSDP run.py eval_hstates callback smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_hstates callback smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py eval_ppl callback and latest-checkpoint smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CUDA run.py eval_ppl callback smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA DDP run.py eval_ppl callback smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CUDA FSDP run.py eval_ppl callback smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_ppl callback smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py bfloat16 eval_ppl callback smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py bfloat16 eval_ppl save_best checkpoint smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py eval_ppl frequency callback smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py eval_ppl save_best checkpoint smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU run.py multi-evaluator callback smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py eval_ppl public HFDataset latest-state smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset latest-state smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset best-state smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py eval_ppl public HFDataset best-state smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU Mohawk checkpoint lm-eval smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU bfloat16 Mohawk checkpoint lm-eval smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU run.py benchmark callback smoke"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU run.py bfloat16 benchmark callback smoke"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][3] == "PASSED_REAL_SMOKE"
    assert rows["Representative Mohawk checkpoint lm-eval"][3] == "BLOCKED_RESOURCE"
    assert rows["Public docs support claims"][3] == "FAILED_FIXED"
    assert "not proof that every surface is fully validated in a fresh public checkout" in rows["Public docs support claims"][2]
    assert "raw hybrid architecture_*.yaml files as include fragments" in rows["Public docs support claims"][2]
    assert "requires a local checkpoint directory with config.json" in rows["Public docs support claims"][2]
    assert "CONTRIBUTING now requires matrix-backed statuses" in rows["Public docs support claims"][2]
    assert "python3 -m pytest -q tests/test_public_static.py" in rows["Public docs support claims"][2]

    assert rows["Public tiny Hugging Face generation"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU Mohawk checkpoint generation"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU bfloat16 Mohawk checkpoint generation"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint generation"][3] == "FAILED_FIXED"
    assert rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA Mohawk checkpoint generation"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA bfloat16 Mohawk checkpoint generation"][3] == "PASSED_REAL_SMOKE"
    assert rows["Representative Mohawk checkpoint generation"][3] == "BLOCKED_RESOURCE"
    assert rows["Public tiny Hugging Face lm-eval"][3] == "PASSED_REAL_SMOKE"
    assert rows["Public tiny Hugging Face perplexity evaluator"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU Mohawk checkpoint perplexity evaluator"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU bfloat16 Mohawk checkpoint perplexity evaluator"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][3] == "FAILED_FIXED"
    assert rows["Tiny CUDA Mohawk checkpoint perplexity evaluator"][3] == "PASSED_REAL_SMOKE"
    assert rows["Tiny CUDA bfloat16 Mohawk checkpoint perplexity evaluator"][3] == "PASSED_REAL_SMOKE"
    assert rows["Representative Mohawk perplexity evaluator"][3] == "BLOCKED_RESOURCE"
    assert rows["Local JSON dataloader path"][3] == "PASSED_REAL_SMOKE"
    assert rows["tools/hybrid_weights_transfer.py execution"][3] == "FAILED_FIXED"
    assert rows["Tiny CPU tools/hybrid_weights_transfer.py public teacher smoke"][3] == "FAILED_FIXED"
    assert rows["Production Qwen2.5-1.5B ARCH40 hybrid transfer, optimizer update, and generation"][3] == "FAILED_FIXED"
    assert rows["Production Llama-3.2-3B ARCH20 random-weight hybrid forward/backward"][3] == "PASSED_REAL_SMOKE"
    assert rows["Public tiny tools/benchmark_throughput.py CUDA smoke"][3] == "FAILED_FIXED"
    assert rows["Production-size Codestral/Falcon/Llamba architecture throughput"][3] == "FAILED_FIXED"
    assert rows["Production-size long-context SSM architecture throughput"][3] == "FAILED_FIXED"
    assert rows["Full tools/benchmark_throughput.py research workflow"][3] == "BLOCKED_RESOURCE"
    assert rows["Public tiny tools/visualize_attention.py smoke"][3] == "FAILED_FIXED"
    assert rows["Full tools/visualize_attention.py research workflow"][3] == "BLOCKED_RESOURCE"
    assert "sshleifer/tiny-gpt2" in rows["Public tiny tools/visualize_attention.py smoke"][2]
    assert "hf-internal-testing/tiny-random-LlamaForCausalLM" in rows["Public tiny tools/visualize_attention.py smoke"][2]
    assert "256 x 622" in rows["Public tiny tools/visualize_attention.py smoke"][2]
    assert "593bb14787d14db3ca6f631649aaf8cf3858cd9a455a8115e7253210a1328806" in rows["Public tiny tools/visualize_attention.py smoke"][2]
    assert "/home/abick/storage/mohawk/visualize_attention_cuda" in rows["Public tiny tools/visualize_attention.py smoke"][2]
    assert "loaded the cached public tiny random Llama model on CUDA" in rows["Public tiny tools/visualize_attention.py smoke"][2]
    assert "db3c010fae59d7418ba0cd40ed91eb5c5a483e58d00d3682e991c46e5edd18ef" in rows["Public tiny tools/visualize_attention.py smoke"][2]
    assert "public tiny CPU/CUDA visualization" in rows["Public tiny tools/visualize_attention.py smoke"][4]
    assert "representative research visualizations" in rows["Public tiny tools/visualize_attention.py smoke"][4]
    representative_custom = rows[
        "Representative cached custom Transformers checkpoint tool smokes"
    ]
    representative_evidence = representative_custom[2].lower()
    assert representative_custom[3] == "FAILED_FIXED"
    assert "--model-registration-module MODULE" in representative_custom[2]
    assert "AvivBick/raven-nslots256-topk64" in representative_custom[2]
    assert "1697254264-byte model" in representative_custom[2]
    assert "424303808 parameters" in representative_custom[2]
    assert "job 750982" in representative_evidence
    assert "job 751419" in representative_evidence
    assert "job 751508" in representative_evidence
    assert "job 751635" in representative_evidence
    assert "job 752591" in representative_evidence
    assert "97.02066040039062" in representative_custom[2]
    assert "61.629803" in representative_custom[2]
    assert "hidden_similarity" in representative_custom[2]
    assert "512 x 906" in representative_custom[2]
    assert "afaa1b151a6434f7bf24424241b236959062a085e15de242d30fda239a12e50f" in representative_custom[2]
    assert "one handcrafted PPL batch" in representative_custom[4]
    assert "one lm-eval sample" in representative_custom[4]
    assert "representative Mohawk hybrid checkpoints" in representative_custom[4]
    assert "attention-head visualization" in representative_custom[4]
    assert "public tiny CPU and CUDA attention visualizations now run" in rows["Full tools/visualize_attention.py research workflow"][2]
    assert "representative cached Raven checkpoint" in rows["Full tools/visualize_attention.py research workflow"][2]
    assert "hidden-state-similarity visualization" in rows["Full tools/visualize_attention.py research workflow"][2]
    assert "meta-llama/Llama-3.2-3B-Instruct" in rows["Full tools/visualize_attention.py research workflow"][2]
    assert "LocalEntryNotFoundError" in rows["Full tools/visualize_attention.py research workflow"][2]
    assert "outgoing Hugging Face traffic was disabled" in rows["Full tools/visualize_attention.py research workflow"][2]
    assert "selected production heads" in rows["Full tools/visualize_attention.py research workflow"][2]

    assert "requirements.txt now excludes CUDA-built SSM kernels" in rows["Default runtime dependency install"][2]
    assert "transformers>=4.55.0,<4.56" in rows["Default runtime dependency install"][2]
    assert "transformers 5.12.1 removed MambaCache" in rows["Default runtime dependency install"][2]
    assert "lm_eval==0.4.11" in rows["Default runtime dependency install"][2]
    assert "torch 2.12.1+cu130" in rows["Default runtime dependency install"][2]
    assert "mohawk-phase1-default-venv-20260626" in rows["Default runtime dependency install"][2]
    assert "mamba_ssm_spec None" in rows["Default runtime dependency install"][2]
    assert "causal_conv1d_spec None" in rows["Default runtime dependency install"][2]
    assert "103 passed, 1 skipped" in rows["Default runtime dependency install"][2]
    assert "storage-backed default-runtime venv" in rows["Default runtime dependency install"][2]
    assert "/home/abick/storage/mohawk/venvs/ssm-cuda-20260626" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_default_runtime_supervised.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_ddp_default_runtime_supervised.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_fsdp_default_runtime_supervised.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_ddp_default_runtime_mixed_precision.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_fsdp_default_runtime_mixed_precision.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_ddp_default_runtime_bfloat16.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_fsdp_default_runtime_bfloat16.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_ddp_default_runtime_resume.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_fsdp_default_runtime_resume.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_ddp_default_runtime_eval_hstates.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_fsdp_default_runtime_eval_hstates.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_ddp_default_runtime_eval_ppl.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_fsdp_default_runtime_eval_ppl.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_ddp_default_runtime_eval_benchmark.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_fsdp_default_runtime_eval_benchmark.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_ddp_default_runtime_multinode_gradient_accumulation.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_fsdp_default_runtime_multinode_gradient_accumulation.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_ddp_default_runtime_multinode_eval_hstates.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_fsdp_default_runtime_multinode_eval_hstates.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_ddp_default_runtime_multinode_eval_ppl.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_fsdp_default_runtime_multinode_eval_ppl.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_ddp_default_runtime_multinode_eval_benchmark.yaml" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/tiny_cuda_fsdp_default_runtime_multinode_eval_benchmark.yaml" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-Default-Runtime-Supervised-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Supervised-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Supervised-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Mixed-Precision-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Mixed-Precision-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-BFloat16-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-BFloat16-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Resume-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Resume-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Eval-HStates-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Eval-HStates-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Eval-PPL-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Eval-PPL-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Eval-Benchmark-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Eval-Benchmark-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Gradient-Accumulation-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Gradient-Accumulation-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Eval-HStates-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Eval-HStates-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Eval-PPL-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Eval-PPL-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Eval-Benchmark-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Eval-Benchmark-Smoke" in rows["Default runtime dependency install"][2]
    assert "optimizer state entries 12" in rows["Default runtime dependency install"][2]
    assert "WORLD_SIZE=2" in rows["Default runtime dependency install"][2]
    assert "wrappers ddp ddp" in rows["Default runtime dependency install"][2]
    assert "fsdp fsdp" in rows["Default runtime dependency install"][2]
    assert "mixed-precision flags True True" in rows["Default runtime dependency install"][2]
    assert "bfloat16 fallback warnings" in rows["Default runtime dependency install"][2]
    assert "FULL_SHARD" in rows["Default runtime dependency install"][2]
    assert "12 torch.bfloat16 tensors" in rows["Default runtime dependency install"][2]
    assert "3222952" in rows["Default runtime dependency install"][2]
    assert "0bc9e79b22d4fce8a926de7260a144c0a5a0ff09a86328af5f35777ab655af15" in rows["Default runtime dependency install"][2]
    assert "Slurm jobs 691369 and 691375" in rows["Default runtime dependency install"][2]
    assert "scheduler state 1/1 -> 2/2" in rows["Default runtime dependency install"][2]
    assert "optimizer state entries 0 -> 12" in rows["Default runtime dependency install"][2]
    assert "changed all 12 tensors" in rows["Default runtime dependency install"][2]
    assert "14d447df4d4d3c6bfff71734e78cb36c71909b38ffb41be77c8b698739e5401b" in rows["Default runtime dependency install"][2]
    assert "tiny_cuda_ddp_default_runtime_multinode_resume.yaml" in rows["Default runtime dependency install"][2]
    assert "tiny_cuda_fsdp_default_runtime_multinode_resume.yaml" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Resume-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Resume-Smoke" in rows["Default runtime dependency install"][2]
    assert "Slurm jobs 692584 and 692601" in rows["Default runtime dependency install"][2]
    assert "nvl72d053-T10" in rows["Default runtime dependency install"][2]
    assert "nvl72d053-T13" in rows["Default runtime dependency install"][2]
    assert "nvl72d053-T17" in rows["Default runtime dependency install"][2]
    assert "nvl72d053-T18" in rows["Default runtime dependency install"][2]
    assert "5faf34670d9ba7799cb14ed17fb29ec3679c01a003c1d9100783181ae2d6b2d9" in rows["Default runtime dependency install"][2]
    assert "Slurm jobs 691481, 691627, 691806, 691852, 691885, and 691925" in rows["Default runtime dependency install"][2]
    assert "eval_hstates produced 1.6627724170684814" in rows["Default runtime dependency install"][2]
    assert "eval_ppl produced 53548.18359375" in rows["Default runtime dependency install"][2]
    assert "wikitext 183619.266616" in rows["Default runtime dependency install"][2]
    assert "b75cb407af8b36f88a630ae2f4a9ddcc240edbf31548899075c4f8136c729e29" in rows["Default runtime dependency install"][2]
    assert "Slurm jobs 692123 and 692150" in rows["Default runtime dependency install"][2]
    assert "WORLD_SIZE=4" in rows["Default runtime dependency install"][2]
    assert "nvl72d066-T08" in rows["Default runtime dependency install"][2]
    assert "nvl72d066-T09" in rows["Default runtime dependency install"][2]
    assert "nvl72d063-T11" in rows["Default runtime dependency install"][2]
    assert "nvl72d063-T12" in rows["Default runtime dependency install"][2]
    assert "configs/smoke/data_ddp_multinode" in rows["Default runtime dependency install"][2]
    assert "tiny_cuda_ddp_default_runtime_multinode_mixed_precision_gradient_accumulation.yaml" in rows["Default runtime dependency install"][2]
    assert "tiny_cuda_fsdp_default_runtime_multinode_mixed_precision_gradient_accumulation.yaml" in rows["Default runtime dependency install"][2]
    assert "tiny_cuda_ddp_default_runtime_multinode_bfloat16_gradient_accumulation.yaml" in rows["Default runtime dependency install"][2]
    assert "tiny_cuda_fsdp_default_runtime_multinode_bfloat16_gradient_accumulation.yaml" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Mixed-Precision-Gradient-Accumulation-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Mixed-Precision-Gradient-Accumulation-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-BFloat16-Gradient-Accumulation-Smoke" in rows["Default runtime dependency install"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-BFloat16-Gradient-Accumulation-Smoke" in rows["Default runtime dependency install"][2]
    assert "Slurm jobs 692716 and 692810" in rows["Default runtime dependency install"][2]
    assert "Slurm jobs 692872 and 692887" in rows["Default runtime dependency install"][2]
    assert "nvl72d125-T07" in rows["Default runtime dependency install"][2]
    assert "nvl72d125-T08" in rows["Default runtime dependency install"][2]
    assert "nvl72d105-T17" in rows["Default runtime dependency install"][2]
    assert "nvl72d105-T18" in rows["Default runtime dependency install"][2]
    assert "nvl72d054-T01" in rows["Default runtime dependency install"][2]
    assert "nvl72d054-T02" in rows["Default runtime dependency install"][2]
    assert "nvl72d108-T15" in rows["Default runtime dependency install"][2]
    assert "nvl72d108-T18" in rows["Default runtime dependency install"][2]
    assert "Slurm jobs 692419, 692430, 692462, 692471, 692475, and 692491" in rows["Default runtime dependency install"][2]
    assert "eval_hstates produced 1.5936508178710938" in rows["Default runtime dependency install"][2]
    assert "eval_ppl produced 44002.2734375" in rows["Default runtime dependency install"][2]
    assert "wikitext 172057.421054" in rows["Default runtime dependency install"][2]
    assert "multi-node default-runtime evaluation" not in rows["Default runtime dependency install"][4]
    assert "multi-node default-runtime training/evaluation" not in rows["Default runtime dependency install"][4]
    assert "default-runtime distributed eval" not in rows["Default runtime dependency install"][4]
    assert "distributed eval/resume" not in rows["Default runtime dependency install"][4]
    assert "distributed eval/resume or bfloat16" not in rows["Default runtime dependency install"][4]
    assert "requirements-ssm-cuda.txt" in rows["Optional CUDA/SSM dependency install"][1]
    assert "mamba-ssm==2.3.2.post1" in rows["Optional CUDA/SSM dependency install"][2]
    assert "causal-conv1d==1.6.2.post1" in rows["Optional CUDA/SSM dependency install"][2]
    assert "quack-kernels==0.5.3" in rows["Optional CUDA/SSM dependency install"][2]
    assert "/home/abick/storage/mohawk/venvs/ssm-cuda-mamba23-20260630" in rows["Optional CUDA/SSM dependency install"][2]
    assert "torch 2.12.1+cu130" in rows["Optional CUDA/SSM dependency install"][2]
    assert "TileLang 0.1.8" in rows["Optional CUDA/SSM dependency install"][2]
    assert "TVM-FFI 0.1.9" in rows["Optional CUDA/SSM dependency install"][2]
    assert "Quack 0.5.3" in rows["Optional CUDA/SSM dependency install"][2]
    assert "CUTLASS DSL 4.6.0.dev0" in rows["Optional CUDA/SSM dependency install"][2]
    assert "job 748528" in rows["Optional CUDA/SSM dependency install"][2]
    assert "Job 748589" in rows["Optional CUDA/SSM dependency install"][2]
    assert "job 748805" in rows["Optional CUDA/SSM dependency install"][2]
    assert "nvidia-cusparselt-cu13 0.8.1 is not supported on this platform" in rows["Optional CUDA/SSM dependency install"][2]
    assert "source installation without the local wheel" in rows["Optional CUDA/SSM dependency install"][4]
    assert "multi-node Mamba 2.3" in rows["Optional CUDA/SSM dependency install"][4]
    assert "Mamba3 kernels" in rows["Optional CUDA/SSM dependency install"][4]
    assert "centralized, one-node two-rank DDP/FSDP" in rows["CUDA availability"][2]
    assert "two-node four-rank DDP/FSDP" in rows["CUDA availability"][2]
    assert "fast DiscreteMamba2 training" in rows["CUDA availability"][2]
    assert "fresh default /tmp venv" in rows["CUDA availability"][2]
    assert "repo-local shared fresh installs" in rows["CUDA availability"][4]
    assert "multi-node default-runtime evaluation" not in rows["CUDA availability"][4]
    assert "multi-node default-runtime training/evaluation" not in rows["CUDA availability"][4]
    assert "default-runtime distributed eval" not in rows["CUDA availability"][4]
    assert "distributed eval/resume" not in rows["CUDA availability"][4]
    assert "distributed eval/resume or bfloat16" not in rows["CUDA availability"][4]

    assert "First local-only run failed" in rows["Public tiny Hugging Face lm-eval"][2]
    assert "hello EverestMist" in rows["Tiny CPU Mohawk checkpoint generation"][2]
    assert "Number of parameters: 1610832" in rows["Tiny CPU Mohawk checkpoint generation"][2]
    assert "Model dtype: torch.float32" in rows["Tiny CPU Mohawk checkpoint generation"][2]
    assert "generate.py --model_dtype auto" in rows["Tiny CPU bfloat16 Mohawk checkpoint generation"][2]
    assert "mohawk_tiny_bfloat16_fallback_cpu_run/save" in rows["Tiny CPU bfloat16 Mohawk checkpoint generation"][2]
    assert "Model dtype: torch.bfloat16" in rows["Tiny CPU bfloat16 Mohawk checkpoint generation"][2]
    assert "Number of parameters: 1610832" in rows["Tiny CPU bfloat16 Mohawk checkpoint generation"][2]
    assert "hello EverestMist" in rows["Tiny CPU bfloat16 Mohawk checkpoint generation"][2]
    assert "all torch.bfloat16" in rows["Tiny CPU bfloat16 Mohawk checkpoint generation"][2]
    assert "a2943247863988b07f553af2359eaaab936674d8c76148ffa7801a27390b4754" in rows["Tiny CPU bfloat16 Mohawk checkpoint generation"][2]
    assert "CUDA bfloat16" in rows["Tiny CPU bfloat16 Mohawk checkpoint generation"][4]
    assert "DiscreteMamba2 autoregressive step requires mamba_ssm selective_state_update" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "pure-PyTorch reference recurrence" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "hello anguish played" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "hello Bian Jian" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "1612596" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "1612324" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "1612660" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "1612628" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "fast Mamba kernels" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint generation"][4]
    assert "CUDA generation" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint generation"][4]
    assert "Slurm GPU allocation" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "artifacts/gpu_smoke/doubleblock_ref_checkpoints/adapter" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "DoubleBlockAdapter" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "DoubleBlockVanilla" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "DoubleBlockHymba" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "DoubleBlockMerger" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "Model dtype: torch.float32" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "1612596" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "1612324" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "1612660" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "1612628" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "hello anguish played" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "hello Bian Jian" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "8ms" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "6ms" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][2]
    assert "single-process CUDA DoubleBlock-family checkpoint generation" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][4]
    assert "fast Mamba kernels" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][4]
    assert "mamba_ssm" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][4]
    assert "distributed inference" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][4]
    assert "representative checkpoints" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation"][4]
    assert "artifacts/gpu_smoke/tiny_cuda_supervised/save" in rows["Tiny CUDA Mohawk checkpoint generation"][2]
    assert "Slurm GPU allocation" in rows["Tiny CUDA Mohawk checkpoint generation"][2]
    assert "Model dtype: torch.float32" in rows["Tiny CUDA Mohawk checkpoint generation"][2]
    assert "Number of parameters: 1610832" in rows["Tiny CUDA Mohawk checkpoint generation"][2]
    assert "hello EverestMist" in rows["Tiny CUDA Mohawk checkpoint generation"][2]
    assert "prompt processing + decoding time: 4ms" in rows["Tiny CUDA Mohawk checkpoint generation"][2]
    assert "tiny_cuda_bfloat16_supervised/save" in rows["Tiny CUDA bfloat16 Mohawk checkpoint generation"][2]
    assert "Tiny-CUDA-BFloat16-Supervised-Smoke" in rows["Tiny CUDA bfloat16 Mohawk checkpoint generation"][2]
    assert "TrainConfig.model_dtype bfloat16" in rows["Tiny CUDA bfloat16 Mohawk checkpoint generation"][2]
    assert "torch.bfloat16" in rows["Tiny CUDA bfloat16 Mohawk checkpoint generation"][2]
    assert "52fe58e628de69a168b351d41c8d01a52017300497a0675770ae75653ccf1ef1" in rows["Tiny CUDA bfloat16 Mohawk checkpoint generation"][2]
    assert "Model dtype: torch.bfloat16" in rows["Tiny CUDA bfloat16 Mohawk checkpoint generation"][2]
    assert "Number of parameters: 1610832" in rows["Tiny CUDA bfloat16 Mohawk checkpoint generation"][2]
    assert "hello EverestMist" in rows["Tiny CUDA bfloat16 Mohawk checkpoint generation"][2]
    assert "prompt processing + decoding time: 4ms" in rows["Tiny CUDA bfloat16 Mohawk checkpoint generation"][2]
    assert "single-process CUDA bfloat16 checkpoint generation" in rows["Tiny CUDA bfloat16 Mohawk checkpoint generation"][4]
    assert "tiny CPU/CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation" in rows["Representative Mohawk checkpoint generation"][2]
    assert "tiny CUDA float32 checkpoint generation" in rows["Representative Mohawk checkpoint generation"][2]
    assert "tiny CUDA bfloat16 checkpoint generation smokes" in rows["Representative Mohawk checkpoint generation"][2]
    assert "do not validate large checkpoints" in rows["Representative Mohawk checkpoint generation"][2]
    assert "network-backed" in rows["Public tiny Hugging Face lm-eval"][2]
    assert "subsequent --local_files_only run" in rows["Public tiny Hugging Face lm-eval"][2]
    assert "wikitext 206463.186451" in rows["Tiny CPU Mohawk checkpoint lm-eval smoke"][2]
    assert "Model dtype: torch.float32" in rows["Tiny CPU Mohawk checkpoint lm-eval smoke"][2]
    assert "hardcoded TrainConfig.model_dtype: float32" in rows["Tiny CPU bfloat16 Mohawk checkpoint lm-eval smoke"][2]
    assert "mohawk_tiny_bfloat16_fallback_cpu_run/save" in rows["Tiny CPU bfloat16 Mohawk checkpoint lm-eval smoke"][2]
    assert "model_dtype: bfloat16" in rows["Tiny CPU bfloat16 Mohawk checkpoint lm-eval smoke"][2]
    assert "Model dtype: torch.bfloat16" in rows["Tiny CPU bfloat16 Mohawk checkpoint lm-eval smoke"][2]
    assert "wikitext 206463.186451" in rows["Tiny CPU bfloat16 Mohawk checkpoint lm-eval smoke"][2]
    assert "CUDA bfloat16" in rows["Tiny CPU bfloat16 Mohawk checkpoint lm-eval smoke"][4]
    assert "mohawk_tiny_doubleblock_discrete_mamba2_ref_cpu_run/save" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "mohawk_tiny_doubleblock_vanilla_discrete_mamba2_ref_cpu_run/save" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "mohawk_tiny_doubleblock_hymba_discrete_mamba2_ref_cpu_run/save" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "mohawk_tiny_doubleblock_merger_discrete_mamba2_ref_cpu_run/save" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "19/19, 17/17, 23/23, and 21/21 keys" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "centralized inference wrapper" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "sshleifer/tiny-gpt2" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "Model dtype: torch.float32" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "wikitext 206463.186451" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "wikitext 218913.474906" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "fast Mamba kernels" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][4]
    assert "CUDA lm-eval" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][4]
    assert "distributed evaluation" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][4]
    assert "representative checkpoints" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][4]
    assert "artifacts/gpu_smoke/doubleblock_ref_checkpoints/adapter" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "artifacts/gpu_smoke/lm_eval_datasets" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "HF_DATASETS_OFFLINE=1" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "--device cuda" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "DoubleBlockAdapter" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "DoubleBlockVanilla" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "DoubleBlockHymba" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "DoubleBlockMerger" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "19/19, 17/17, 23/23, and 21/21 keys" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "cuda:0" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "sshleifer/tiny-gpt2" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "Model dtype: torch.float32" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "wikitext 206463.186451" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "wikitext 218913.474906" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][2]
    assert "single-process CUDA DoubleBlock-family checkpoint lm-eval" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][4]
    assert "fast Mamba kernels" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][4]
    assert "distributed evaluation" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][4]
    assert "representative checkpoints" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smoke"][4]
    assert "tiny CPU/CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval smokes" in rows["Representative Mohawk checkpoint lm-eval"][2]
    assert "EvalConfig -> Evaluator -> evals.benchmark.evaluator" in rows["Tiny CPU run.py benchmark callback smoke"][2]
    assert "read-only home datasets cache" in rows["Tiny CPU run.py benchmark callback smoke"][2]
    assert "datasets.config.HF_DATASETS_CACHE" in rows["Tiny CPU run.py benchmark callback smoke"][2]
    assert "wikitext 206537.186039" in rows["Tiny CPU run.py benchmark callback smoke"][2]
    assert "mohawk_tiny_eval_benchmark_cpu_run/save" in rows["Tiny CPU run.py benchmark callback smoke"][2]
    assert "8a5c277726a85ba8b5646cebb6294fc867d8824e98b208dfca303c2747349370" in rows["Tiny CPU run.py benchmark callback smoke"][2]
    assert "pre-populated public task data" in rows["Tiny CPU run.py benchmark callback smoke"][4]
    assert "tiny_cpu_bfloat16_eval_benchmark.yaml" in rows["Tiny CPU run.py bfloat16 benchmark callback smoke"][2]
    assert "model_dtype: bfloat16" in rows["Tiny CPU run.py bfloat16 benchmark callback smoke"][2]
    assert "wikitext 209029.094266" in rows["Tiny CPU run.py bfloat16 benchmark callback smoke"][2]
    assert "mohawk_tiny_bfloat16_eval_benchmark_cpu_run/save" in rows["Tiny CPU run.py bfloat16 benchmark callback smoke"][2]
    assert "EvalConfig[0].Evaluation benchmark" in rows["Tiny CPU run.py bfloat16 benchmark callback smoke"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CPU run.py bfloat16 benchmark callback smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py bfloat16 benchmark callback smoke"][2]
    assert "a2943247863988b07f553af2359eaaab936674d8c76148ffa7801a27390b4754" in rows["Tiny CPU run.py bfloat16 benchmark callback smoke"][2]
    assert "CUDA bfloat16" in rows["Tiny CPU run.py bfloat16 benchmark callback smoke"][4]
    assert "tiny_cuda_ddp_eval_benchmark.yaml" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "tiny_cuda_fsdp_eval_benchmark.yaml" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "tiny_cuda_ddp_default_runtime_eval_benchmark.yaml" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "tiny_cuda_fsdp_default_runtime_eval_benchmark.yaml" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "tiny_cuda_ddp_multinode_eval_benchmark.yaml" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "tiny_cuda_fsdp_multinode_eval_benchmark.yaml" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "tiny_cuda_ddp_default_runtime_multinode_eval_benchmark.yaml" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "tiny_cuda_fsdp_default_runtime_multinode_eval_benchmark.yaml" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "HF_DATASETS_CACHE=/home/abick/mohawk/artifacts/gpu_smoke/lm_eval_datasets" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "limit >= WORLD_SIZE" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "task.build_requests() did not find any docs!" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "wikitext_detokenizer" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "aggregated pandas results" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "lm-eval returned None on nonzero ranks" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "aggregates only on rank 0" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "Slurm job 688237" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "eval_score: 183619.266616" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "wikitext 183619.266616" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "AVG 183619.266616" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "tiny_cuda_ddp_eval_benchmark/save" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "Slurm job 688259" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "wrappers fsdp fsdp" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "FULL_SHARD FULL_SHARD" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "Tiny-CUDA-FSDP-Eval-Benchmark-Smoke" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "tiny_cuda_fsdp_eval_benchmark/save" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "b75cb407af8b36f88a630ae2f4a9ddcc240edbf31548899075c4f8136c729e29" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "matching DDP/FSDP hash" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Eval-Benchmark-Smoke" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Eval-Benchmark-Smoke" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "Slurm jobs 691885 and 691925" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "tiny_cuda_ddp_default_runtime_eval_benchmark/save" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "tiny_cuda_fsdp_default_runtime_eval_benchmark/save" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "optimizer state entries 12" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "matching default-runtime benchmark hash" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "Slurm job 688389" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "Inplace update to inference tensor outside InferenceMode" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "torch.no_grad()" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "Slurm job 688406" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "Slurm job 688414" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "WORLD_SIZE=4" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "configs/smoke/data_ddp_multinode" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "benchmark limit: 4" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "eval_score: 172057.421054" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "wikitext 172057.421054" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "tiny_cuda_ddp_multinode_eval_benchmark/save" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "tiny_cuda_fsdp_multinode_eval_benchmark/save" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "scheduler _step_count == 1" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Eval-Benchmark-Smoke" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Eval-Benchmark-Smoke" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "Slurm jobs 692475 and 692491" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "nvl72d111-T12" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "nvl72d111-T16" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "nvl72d112-T10" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "nvl72d112-T14" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "tiny_cuda_ddp_default_runtime_multinode_eval_benchmark/save" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "tiny_cuda_fsdp_default_runtime_multinode_eval_benchmark/save" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "post-save FSDP rendezvous shutdown warning" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "matching default-runtime multi-node benchmark hash" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][2]
    assert "one-node two-rank and two-node four-rank under shared ML env" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][4]
    assert "one-node two-rank and two-node four-rank storage-backed default runtime" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][4]
    assert "mixed-precision or bfloat16 distributed benchmark callbacks" in rows["Tiny CUDA DDP/FSDP run.py benchmark callback smokes"][4]
    assert "tiny_cpu_bfloat16_eval_ppl.yaml" in rows["Tiny CPU run.py bfloat16 eval_ppl callback smoke"][2]
    assert "TrainConfig.model_dtype: bfloat16" in rows["Tiny CPU run.py bfloat16 eval_ppl callback smoke"][2]
    assert "TeacherConfig.model_dtype: bfloat16" in rows["Tiny CPU run.py bfloat16 eval_ppl callback smoke"][2]
    assert "EvalConfig -> eval_ppl" in rows["Tiny CPU run.py bfloat16 eval_ppl callback smoke"][2]
    assert "EvalConfig[0].Evaluation eval_ppl" in rows["Tiny CPU run.py bfloat16 eval_ppl callback smoke"][2]
    assert "49020.812500" in rows["Tiny CPU run.py bfloat16 eval_ppl callback smoke"][2]
    assert "latest_Perplexity" in rows["Tiny CPU run.py bfloat16 eval_ppl callback smoke"][2]
    assert "Dataloader state is not implemented" in rows["Tiny CPU run.py bfloat16 eval_ppl callback smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py bfloat16 eval_ppl callback smoke"][2]
    assert "e5185fe0ec38fae0b521167cd3fa54d81d2b15b9025f33b3b8af71a89a38a2b3" in rows["Tiny CPU run.py bfloat16 eval_ppl callback smoke"][2]
    assert "CUDA bfloat16" in rows["Tiny CPU run.py bfloat16 eval_ppl callback smoke"][4]
    assert "tiny_cpu_bfloat16_eval_ppl_save_best.yaml" in rows["Tiny CPU run.py bfloat16 eval_ppl save_best checkpoint smoke"][2]
    assert "TrainConfig.model_dtype: bfloat16" in rows["Tiny CPU run.py bfloat16 eval_ppl save_best checkpoint smoke"][2]
    assert "TeacherConfig.model_dtype: bfloat16" in rows["Tiny CPU run.py bfloat16 eval_ppl save_best checkpoint smoke"][2]
    assert "EvalConfig -> eval_ppl" in rows["Tiny CPU run.py bfloat16 eval_ppl save_best checkpoint smoke"][2]
    assert "save_best True" in rows["Tiny CPU run.py bfloat16 eval_ppl save_best checkpoint smoke"][2]
    assert "save_latest False" in rows["Tiny CPU run.py bfloat16 eval_ppl save_best checkpoint smoke"][2]
    assert "Saving best model with Perplexity score of 49020.8125" in rows["Tiny CPU run.py bfloat16 eval_ppl save_best checkpoint smoke"][2]
    assert "best_Perplexity" in rows["Tiny CPU run.py bfloat16 eval_ppl save_best checkpoint smoke"][2]
    assert "Dataloader state is not implemented" in rows["Tiny CPU run.py bfloat16 eval_ppl save_best checkpoint smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py bfloat16 eval_ppl save_best checkpoint smoke"][2]
    assert "e5185fe0ec38fae0b521167cd3fa54d81d2b15b9025f33b3b8af71a89a38a2b3" in rows["Tiny CPU run.py bfloat16 eval_ppl save_best checkpoint smoke"][2]
    assert "CUDA bfloat16" in rows["Tiny CPU run.py bfloat16 eval_ppl save_best checkpoint smoke"][4]
    assert "tiny CPU float32 checkpoint" in rows["Representative Mohawk checkpoint lm-eval"][2]
    assert "tiny CPU bfloat16 checkpoint" in rows["Representative Mohawk checkpoint lm-eval"][2]
    assert "production hybrid checkpoints" in rows["Representative Mohawk checkpoint lm-eval"][2]
    assert "50496.07421875" in rows["Public tiny Hugging Face perplexity evaluator"][2]
    assert '"model_dtype": "torch.float32"' in rows["Public tiny Hugging Face perplexity evaluator"][2]
    assert "46163.3125" in rows["Tiny CPU Mohawk checkpoint perplexity evaluator"][2]
    assert '"model_dtype": "torch.float32"' in rows["Tiny CPU Mohawk checkpoint perplexity evaluator"][2]
    assert "mohawk_tiny_bfloat16_fallback_cpu_run/save" in rows["Tiny CPU bfloat16 Mohawk checkpoint perplexity evaluator"][2]
    assert "49020.8125" in rows["Tiny CPU bfloat16 Mohawk checkpoint perplexity evaluator"][2]
    assert '"model_dtype": "torch.bfloat16"' in rows["Tiny CPU bfloat16 Mohawk checkpoint perplexity evaluator"][2]
    assert "default --backend auto" in rows["Tiny CPU bfloat16 Mohawk checkpoint perplexity evaluator"][2]
    assert "all torch.bfloat16" in rows["Tiny CPU bfloat16 Mohawk checkpoint perplexity evaluator"][2]
    assert "a2943247863988b07f553af2359eaaab936674d8c76148ffa7801a27390b4754" in rows["Tiny CPU bfloat16 Mohawk checkpoint perplexity evaluator"][2]
    assert "CUDA bfloat16" in rows["Tiny CPU bfloat16 Mohawk checkpoint perplexity evaluator"][4]
    assert "mohawk_tiny_doubleblock_discrete_mamba2_ref_cpu_run/save" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "mohawk_tiny_doubleblock_vanilla_discrete_mamba2_ref_cpu_run/save" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "mohawk_tiny_doubleblock_hymba_discrete_mamba2_ref_cpu_run/save" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "mohawk_tiny_doubleblock_merger_discrete_mamba2_ref_cpu_run/save" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "centralized inference wrapper" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "eval_ppl.evaluator" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "JSON/tokenizer/padding/Torch dataloader" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "66080.9453125" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "42200.625" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "64897.3828125" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "70790.328125" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert '"model_dtype": "torch.float32"' in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "fast Mamba kernels" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][4]
    assert "CUDA eval" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][4]
    assert "distributed wrappers" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][4]
    assert "representative checkpoints" in rows["Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][4]
    assert "FileNotFoundError" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "node-local path" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "artifacts/gpu_smoke/doubleblock_ref_checkpoints" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "DoubleBlockAdapter" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "DoubleBlockVanilla" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "DoubleBlockHymba" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "DoubleBlockMerger" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "--device cuda" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "66080.9453125" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "42200.62890625" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "64897.38671875" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "70790.3203125" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert '"model_dtype": "torch.float32"' in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][2]
    assert "single-process CUDA DoubleBlock-family checkpoint PPL" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][4]
    assert "distributed wrappers" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][4]
    assert "representative checkpoints" in rows["Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint perplexity evaluator"][4]
    assert "artifacts/gpu_smoke/tiny_cuda_supervised/save" in rows["Tiny CUDA Mohawk checkpoint perplexity evaluator"][2]
    assert "--device cuda" in rows["Tiny CUDA Mohawk checkpoint perplexity evaluator"][2]
    assert "Slurm GPU allocation" in rows["Tiny CUDA Mohawk checkpoint perplexity evaluator"][2]
    assert "46136.37890625" in rows["Tiny CUDA Mohawk checkpoint perplexity evaluator"][2]
    assert '"model_dtype": "torch.float32"' in rows["Tiny CUDA Mohawk checkpoint perplexity evaluator"][2]
    assert "single-process CUDA PPL smoke" in rows["Tiny CUDA Mohawk checkpoint perplexity evaluator"][4]
    assert "tiny_cuda_bfloat16_supervised/save" in rows["Tiny CUDA bfloat16 Mohawk checkpoint perplexity evaluator"][2]
    assert "Tiny-CUDA-BFloat16-Supervised-Smoke" in rows["Tiny CUDA bfloat16 Mohawk checkpoint perplexity evaluator"][2]
    assert "TrainConfig.model_dtype bfloat16" in rows["Tiny CUDA bfloat16 Mohawk checkpoint perplexity evaluator"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CUDA bfloat16 Mohawk checkpoint perplexity evaluator"][2]
    assert "torch.bfloat16" in rows["Tiny CUDA bfloat16 Mohawk checkpoint perplexity evaluator"][2]
    assert "52fe58e628de69a168b351d41c8d01a52017300497a0675770ae75653ccf1ef1" in rows["Tiny CUDA bfloat16 Mohawk checkpoint perplexity evaluator"][2]
    assert "49020.80859375" in rows["Tiny CUDA bfloat16 Mohawk checkpoint perplexity evaluator"][2]
    assert '"model_dtype": "torch.bfloat16"' in rows["Tiny CUDA bfloat16 Mohawk checkpoint perplexity evaluator"][2]
    assert "single-process CUDA bfloat16 PPL smoke" in rows["Tiny CUDA bfloat16 Mohawk checkpoint perplexity evaluator"][4]
    assert "tiny CPU/CUDA DoubleBlock DiscreteMamba2 reference checkpoint" in rows["Representative Mohawk perplexity evaluator"][2]
    assert "tiny CUDA float32 Mohawk checkpoint" in rows["Representative Mohawk perplexity evaluator"][2]
    assert "tiny CUDA bfloat16 Mohawk checkpoint PPL smokes" in rows["Representative Mohawk perplexity evaluator"][2]
    assert 'checkpoint_dir / "config.json"' in rows["Representative Mohawk perplexity evaluator"][2]
    assert "HYBRID_SHARD" in rows["Representative Mohawk perplexity evaluator"][2]
    assert "configs/Llama/8B" in rows["Representative Mohawk perplexity evaluator"][2]
    assert "failed before model initialization or checkpoint save" in rows["Representative Mohawk perplexity evaluator"][2]
    assert "do not validate representative checkpoints" in rows["Representative Mohawk perplexity evaluator"][2]
    assert "real representative local Mohawk checkpoint directory" in rows["Representative Mohawk perplexity evaluator"][4]
    assert "evals/eval_ppl.py --backend mohawk" in rows["Representative Mohawk perplexity evaluator"][4]
    assert "8 tests" in rows["Local JSON dataloader path"][2]
    assert "consumed samples" in rows["Local JSON dataloader path"][2]
    assert "configurable streaming shuffle-buffer" in rows["Local JSON dataloader path"][2]
    assert "streaming=True" in rows["Public remote streaming HFDataset + Tokenize smoke"][2]
    assert "_index: -1" in rows["Public remote streaming HFDataset + Tokenize smoke"][2]
    assert "_index: 1" in rows["Public remote streaming HFDataset + Tokenize smoke"][2]
    assert "first tokens [14036, 198, 198, 40, 423]" in rows["Public remote streaming HFDataset + Tokenize smoke"][2]
    assert "HFDataset -> Tokenize -> PaddingDataLoader -> CycleDataLoader -> TorchDataLoader -> run.py" in rows["Tiny CPU run.py public HFDataset training smoke"][2]
    assert "NeelNanda/pile-10k" in rows["Tiny CPU run.py public HFDataset training smoke"][2]
    assert "ignored HF_DATASETS_CACHE" in rows["Tiny CPU run.py public HFDataset training smoke"][2]
    assert 'cache_dir=os.environ.get("HF_DATASETS_CACHE")' in rows["Tiny CPU run.py public HFDataset training smoke"][2]
    assert "HF_DATASETS_OFFLINE=1" in rows["Tiny CPU run.py public HFDataset training smoke"][2]
    assert "110821dc9516f4eabc3c66ea9fb20697e40803f889786bc257f4deed9e1b732f" in rows["Tiny CPU run.py public HFDataset training smoke"][2]
    assert "does not validate large/gated datasets" in rows["Tiny CPU run.py public HFDataset training smoke"][4]
    assert "distributed sharding" in rows["Tiny CPU run.py public HFDataset training smoke"][4]
    assert "allenai/c4" in rows["Public C4 streaming HFDataset + Tokenize smoke"][2]
    assert "shuffle_buffer_size=2" in rows["Public C4 streaming HFDataset + Tokenize smoke"][2]
    assert "batch_shape (1, 495)" in rows["Public C4 streaming HFDataset + Tokenize smoke"][2]
    assert "first tokens [33, 522, 12917, 784, 3250]" in rows["Public C4 streaming HFDataset + Tokenize smoke"][2]
    assert "does not validate full training" in rows["Public C4 streaming HFDataset + Tokenize smoke"][4]
    assert "tiny_cpu_fineweb_edu_streaming_supervised.yaml" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "HuggingFaceFW/fineweb-edu" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "sample-100BT" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "streaming: true" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "shuffle_buffer_size: 2" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "/home/abick/storage/mohawk/fineweb_edu_streaming_smoke" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "12/12 teacher keys" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "exited with signal 139" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "Fatal Python error: PyGILState_Release" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "dataloader-only probe consumed a (1, 8) batch and exited cleanly" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "full run.py retry still exited 139" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "_step_count == 2" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "71b740e0d4bee1b557cb83ce2e0db4a8c1352adfed7148b369edced4956f0b3e" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "minimal Hugging Face" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "429 Too Many Requests" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "HF_TOKEN: " in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "no token was available" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "tiny_cpu_fineweb_edu_streaming_supervised_default_runtime.yaml" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "/home/abick/storage/mohawk/fineweb_edu_streaming_smoke_default_runtime" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "datasets 5.0.0" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "exited with status 0" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "Tokenize.local_files_only: false" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "9e447f7af4d9d10fbd0bab04b7078d39bc6239f1681a73cdebc2c8c7762676d2" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "19M" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "default 10000 shuffle buffer" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][4]
    assert "production PackingDataLoader" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][4]
    assert "shared ML env/datasets 3.3.0 finalization crash" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][4]
    assert "optional-dependency stub" in rows["Legacy local C4DataLoader"][2]
    assert "No module named 'streaming'" in rows["Legacy local C4DataLoader"][2]
    assert "former placeholder references" in rows["Legacy local C4DataLoader"][2]
    assert "removed or replaced with current public HFDataset C4 config" in rows["Legacy local C4DataLoader"][2]
    assert "keep this registry path unadvertised" in rows["Legacy local C4DataLoader"][4]
    assert "public C4 streaming one-batch smoke" in rows["Large/gated remote training datasets"][2]
    assert "storage-backed tiny FineWeb-Edu default-runtime run.py smoke" in rows["Large/gated remote training datasets"][2]
    assert "bounded public-data paths" in rows["Large/gated remote training datasets"][2]
    assert "9e447f7af4d9d10fbd0bab04b7078d39bc6239f1681a73cdebc2c8c7762676d2" in rows["Large/gated remote training datasets"][2]
    assert "55d4b4188067a61a6fcd6799b460e2059536b6b92f265c9e74e773ea96f600a1" in rows["Large/gated remote training datasets"][2]
    assert "Fatal Python error: PyGILState_Release" in rows["Large/gated remote training datasets"][2]
    assert "429 Too Many Requests" in rows["Large/gated remote training datasets"][2]
    assert "configs/Llama/8B/llama/hstates.yaml" in rows["Large/gated remote training datasets"][2]
    assert "HYBRID_SHARD" in rows["Large/gated remote training datasets"][2]
    assert "HuggingFaceTB/finemath" in rows["Large/gated remote training datasets"][2]
    assert "PackingDataLoader.max_seq_len: 2048" in rows["Large/gated remote training datasets"][2]
    assert "before model init, teacher load, optimizer construction, or training" in rows["Large/gated remote training datasets"][2]
    fineweb_cuda = rows["Tiny CUDA run.py FineWeb-Edu production-loader training smoke"]
    assert "tiny_cuda_fineweb_edu_production_loader_supervised.yaml" in fineweb_cuda[2]
    assert "runtime default 10000" in fineweb_cuda[2]
    assert "PackingDataLoader.max_seq_len: 2048" in fineweb_cuda[2]
    assert "Slurm job 706699" in fineweb_cuda[2]
    assert "12 optimizer states" in fineweb_cuda[2]
    assert "55d4b4188067a61a6fcd6799b460e2059536b6b92f265c9e74e773ea96f600a1" in fineweb_cuda[2]
    fineweb_distributed = rows["Tiny CUDA DDP/FSDP FineWeb-Edu production-loader training smokes"]
    assert "split_dataset_by_node" in fineweb_distributed[2]
    assert "Slurm job 706647" in fineweb_distributed[2]
    assert "zero optimizer states" in fineweb_distributed[2]
    assert "job 706730" in fineweb_distributed[2]
    assert "429 Too Many Requests" in fineweb_distributed[2]
    assert "tiny_cuda_ddp_fineweb_edu_direct_shard_supervised.yaml" in fineweb_distributed[2]
    assert "tiny_cuda_fsdp_fineweb_edu_direct_shard_supervised.yaml" in fineweb_distributed[2]
    assert "Slurm jobs 757646 and 757737" in fineweb_distributed[2]
    assert "12 optimizer state entries" in fineweb_distributed[2]
    assert "4894e81ba6c093dcbe2e603f798803d8564a7d5d683b15c170fc07f5fb9dbd46" in fineweb_distributed[2]
    assert "one explicit public parquet shard" in fineweb_distributed[4]
    hybrid_family = rows["Hybrid adapter/merger/vanilla/hymba configs"]
    assert "all four block combiners" in hybrid_family[2]
    assert "attention-only, DiscreteMamba2 reference, and fast-kernel paths" in hybrid_family[2]
    assert "1.80B-parameter DoubleBlockAdapter" in hybrid_family[2]
    assert "real public Qwen weights/tokenizer" in hybrid_family[2]
    assert "production Llama scenario" in hybrid_family[2]
    assert "4.07B-parameter ARCH20 forward/backward" in hybrid_family[2]
    assert "other block combiners" in hybrid_family[4]
    assert "full CUDA DoubleBlock family" not in rows["Hybrid adapter/merger/vanilla/hymba configs"][2]
    assert "Qwen2Model -> Qwen2Block -> Qwen2Attention" in rows["Tiny CPU run.py Qwen2 supervised training smoke"][2]
    assert "loaded 15/15 teacher keys" in rows["Tiny CPU run.py Qwen2 supervised training smoke"][2]
    assert "mohawk_tiny_qwen2_cpu_run/save" in rows["Tiny CPU run.py Qwen2 supervised training smoke"][2]
    assert "Llama-style, Qwen2, PhiAttention, and FalconBlock registry stacks" in rows["Tiny local Llama/Qwen/Phi/Falcon component paths"][2]
    assert "MambaPhi -> PhiAttention" in rows["Tiny CPU run.py Phi supervised training smoke"][2]
    assert "Cannot copy out of meta tensor" in rows["Tiny CPU run.py Phi supervised training smoke"][2]
    assert "loaded 18/18 teacher keys" in rows["Tiny CPU run.py Phi supervised training smoke"][2]
    assert "mohawk_tiny_phi_cpu_run/save" in rows["Tiny CPU run.py Phi supervised training smoke"][2]
    assert "FalconBlock -> FalconMambaMixer" in rows["Tiny CPU run.py Falcon supervised training smoke"][2]
    assert "Buffer is a Parameter" in rows["Tiny CPU run.py Falcon supervised training smoke"][2]
    assert "loaded 13/13 teacher keys" in rows["Tiny CPU run.py Falcon supervised training smoke"][2]
    assert "mohawk_tiny_falcon_cpu_run/save" in rows["Tiny CPU run.py Falcon supervised training smoke"][2]
    assert "sequential fallback" in rows["Tiny CPU run.py Falcon supervised training smoke"][4]
    assert "mamba-ssm 2.2.6.post3" in rows["Mamba/SSM fast CUDA kernel components"][2]
    assert "causal-conv1d 1.6.2.post1" in rows["Mamba/SSM fast CUDA kernel components"][2]
    assert "Slurm job 706111" in rows["Mamba/SSM fast CUDA kernel components"][2]
    assert "NVIDIA GB300" in rows["Mamba/SSM fast CUDA kernel components"][2]
    assert "use_ref_impl=False" in rows["Mamba/SSM fast CUDA kernel components"][2]
    assert "initializer={'z':'default','out':'default','convolution':'identity'}" in rows["Mamba/SSM fast CUDA kernel components"][2]
    assert "parameter-gradient sum was 0.08953561447560787" in rows["Mamba/SSM fast CUDA kernel components"][2]
    assert "DoubleBlockAdapter" in rows["Mamba/SSM fast CUDA kernel components"][2]
    assert "DoubleBlockVanilla" in rows["Mamba/SSM fast CUDA kernel components"][2]
    assert "Mamba 2.3.2.post1" in rows["Mamba/SSM fast CUDA kernel components"][2]
    assert "job 748528" in rows["Mamba/SSM fast CUDA kernel components"][2]
    assert "job 748589" in rows["Mamba/SSM fast CUDA kernel components"][2]
    assert "job 748805" in rows["Mamba/SSM fast CUDA kernel components"][2]
    fresh_mamba23 = rows["Fresh Mamba 2.3 optional-stack install and Mohawk CUDA smoke"]
    assert "tiny_cuda_doubleblock_discrete_mamba2_fast_mamba23_supervised.yaml" in fresh_mamba23[1]
    assert "job 748228" in fresh_mamba23[2]
    assert "/home/abick/storage/mohawk/venvs/ssm-cuda-mamba23-20260630" in fresh_mamba23[2]
    assert "Mamba 2.3.2.post1" in fresh_mamba23[2]
    assert "TileLang 0.1.8" in fresh_mamba23[2]
    assert "TVM-FFI 0.1.9" in fresh_mamba23[2]
    assert "Quack 0.5.3" in fresh_mamba23[2]
    assert "CUTLASS DSL 4.6.0.dev0" in fresh_mamba23[2]
    assert "job 748528" in fresh_mamba23[2]
    assert "Job 748589" in fresh_mamba23[2]
    assert "Job 748805" in fresh_mamba23[2]
    assert "19 optimizer states" in fresh_mamba23[2]
    assert "f7c108e9d676a1a4e81f1ffe0d95d5639e345cf8696359c009305940305224cf" in fresh_mamba23[2]
    assert "hello anguish played" in fresh_mamba23[2]
    assert fresh_mamba23[3] == "PASSED_REAL_SMOKE"
    assert "source installation without the cached wheel" in fresh_mamba23[4]
    assert "Mamba3 kernels" in fresh_mamba23[4]
    assert "multi-node Mamba 2.3" in fresh_mamba23[4]
    assert "representative lengths" in fresh_mamba23[4]
    fast_row = rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 fast training smokes"]
    assert "tiny_cuda_doubleblock*_discrete_mamba2_fast_supervised.yaml" in fast_row[2]
    assert "tiny_cuda_ddp_doubleblock*_discrete_mamba2_fast_supervised.yaml" in fast_row[2]
    assert "tiny_cuda_fsdp_doubleblock*_discrete_mamba2_fast_supervised.yaml" in fast_row[2]
    assert "DoubleBlockVanilla" in fast_row[2]
    assert "DoubleBlockHymba" in fast_row[2]
    assert "DoubleBlockMerger" in fast_row[2]
    assert "Slurm job 746718" in fast_row[2]
    assert "Slurm job 746732" in fast_row[2]
    assert "WORLD_SIZE=2" in fast_row[2]
    assert "FULL_SHARD" in fast_row[2]
    assert "0041dc4ab3b24ccff2d5368840b72f6499ca1385d6e1c63dfa96c22bc8588f6f" in fast_row[2]
    assert "c50c1bfa8e56246bff13e2e3b51d72e118d11f4a74b8f5d0bc0b7405f6ef0ce6" in fast_row[2]
    multinode_fast_row = rows[
        "Tiny CUDA multi-node DDP/FSDP run.py DoubleBlock DiscreteMamba2 fast training smokes"
    ]
    assert "tiny_cuda_ddp_multinode_doubleblock_discrete_mamba2_fast_supervised.yaml" in multinode_fast_row[2]
    assert "tiny_cuda_fsdp_multinode_doubleblock_discrete_mamba2_fast_supervised.yaml" in multinode_fast_row[2]
    assert "use_ref_impl: false" in multinode_fast_row[2]
    assert "configs/smoke/data_ddp_multinode" in multinode_fast_row[2]
    assert "effective_batch_size: 8" in multinode_fast_row[2]
    assert "n_tokens: 128" in multinode_fast_row[2]
    assert "Slurm jobs 747896 and 747933" in multinode_fast_row[2]
    assert "WORLD_SIZE=4" in multinode_fast_row[2]
    assert "wrappers ddp ddp" in multinode_fast_row[2]
    assert "wrappers fsdp fsdp" in multinode_fast_row[2]
    assert "FULL_SHARD FULL_SHARD" in multinode_fast_row[2]
    assert "Total Grad Steps: 2" in multinode_fast_row[2]
    assert "19 optimizer states" in multinode_fast_row[2]
    assert "0fe968211b13f8e38a20391db9fe7aca14698f159f215848ed9dbedaae792272" in multinode_fast_row[2]
    assert "more than two nodes" in multinode_fast_row[4]
    assert "production scale" in multinode_fast_row[4]
    fast_generation_row = rows["Tiny CUDA DoubleBlock DiscreteMamba2 fast checkpoint generation"]
    assert "KeyError: 'conv'" in fast_generation_row[2]
    assert "CUDA-graph capture" in fast_generation_row[2]
    assert "causal_conv1d" in fast_generation_row[2]
    assert "hello Bian Jian" in fast_generation_row[2]
    assert "7ms" in fast_generation_row[2]
    assert "component-only config.json" in fast_generation_row[2]
    assert "config.yaml" in fast_generation_row[2]
    assert "Slurm job 748021" in fast_generation_row[2]
    assert "LayeredMambaLM -> LlamaModel -> LlamaBlock -> DiscreteMamba2" in rows["Tiny CPU run.py DiscreteMamba2 reference training smoke"][2]
    assert "use_ref_impl: true" in rows["Tiny CPU run.py DiscreteMamba2 reference training smoke"][2]
    assert "mamba_ssm_spec None" in rows["Tiny CPU run.py DiscreteMamba2 reference training smoke"][2]
    assert "causal_conv1d_spec None" in rows["Tiny CPU run.py DiscreteMamba2 reference training smoke"][2]
    assert "loaded 13/13 teacher keys" in rows["Tiny CPU run.py DiscreteMamba2 reference training smoke"][2]
    assert "can only concatenate tuple" in rows["Tiny CPU run.py DiscreteMamba2 reference training smoke"][2]
    assert "mohawk_tiny_discrete_mamba2_ref_cpu_run/save" in rows["Tiny CPU run.py DiscreteMamba2 reference training smoke"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CPU run.py DiscreteMamba2 reference training smoke"][2]
    assert "d0c1a65b9aa684fda86f08e60addbaaaf22e28fac02faaa32fd433d7d852c633" in rows["Tiny CPU run.py DiscreteMamba2 reference training smoke"][2]
    assert "fast Mamba kernels" in rows["Tiny CPU run.py DiscreteMamba2 reference training smoke"][4]
    assert "LayeredMambaLM -> LlamaModel -> DoubleBlockAdapter" in rows["Tiny CPU run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "DiscreteMamba2 use_ref_impl: true" in rows["Tiny CPU run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "mixer2" in rows["Tiny CPU run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "mamba_ssm_spec None" in rows["Tiny CPU run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "causal_conv1d_spec None" in rows["Tiny CPU run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "loaded 19/19 teacher keys" in rows["Tiny CPU run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "mohawk_tiny_doubleblock_discrete_mamba2_ref_cpu_run/save" in rows["Tiny CPU run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CPU run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "fee4e7c14156ce0404efa63f631a89d6b1effb895f4f5c77e40696a4035ddc46" in rows["Tiny CPU run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "all 19 tensors changed" in rows["Tiny CPU run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "fast Mamba kernels" in rows["Tiny CPU run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][4]
    assert "configs/smoke/tiny_cuda_doubleblock_discrete_mamba2_ref_supervised.yaml" in rows["Tiny CUDA run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "LayeredMambaLM -> LlamaModel -> DoubleBlockAdapter" in rows["Tiny CUDA run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "DiscreteMamba2 use_ref_impl: true" in rows["Tiny CUDA run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "cuda:0" in rows["Tiny CUDA run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "Slurm job 689221" in rows["Tiny CUDA run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "Trainable parameters: 1,612,596 / 1,612,596" in rows["Tiny CUDA run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "doubleblock_ref_checkpoints/adapter" in rows["Tiny CUDA run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "loaded 19/19 teacher keys" in rows["Tiny CUDA run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "tiny_cuda_doubleblock_discrete_mamba2_ref/save" in rows["Tiny CUDA run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CUDA run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "num_steps == 2" in rows["Tiny CUDA run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "26410b59df6bd130e8b87ca6ce4f35a4040fb7dad89edc6e4392afc47c85eb0c" in rows["Tiny CUDA run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "optimizer state entries 19" in rows["Tiny CUDA run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "max diff 0.00019963830709457397" in rows["Tiny CUDA run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][2]
    assert "CUDA DoubleBlock-family and DDP/FSDP adapter coverage are tracked separately below" in rows["Tiny CUDA run.py DoubleBlockAdapter DiscreteMamba2 reference training smoke"][4]
    assert "tiny_cuda_doubleblock_vanilla_discrete_mamba2_ref_supervised.yaml" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "tiny_cuda_doubleblock_hymba_discrete_mamba2_ref_supervised.yaml" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "tiny_cuda_doubleblock_merger_discrete_mamba2_ref_supervised.yaml" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "LayeredMambaLM -> LlamaModel -> DoubleBlockAdapter" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "DoubleBlockVanilla" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "DoubleBlockHymba" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "DoubleBlockMerger" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "DiscreteMamba2 use_ref_impl: true" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "cuda:0" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "689221" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "689237" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "689254" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "689262" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "1,612,596" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "1,612,324" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "1,612,660" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "1,612,628" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "19/19" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "17/17" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "23/23" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "21/21" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "tiny_cuda_doubleblock_discrete_mamba2_ref/save" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "tiny_cuda_doubleblock_vanilla_discrete_mamba2_ref/save" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "tiny_cuda_doubleblock_hymba_discrete_mamba2_ref/save" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "tiny_cuda_doubleblock_merger_discrete_mamba2_ref/save" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "26410b59df6bd130e8b87ca6ce4f35a4040fb7dad89edc6e4392afc47c85eb0c" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "688309cc0e531b8335568d7bcc5739087fcbf9ad82729bea0efa5711b1465053" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "668075227edf9868305ccbbcfcd287f81a51c3325acc7bbc3eb25efe57aca8fd" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "282710e0946b6830ef1d3026e4a456d572e797c2742911053293df786dd85459" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "optimizer state entries were 19, 17, 23, and 21" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "Hymba changed 10/23 tensors" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "0.00019951441208831966" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "fast Mamba kernels" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][4]
    assert "DDP/FSDP family coverage is tracked separately below" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][4]
    assert "multi-node distributed hybrid training" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][4]
    assert "representative checkpoints" in rows["Tiny CUDA run.py DoubleBlock DiscreteMamba2 reference training smokes"][4]
    dist_row = rows["Tiny CUDA DDP/FSDP run.py DoubleBlock DiscreteMamba2 reference training smokes"]
    assert "tiny_cuda_ddp_doubleblock_discrete_mamba2_ref_supervised.yaml" in dist_row[2]
    assert "tiny_cuda_fsdp_doubleblock_discrete_mamba2_ref_supervised.yaml" in dist_row[2]
    assert "tiny_cuda_ddp_doubleblock_vanilla_discrete_mamba2_ref_supervised.yaml" in dist_row[2]
    assert "tiny_cuda_fsdp_doubleblock_vanilla_discrete_mamba2_ref_supervised.yaml" in dist_row[2]
    assert "tiny_cuda_ddp_doubleblock_hymba_discrete_mamba2_ref_supervised.yaml" in dist_row[2]
    assert "tiny_cuda_fsdp_doubleblock_hymba_discrete_mamba2_ref_supervised.yaml" in dist_row[2]
    assert "tiny_cuda_ddp_doubleblock_merger_discrete_mamba2_ref_supervised.yaml" in dist_row[2]
    assert "tiny_cuda_fsdp_doubleblock_merger_discrete_mamba2_ref_supervised.yaml" in dist_row[2]
    assert "LayeredMambaLM -> LlamaModel -> DoubleBlockAdapter" in dist_row[2]
    assert "DoubleBlockVanilla" in dist_row[2]
    assert "DoubleBlockHymba" in dist_row[2]
    assert "DoubleBlockMerger" in dist_row[2]
    assert "DiscreteMamba2 use_ref_impl: true" in dist_row[2]
    assert "configs/smoke/data_ddp" in dist_row[2]
    assert "effective_batch_size: 2" in dist_row[2]
    assert "n_tokens: 32" in dist_row[2]
    assert "689318" in dist_row[2]
    assert "689363" in dist_row[2]
    assert "689378" in dist_row[2]
    assert "689404" in dist_row[2]
    assert "689324" in dist_row[2]
    assert "689371" in dist_row[2]
    assert "689396" in dist_row[2]
    assert "689411" in dist_row[2]
    assert "WORLD_SIZE=2" in dist_row[2]
    assert "wrapper ddp" in dist_row[2]
    assert "NO_SHARD" in dist_row[2]
    assert "wrapper fsdp" in dist_row[2]
    assert "FULL_SHARD" in dist_row[2]
    assert "broadcast weights from rank 0" in dist_row[2]
    assert "Model dtype: torch.float32" in dist_row[2]
    assert "19/19" in dist_row[2]
    assert "17/17" in dist_row[2]
    assert "23/23" in dist_row[2]
    assert "21/21" in dist_row[2]
    assert "tiny_cuda_ddp_doubleblock*_discrete_mamba2_ref/save" in dist_row[2]
    assert "tiny_cuda_fsdp_doubleblock*_discrete_mamba2_ref/save" in dist_row[2]
    assert "86be1ef437abce55cc58f7bffd6f3fdeecab0639a4014967314935dc30041bf4" in dist_row[2]
    assert "ebd61d3b2f1d789631a25612b27c51ca0ed8a5ef0ab516e32aa54a85fde2df8a" in dist_row[2]
    assert "9ec928a4d5863363f05235ba63eb7a065f3152f4e5a0867f34feb85217614d98" in dist_row[2]
    assert "12a16f0b147becab00aa2e6e3a69895368f215bb3c871b0e903c21e4faf2f6e7" in dist_row[2]
    assert "Hymba changed 10/23 tensors" in dist_row[2]
    assert "0.00019961617363151163" in dist_row[2]
    assert "multi-node distributed hybrid training" in dist_row[4]
    assert "representative checkpoints" in dist_row[4]
    assert "tiny_cpu_doubleblock_discrete_mamba2_ref_base.yaml" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "DoubleBlockAdapter" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "DoubleBlockVanilla" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "DoubleBlockHymba" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "DoubleBlockMerger" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "DiscreteMamba2 use_ref_impl: true" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "mixer2" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "19/19" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "17/17" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "23/23" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "21/21" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "mohawk_tiny_doubleblock_vanilla_discrete_mamba2_ref_cpu_run/save" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "mohawk_tiny_doubleblock_hymba_discrete_mamba2_ref_cpu_run/save" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "mohawk_tiny_doubleblock_merger_discrete_mamba2_ref_cpu_run/save" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "fee4e7c14156ce0404efa63f631a89d6b1effb895f4f5c77e40696a4035ddc46" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "1c779255a7d2b4920e3c09d87f1e51efc6f7a388ff44c5bd4f513d0bebb648a7" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "36da178993d39692d8c1cba3163fdc78b0f6ad330c0bbd7fa6b5065297c8d449" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "56bb1d2f4226b9f1fa5ce9b945052204e4254a8bc9aeebdddae868e75525b3cc" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "Hymba changed 17/23 tensors" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "mixer1.conv1d.weight" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][2]
    assert "fast Mamba kernels" in rows["Tiny CPU run.py DoubleBlock DiscreteMamba2 reference training smokes"][4]
    assert "DoubleBlockAdapter" in rows["Tiny CPU run.py DoubleBlock attention-only training smokes"][2]
    assert "DoubleBlockVanilla" in rows["Tiny CPU run.py DoubleBlock attention-only training smokes"][2]
    assert "DoubleBlockHymba" in rows["Tiny CPU run.py DoubleBlock attention-only training smokes"][2]
    assert "DoubleBlockMerger" in rows["Tiny CPU run.py DoubleBlock attention-only training smokes"][2]
    assert "LlamaAttention used as both tiny mixers" in rows["Tiny CPU run.py DoubleBlock attention-only training smokes"][2]
    assert "18/18" in rows["Tiny CPU run.py DoubleBlock attention-only training smokes"][2]
    assert "16/16" in rows["Tiny CPU run.py DoubleBlock attention-only training smokes"][2]
    assert "22/22" in rows["Tiny CPU run.py DoubleBlock attention-only training smokes"][2]
    assert "20/20" in rows["Tiny CPU run.py DoubleBlock attention-only training smokes"][2]
    assert "mohawk_tiny_doubleblock_adapter_cpu_run/save" in rows["Tiny CPU run.py DoubleBlock attention-only training smokes"][2]
    assert "5d8be23f394221dfdb2a79e6921b0f9f32628de652764027e3ad158530cfc584" in rows["Tiny CPU run.py DoubleBlock attention-only training smokes"][2]
    assert "bounded SSM reference mixer coverage is tracked separately" in rows["Tiny CPU run.py DoubleBlock attention-only training smokes"][4]
    assert "fast kernels" in rows["Tiny CPU run.py DoubleBlock attention-only training smokes"][4]
    assert "EvalConfig -> Evaluator -> evals.eval_hstates.evaluator" in rows["Tiny CPU run.py eval_hstates callback smoke"][2]
    assert ".to(local_rank)" in rows["Tiny CPU run.py eval_hstates callback smoke"][2]
    assert "RuntimeError: No CUDA GPUs are available" in rows["Tiny CPU run.py eval_hstates callback smoke"][2]
    assert "get_device()" in rows["Tiny CPU run.py eval_hstates callback smoke"][2]
    assert "all_hidden_states" in rows["Tiny CPU run.py eval_hstates callback smoke"][2]
    assert "eval_score: 1.291738" in rows["Tiny CPU run.py eval_hstates callback smoke"][2]
    assert "mohawk_tiny_eval_hstates_cpu_run/save" in rows["Tiny CPU run.py eval_hstates callback smoke"][2]
    assert "b29506d2ca727b64122a5ad446bd78fb71bf5245b17a6e6cb979c1653fd86f26" in rows["Tiny CPU run.py eval_hstates callback smoke"][2]
    assert "does not validate representative hidden-state eval data" in rows["Tiny CPU run.py eval_hstates callback smoke"][4]
    assert "tiny_cpu_bfloat16_eval_hstates.yaml" in rows["Tiny CPU run.py bfloat16 eval_hstates callback smoke"][2]
    assert "TrainConfig.model_dtype: bfloat16" in rows["Tiny CPU run.py bfloat16 eval_hstates callback smoke"][2]
    assert "TeacherConfig.model_dtype: bfloat16" in rows["Tiny CPU run.py bfloat16 eval_hstates callback smoke"][2]
    assert "EvalConfig -> eval_hstates" in rows["Tiny CPU run.py bfloat16 eval_hstates callback smoke"][2]
    assert "EvalConfig[0].Evaluation eval_hstates" in rows["Tiny CPU run.py bfloat16 eval_hstates callback smoke"][2]
    assert "1.289062" in rows["Tiny CPU run.py bfloat16 eval_hstates callback smoke"][2]
    assert "latest_eval_hstates" in rows["Tiny CPU run.py bfloat16 eval_hstates callback smoke"][2]
    assert "Dataloader state is not implemented" in rows["Tiny CPU run.py bfloat16 eval_hstates callback smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py bfloat16 eval_hstates callback smoke"][2]
    assert "c8ab9d7d3fa494e07c5bfd35e7b663f2876772173d1e1d3c049008cb64450aaa" in rows["Tiny CPU run.py bfloat16 eval_hstates callback smoke"][2]
    assert "CUDA bfloat16" in rows["Tiny CPU run.py bfloat16 eval_hstates callback smoke"][4]
    assert "tiny_cuda_eval_hstates.yaml" in rows["Tiny CUDA run.py eval_hstates callback smoke"][2]
    assert "eval_score: 1.636902" in rows["Tiny CUDA run.py eval_hstates callback smoke"][2]
    assert "latest_eval_hstates.txt" in rows["Tiny CUDA run.py eval_hstates callback smoke"][2]
    assert "287b32e35195cb73a9a0f35493802b98f510bff5269543d6ca721750a8a2bb46" in rows["Tiny CUDA run.py eval_hstates callback smoke"][2]
    assert "representative hidden-state eval data" in rows["Tiny CUDA run.py eval_hstates callback smoke"][4]
    assert "tiny_cuda_ddp_eval_hstates.yaml" in rows["Tiny CUDA DDP run.py eval_hstates callback smoke"][2]
    assert "tiny_cuda_ddp_default_runtime_eval_hstates.yaml" in rows["Tiny CUDA DDP run.py eval_hstates callback smoke"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Eval-HStates-Smoke" in rows["Tiny CUDA DDP run.py eval_hstates callback smoke"][2]
    assert "Slurm job 691481" in rows["Tiny CUDA DDP run.py eval_hstates callback smoke"][2]
    assert "nvl72d122-T06.cm.cluster" in rows["Tiny CUDA DDP run.py eval_hstates callback smoke"][2]
    assert "wrappers ddp ddp" in rows["Tiny CUDA DDP run.py eval_hstates callback smoke"][2]
    assert "eval data configs/smoke/data_ddp" in rows["Tiny CUDA DDP run.py eval_hstates callback smoke"][2]
    assert "1.6627724170684814" in rows["Tiny CUDA DDP run.py eval_hstates callback smoke"][2]
    assert "latest_eval_hstates.txt" in rows["Tiny CUDA DDP run.py eval_hstates callback smoke"][2]
    assert "tiny_cuda_ddp_default_runtime_eval_hstates/save" in rows["Tiny CUDA DDP run.py eval_hstates callback smoke"][2]
    assert "01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in rows["Tiny CUDA DDP run.py eval_hstates callback smoke"][2]
    assert "tiny_cuda_fsdp_eval_hstates.yaml" in rows["Tiny CUDA FSDP run.py eval_hstates callback smoke"][2]
    assert "tiny_cuda_fsdp_default_runtime_eval_hstates.yaml" in rows["Tiny CUDA FSDP run.py eval_hstates callback smoke"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Eval-HStates-Smoke" in rows["Tiny CUDA FSDP run.py eval_hstates callback smoke"][2]
    assert "Slurm job 691627" in rows["Tiny CUDA FSDP run.py eval_hstates callback smoke"][2]
    assert "nvl72d112-T17.cm.cluster" in rows["Tiny CUDA FSDP run.py eval_hstates callback smoke"][2]
    assert "torch.inference_mode()" in rows["Tiny CUDA FSDP run.py eval_hstates callback smoke"][2]
    assert "Inplace update to inference tensor outside InferenceMode" in rows["Tiny CUDA FSDP run.py eval_hstates callback smoke"][2]
    assert "torch.no_grad()" in rows["Tiny CUDA FSDP run.py eval_hstates callback smoke"][2]
    assert "wrappers fsdp fsdp" in rows["Tiny CUDA FSDP run.py eval_hstates callback smoke"][2]
    assert "FULL_SHARD FULL_SHARD" in rows["Tiny CUDA FSDP run.py eval_hstates callback smoke"][2]
    assert "Model dtype: torch.float32" in rows["Tiny CUDA FSDP run.py eval_hstates callback smoke"][2]
    assert "latest_eval_hstates.txt" in rows["Tiny CUDA FSDP run.py eval_hstates callback smoke"][2]
    assert "tiny_cuda_fsdp_default_runtime_eval_hstates/save" in rows["Tiny CUDA FSDP run.py eval_hstates callback smoke"][2]
    assert "multi-node FSDP eval" in rows["Tiny CUDA FSDP run.py eval_hstates callback smoke"][4]
    assert "tiny_cuda_bfloat16_eval_hstates.yaml" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_hstates callback smokes"][2]
    assert "tiny_cuda_ddp_bfloat16_eval_hstates.yaml" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_hstates callback smokes"][2]
    assert "tiny_cuda_fsdp_bfloat16_eval_hstates.yaml" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_hstates callback smokes"][2]
    assert "EvalConfig -> eval_hstates" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_hstates callback smokes"][2]
    assert "latest_eval_hstates" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_hstates callback smokes"][2]
    assert "1.640625" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_hstates callback smokes"][2]
    assert "1.664062" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_hstates callback smokes"][2]
    assert "torch.bfloat16" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_hstates callback smokes"][2]
    assert "52fe58e628de69a168b351d41c8d01a52017300497a0675770ae75653ccf1ef1" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_hstates callback smokes"][2]
    assert "0bc9e79b22d4fce8a926de7260a144c0a5a0ff09a86328af5f35777ab655af15" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_hstates callback smokes"][2]
    assert "representative hidden-state eval data" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_hstates callback smokes"][4]
    assert "EvalConfig -> Evaluator -> evals.eval_ppl.evaluator" in rows["Tiny CPU run.py eval_ppl callback and latest-checkpoint smoke"][2]
    assert "save_latest" in rows["Tiny CPU run.py eval_ppl callback and latest-checkpoint smoke"][2]
    assert "eval_score: 46137.343750" in rows["Tiny CPU run.py eval_ppl callback and latest-checkpoint smoke"][2]
    assert "latest_Perplexity" in rows["Tiny CPU run.py eval_ppl callback and latest-checkpoint smoke"][2]
    assert "NotImplementedError" in rows["Tiny CPU run.py eval_ppl callback and latest-checkpoint smoke"][2]
    assert "dataloader_state_dict.pth" in rows["Tiny CPU run.py eval_ppl callback and latest-checkpoint smoke"][2]
    assert "46137.34375" in rows["Tiny CPU run.py eval_ppl callback and latest-checkpoint smoke"][2]
    assert "ada2086c37b7c885e663df2589d4aa4a318178e2163906772a2efd4eaa8c36b0" in rows["Tiny CPU run.py eval_ppl callback and latest-checkpoint smoke"][2]
    assert "non-resumable local JSON training loader" in rows["Tiny CPU run.py eval_ppl callback and latest-checkpoint smoke"][4]
    assert "public HFDataset latest-state smoke" in rows["Tiny CPU run.py eval_ppl callback and latest-checkpoint smoke"][4]
    assert "tiny_cuda_eval_ppl.yaml" in rows["Tiny CUDA run.py eval_ppl callback smoke"][2]
    assert "cuda:0" in rows["Tiny CUDA run.py eval_ppl callback smoke"][2]
    assert "EvalConfig[0].Evaluation eval_ppl" in rows["Tiny CUDA run.py eval_ppl callback smoke"][2]
    assert "46136.37890625" in rows["Tiny CUDA run.py eval_ppl callback smoke"][2]
    assert "latest_Perplexity.txt" in rows["Tiny CUDA run.py eval_ppl callback smoke"][2]
    assert "torch.float32" in rows["Tiny CUDA run.py eval_ppl callback smoke"][2]
    assert "287b32e35195cb73a9a0f35493802b98f510bff5269543d6ca721750a8a2bb46" in rows["Tiny CUDA run.py eval_ppl callback smoke"][2]
    assert "single-process CUDA wrapper-driven eval_ppl" in rows["Tiny CUDA run.py eval_ppl callback smoke"][4]
    assert "tiny_cuda_ddp_eval_ppl.yaml" in rows["Tiny CUDA DDP run.py eval_ppl callback smoke"][2]
    assert "tiny_cuda_ddp_default_runtime_eval_ppl.yaml" in rows["Tiny CUDA DDP run.py eval_ppl callback smoke"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Eval-PPL-Smoke" in rows["Tiny CUDA DDP run.py eval_ppl callback smoke"][2]
    assert "Slurm job 691806" in rows["Tiny CUDA DDP run.py eval_ppl callback smoke"][2]
    assert "nvl72d062-T10.cm.cluster" in rows["Tiny CUDA DDP run.py eval_ppl callback smoke"][2]
    assert "rank 1 had no eval batch" in rows["Tiny CUDA DDP run.py eval_ppl callback smoke"][2]
    assert "configs/smoke/data_ddp" in rows["Tiny CUDA DDP run.py eval_ppl callback smoke"][2]
    assert "WORLD_SIZE=2" in rows["Tiny CUDA DDP run.py eval_ppl callback smoke"][2]
    assert "wrapper_type: ddp" in rows["Tiny CUDA DDP run.py eval_ppl callback smoke"][2]
    assert "53548.18359375" in rows["Tiny CUDA DDP run.py eval_ppl callback smoke"][2]
    assert "latest_Perplexity.txt" in rows["Tiny CUDA DDP run.py eval_ppl callback smoke"][2]
    assert "tiny_cuda_ddp_default_runtime_eval_ppl/save" in rows["Tiny CUDA DDP run.py eval_ppl callback smoke"][2]
    assert "01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in rows["Tiny CUDA DDP run.py eval_ppl callback smoke"][2]
    assert "two-rank DDP wrapper-driven eval_ppl" in rows["Tiny CUDA DDP run.py eval_ppl callback smoke"][4]
    assert "tiny_cuda_fsdp_eval_ppl.yaml" in rows["Tiny CUDA FSDP run.py eval_ppl callback smoke"][2]
    assert "tiny_cuda_fsdp_default_runtime_eval_ppl.yaml" in rows["Tiny CUDA FSDP run.py eval_ppl callback smoke"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Eval-PPL-Smoke" in rows["Tiny CUDA FSDP run.py eval_ppl callback smoke"][2]
    assert "Slurm job 691852" in rows["Tiny CUDA FSDP run.py eval_ppl callback smoke"][2]
    assert "nvl72d029-T13.cm.cluster" in rows["Tiny CUDA FSDP run.py eval_ppl callback smoke"][2]
    assert "FULL_SHARD FULL_SHARD" in rows["Tiny CUDA FSDP run.py eval_ppl callback smoke"][2]
    assert "Inplace update to inference tensor outside InferenceMode" in rows["Tiny CUDA FSDP run.py eval_ppl callback smoke"][2]
    assert "save_weights() on all ranks" in rows["Tiny CUDA FSDP run.py eval_ppl callback smoke"][2]
    assert "torch.no_grad()" in rows["Tiny CUDA FSDP run.py eval_ppl callback smoke"][2]
    assert "Model dtype: torch.float32" in rows["Tiny CUDA FSDP run.py eval_ppl callback smoke"][2]
    assert "53548.18359375" in rows["Tiny CUDA FSDP run.py eval_ppl callback smoke"][2]
    assert "latest_Perplexity.txt" in rows["Tiny CUDA FSDP run.py eval_ppl callback smoke"][2]
    assert "tiny_cuda_fsdp_default_runtime_eval_ppl/save" in rows["Tiny CUDA FSDP run.py eval_ppl callback smoke"][2]
    assert "01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in rows["Tiny CUDA FSDP run.py eval_ppl callback smoke"][2]
    assert "two-rank FSDP FULL_SHARD wrapper-driven eval_ppl" in rows["Tiny CUDA FSDP run.py eval_ppl callback smoke"][4]
    assert "tiny_cuda_bfloat16_eval_ppl.yaml" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_ppl callback smokes"][2]
    assert "tiny_cuda_ddp_bfloat16_eval_ppl.yaml" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_ppl callback smokes"][2]
    assert "tiny_cuda_fsdp_bfloat16_eval_ppl.yaml" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_ppl callback smokes"][2]
    assert "TrainConfig.model_dtype: bfloat16" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_ppl callback smokes"][2]
    assert "TeacherConfig.model_dtype: bfloat16" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_ppl callback smokes"][2]
    assert "latest_Perplexity" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_ppl callback smokes"][2]
    assert "49020.808594" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_ppl callback smokes"][2]
    assert "55746.601562" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_ppl callback smokes"][2]
    assert "torch.bfloat16" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_ppl callback smokes"][2]
    assert "52fe58e628de69a168b351d41c8d01a52017300497a0675770ae75653ccf1ef1" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_ppl callback smokes"][2]
    assert "0bc9e79b22d4fce8a926de7260a144c0a5a0ff09a86328af5f35777ab655af15" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_ppl callback smokes"][2]
    assert "representative eval data" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 eval_ppl callback smokes"][4]
    assert "tiny_cpu_eval_ppl_frequency.yaml" in rows["Tiny CPU run.py eval_ppl frequency callback smoke"][2]
    assert "frequency: 1" in rows["Tiny CPU run.py eval_ppl frequency callback smoke"][2]
    assert "eval_at_start: false" in rows["Tiny CPU run.py eval_ppl frequency callback smoke"][2]
    assert "eval_at_end: false" in rows["Tiny CPU run.py eval_ppl frequency callback smoke"][2]
    assert "Evaluator.is_my_turn" in rows["Tiny CPU run.py eval_ppl frequency callback smoke"][2]
    assert "latest_Perplexity" in rows["Tiny CPU run.py eval_ppl frequency callback smoke"][2]
    assert "46137.34375" in rows["Tiny CPU run.py eval_ppl frequency callback smoke"][2]
    assert "ada2086c37b7c885e663df2589d4aa4a318178e2163906772a2efd4eaa8c36b0" in rows["Tiny CPU run.py eval_ppl frequency callback smoke"][2]
    assert "periodic evaluator scheduling" in rows["Tiny CPU run.py eval_ppl frequency callback smoke"][4]
    assert "tiny_cpu_eval_ppl_save_best.yaml" in rows["Tiny CPU run.py eval_ppl save_best checkpoint smoke"][2]
    assert "save_best" in rows["Tiny CPU run.py eval_ppl save_best checkpoint smoke"][2]
    assert "did not log Saving best" in rows["Tiny CPU run.py eval_ppl save_best checkpoint smoke"][2]
    assert "best_Perplexity" in rows["Tiny CPU run.py eval_ppl save_best checkpoint smoke"][2]
    assert "current_best is None" in rows["Tiny CPU run.py eval_ppl save_best checkpoint smoke"][2]
    assert "best_Perplexity.txt" in rows["Tiny CPU run.py eval_ppl save_best checkpoint smoke"][2]
    assert "46137.34375" in rows["Tiny CPU run.py eval_ppl save_best checkpoint smoke"][2]
    assert "ada2086c37b7c885e663df2589d4aa4a318178e2163906772a2efd4eaa8c36b0" in rows["Tiny CPU run.py eval_ppl save_best checkpoint smoke"][2]
    assert "non-resumable local JSON training loader" in rows["Tiny CPU run.py eval_ppl save_best checkpoint smoke"][4]
    assert "tiny_cpu_eval_multi.yaml" in rows["Tiny CPU run.py multi-evaluator callback smoke"][2]
    assert "two real evaluator objects" in rows["Tiny CPU run.py multi-evaluator callback smoke"][2]
    assert "eval_ppl" in rows["Tiny CPU run.py multi-evaluator callback smoke"][2]
    assert "benchmark" in rows["Tiny CPU run.py multi-evaluator callback smoke"][2]
    assert "latest_Perplexity" in rows["Tiny CPU run.py multi-evaluator callback smoke"][2]
    assert "wikitext 206549.081664" in rows["Tiny CPU run.py multi-evaluator callback smoke"][2]
    assert "mohawk_tiny_eval_multi_cpu_run/save" in rows["Tiny CPU run.py multi-evaluator callback smoke"][2]
    assert "ada2086c37b7c885e663df2589d4aa4a318178e2163906772a2efd4eaa8c36b0" in rows["Tiny CPU run.py multi-evaluator callback smoke"][2]
    assert "cached task data" in rows["Tiny CPU run.py multi-evaluator callback smoke"][4]
    assert "tiny_cpu_eval_ppl_hfdata_save_latest.yaml" in rows["Tiny CPU run.py eval_ppl public HFDataset latest-state smoke"][2]
    assert "HFDataset -> Tokenize -> PaddingDataLoader -> CycleDataLoader -> TorchDataLoader -> run.py" in rows["Tiny CPU run.py eval_ppl public HFDataset latest-state smoke"][2]
    assert "HF_DATASETS_OFFLINE=1" in rows["Tiny CPU run.py eval_ppl public HFDataset latest-state smoke"][2]
    assert "29/29" in rows["Tiny CPU run.py eval_ppl public HFDataset latest-state smoke"][2]
    assert "eval_score: 46154.773438" in rows["Tiny CPU run.py eval_ppl public HFDataset latest-state smoke"][2]
    assert "dataloader_state_dict.pth" in rows["Tiny CPU run.py eval_ppl public HFDataset latest-state smoke"][2]
    assert "'_index': 2" in rows["Tiny CPU run.py eval_ppl public HFDataset latest-state smoke"][2]
    assert "110821dc9516f4eabc3c66ea9fb20697e40803f889786bc257f4deed9e1b732f" in rows["Tiny CPU run.py eval_ppl public HFDataset latest-state smoke"][2]
    assert "distributed sharding" in rows["Tiny CPU run.py eval_ppl public HFDataset latest-state smoke"][4]
    assert "tiny_cpu_bfloat16_eval_ppl_hfdata_save_latest.yaml" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset latest-state smoke"][2]
    assert "HFDataset -> Tokenize -> PaddingDataLoader -> CycleDataLoader -> TorchDataLoader -> run.py" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset latest-state smoke"][2]
    assert "TrainConfig.model_dtype: bfloat16" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset latest-state smoke"][2]
    assert "TeacherConfig.model_dtype: bfloat16" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset latest-state smoke"][2]
    assert "29/29" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset latest-state smoke"][2]
    assert "49020.812500" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset latest-state smoke"][2]
    assert "dataloader_state_dict.pth" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset latest-state smoke"][2]
    assert "'_index': 2" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset latest-state smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset latest-state smoke"][2]
    assert "d914f7dc45244954594966c9e3bff9884f82d3947c703513c16014f2fc6d40c5" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset latest-state smoke"][2]
    assert "CUDA bfloat16" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset latest-state smoke"][4]
    assert "tiny_cpu_bfloat16_eval_ppl_hfdata_save_best.yaml" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset best-state smoke"][2]
    assert "HFDataset -> Tokenize -> PaddingDataLoader -> CycleDataLoader -> TorchDataLoader -> run.py" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset best-state smoke"][2]
    assert "TrainConfig.model_dtype: bfloat16" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset best-state smoke"][2]
    assert "TeacherConfig.model_dtype: bfloat16" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset best-state smoke"][2]
    assert "save_best: true" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset best-state smoke"][2]
    assert "save_best True" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset best-state smoke"][2]
    assert "save_latest False" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset best-state smoke"][2]
    assert "29/29" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset best-state smoke"][2]
    assert "49020.812500" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset best-state smoke"][2]
    assert "best_Perplexity" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset best-state smoke"][2]
    assert "dataloader_state_dict.pth" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset best-state smoke"][2]
    assert "'_index': 2" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset best-state smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset best-state smoke"][2]
    assert "d914f7dc45244954594966c9e3bff9884f82d3947c703513c16014f2fc6d40c5" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset best-state smoke"][2]
    assert "CUDA bfloat16" in rows["Tiny CPU run.py bfloat16 eval_ppl public HFDataset best-state smoke"][4]
    assert "tiny_cpu_eval_ppl_hfdata_save_best.yaml" in rows["Tiny CPU run.py eval_ppl public HFDataset best-state smoke"][2]
    assert "HFDataset -> Tokenize -> PaddingDataLoader -> CycleDataLoader -> TorchDataLoader -> run.py" in rows["Tiny CPU run.py eval_ppl public HFDataset best-state smoke"][2]
    assert "save_best: true" in rows["Tiny CPU run.py eval_ppl public HFDataset best-state smoke"][2]
    assert "29/29" in rows["Tiny CPU run.py eval_ppl public HFDataset best-state smoke"][2]
    assert "eval_score: 46154.773438" in rows["Tiny CPU run.py eval_ppl public HFDataset best-state smoke"][2]
    assert "best_Perplexity" in rows["Tiny CPU run.py eval_ppl public HFDataset best-state smoke"][2]
    assert "dataloader_state_dict.pth" in rows["Tiny CPU run.py eval_ppl public HFDataset best-state smoke"][2]
    assert "'_index': 2" in rows["Tiny CPU run.py eval_ppl public HFDataset best-state smoke"][2]
    assert "110821dc9516f4eabc3c66ea9fb20697e40803f889786bc257f4deed9e1b732f" in rows["Tiny CPU run.py eval_ppl public HFDataset best-state smoke"][2]
    assert "distributed sharding" in rows["Tiny CPU run.py eval_ppl public HFDataset best-state smoke"][4]
    assert "tiny_cpu_c4_streaming_supervised.yaml" in rows["Tiny CPU run.py public C4 streaming training smoke"][2]
    assert "HFDataset -> Tokenize -> PaddingDataLoader -> CycleDataLoader -> TorchDataLoader -> run.py" in rows["Tiny CPU run.py public C4 streaming training smoke"][2]
    assert "allenai/c4" in rows["Tiny CPU run.py public C4 streaming training smoke"][2]
    assert "en.noclean" in rows["Tiny CPU run.py public C4 streaming training smoke"][2]
    assert "streaming: true" in rows["Tiny CPU run.py public C4 streaming training smoke"][2]
    assert "shuffle_buffer_size: 2" in rows["Tiny CPU run.py public C4 streaming training smoke"][2]
    assert "Hugging Face DNS" in rows["Tiny CPU run.py public C4 streaming training smoke"][2]
    assert "12/12 teacher keys" in rows["Tiny CPU run.py public C4 streaming training smoke"][2]
    assert "mohawk_tiny_c4_streaming_cpu_run/save" in rows["Tiny CPU run.py public C4 streaming training smoke"][2]
    assert "_step_count == 2" in rows["Tiny CPU run.py public C4 streaming training smoke"][2]
    assert "dataloader_state_dict.pth" in rows["Tiny CPU run.py public C4 streaming training smoke"][2]
    assert "c5f9a133cbcd343d88c39edcf9b35bf83f46315f76ea3cf44c9d25257dc849dc" in rows["Tiny CPU run.py public C4 streaming training smoke"][2]
    assert "default-buffer streaming throughput" in rows["Tiny CPU run.py public C4 streaming training smoke"][4]
    assert "tiny_cpu_fineweb_edu_streaming_supervised.yaml" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "HuggingFaceFW/fineweb-edu" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "sample-100BT" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "streaming: true" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "shuffle_buffer_size: 2" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "signal 139" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "PyGILState_Release" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "71b740e0d4bee1b557cb83ce2e0db4a8c1352adfed7148b369edced4956f0b3e" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "429 Too Many Requests" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "no-shuffle minimal read" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "PyTorch's internal _dataset_fetcher.dataset_iter" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "tiny_cpu_fineweb_edu_streaming_supervised_default_runtime.yaml" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "exited with status 0" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "9e447f7af4d9d10fbd0bab04b7078d39bc6239f1681a73cdebc2c8c7762676d2" in rows["Tiny CPU run.py public FineWeb-Edu streaming training smoke"][2]
    assert "tiny_cpu_random_data_supervised.yaml" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "RandomDataLoader.local_files_only" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "RandomDataLoader -> CycleDataLoader -> run.py" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "TorchDataLoader is deliberately not in this loader chain" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "max_seq_len: 8" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "RandomDataLoader.data_cfg.train.n_tokens: 16" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "batch1_shape (1, 8)" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "batch2_shape (1, 8)" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "batch3_shape (1, 8)" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "batch_dtype torch.int64" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "[32487, 27591, 7093, 953, 10379, 5139, 38222, 10161]" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "[28812, 40410, 8215, 14673, 11984, 16660, 49908, 34954]" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "[35078, 11511, 47418, 15036, 44506, 263, 20248, 44628]" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "different_batches True" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "cycled_after_two True" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "num_samples 2" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "12/12 local teacher keys" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "mohawk_tiny_random_data_cpu_run/save" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "optimizer_groups 1" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "5f5f7f11632e49ebdeddde02acffca65dd5c876cfb8cf13e8224f2d6d666d9d9" in rows["Tiny CPU run.py RandomDataLoader training smoke"][2]
    assert "synthetic random-token batches" in rows["Tiny CPU run.py RandomDataLoader training smoke"][4]
    assert "real data loading" in rows["Tiny CPU run.py RandomDataLoader training smoke"][4]
    assert "TorchDataLoader wrapping" in rows["Tiny CPU run.py RandomDataLoader training smoke"][4]
    assert "tiny_cpu_shuffle_loader_supervised.yaml" in rows["Tiny CPU run.py ShuffleLoader training smoke"][2]
    assert "tiny_shuffle_source_a.yaml" in rows["Tiny CPU run.py ShuffleLoader training smoke"][2]
    assert "tiny_shuffle_source_b.yaml" in rows["Tiny CPU run.py ShuffleLoader training smoke"][2]
    assert "data_shuffle_a/tiny_shuffle_a.json" in rows["Tiny CPU run.py ShuffleLoader training smoke"][2]
    assert "data_shuffle_b/tiny_shuffle_b.json" in rows["Tiny CPU run.py ShuffleLoader training smoke"][2]
    assert "ShuffleLoader -> CycleDataLoader -> run.py" in rows["Tiny CPU run.py ShuffleLoader training smoke"][2]
    assert "JSONIterableDataset -> Tokenize(collate_type=text) -> PaddingDataLoader -> TorchDataLoader" in rows["Tiny CPU run.py ShuffleLoader training smoke"][2]
    assert "shapes [(1, 8), (1, 8), (1, 8), (1, 8)]" in rows["Tiny CPU run.py ShuffleLoader training smoke"][2]
    assert "dtype torch.int64" in rows["Tiny CPU run.py ShuffleLoader training smoke"][2]
    assert "[1477, 18137, 17130, 285, 1219, 19301, 3047, 6291]" in rows["Tiny CPU run.py ShuffleLoader training smoke"][2]
    assert "[1477, 18137, 12159, 285, 1219, 19301, 3047, 6291]" in rows["Tiny CPU run.py ShuffleLoader training smoke"][2]
    assert "unique_first_cycle 2" in rows["Tiny CPU run.py ShuffleLoader training smoke"][2]
    assert "third_equals_second True" in rows["Tiny CPU run.py ShuffleLoader training smoke"][2]
    assert "12/12 local teacher keys" in rows["Tiny CPU run.py ShuffleLoader training smoke"][2]
    assert "mohawk_tiny_shuffle_loader_cpu_run/save" in rows["Tiny CPU run.py ShuffleLoader training smoke"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CPU run.py ShuffleLoader training smoke"][2]
    assert "optimizer_groups 1" in rows["Tiny CPU run.py ShuffleLoader training smoke"][2]
    assert "ff21b755d76cb2a337aab8e08443b0c98fa04aed66bdd30fd0456a6a30eb6e30" in rows["Tiny CPU run.py ShuffleLoader training smoke"][2]
    assert "large shuffle pools" in rows["Tiny CPU run.py ShuffleLoader training smoke"][4]
    assert "source weighting" in rows["Tiny CPU run.py ShuffleLoader training smoke"][4]
    assert "dataloader-state resume" in rows["Tiny CPU run.py ShuffleLoader training smoke"][4]
    assert "tiny_cpu_conversation_collate.yaml" in rows["Tiny CPU run.py conversation-collate training smoke"][2]
    assert "data_conversation/tiny_conversation.json" in rows["Tiny CPU run.py conversation-collate training smoke"][2]
    assert "tiny_chat_template.jinja" in rows["Tiny CPU run.py conversation-collate training smoke"][2]
    assert "JSONIterableDataset -> Tokenize(collate_type=conversation) -> PaddingDataLoader -> CycleDataLoader -> TorchDataLoader -> run.py" in rows["Tiny CPU run.py conversation-collate training smoke"][2]
    assert "torch.int64" in rows["Tiny CPU run.py conversation-collate training smoke"][2]
    assert "12/12 local teacher keys" in rows["Tiny CPU run.py conversation-collate training smoke"][2]
    assert "mohawk_tiny_conversation_collate_cpu_run/save" in rows["Tiny CPU run.py conversation-collate training smoke"][2]
    assert "batch_shape (1, 8)" in rows["Tiny CPU run.py conversation-collate training smoke"][2]
    assert "[7220, 25, 13816, 23748, 13, 198, 562, 10167]" in rows["Tiny CPU run.py conversation-collate training smoke"][2]
    assert "collate_type conversation" in rows["Tiny CPU run.py conversation-collate training smoke"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CPU run.py conversation-collate training smoke"][2]
    assert "30a57cc4eff783c7161192d04b48898326f3e3738ab83440b482182355f22624" in rows["Tiny CPU run.py conversation-collate training smoke"][2]
    assert "large remote conversation datasets" in rows["Tiny CPU run.py conversation-collate training smoke"][4]
    assert "chat-tokenizer default templates" in rows["Tiny CPU run.py conversation-collate training smoke"][4]
    assert "tiny_cpu_classic_collate.yaml" in rows["Tiny CPU run.py classic-collate training smoke"][2]
    assert "data_classic/tiny_classic.json" in rows["Tiny CPU run.py classic-collate training smoke"][2]
    assert "input_ids" in rows["Tiny CPU run.py classic-collate training smoke"][2]
    assert "JSONIterableDataset -> Tokenize(collate_type=classic) -> PaddingDataLoader -> CycleDataLoader -> TorchDataLoader -> run.py" in rows["Tiny CPU run.py classic-collate training smoke"][2]
    assert "torch.int64" in rows["Tiny CPU run.py classic-collate training smoke"][2]
    assert "12/12 local teacher keys" in rows["Tiny CPU run.py classic-collate training smoke"][2]
    assert "mohawk_tiny_classic_collate_cpu_run/save" in rows["Tiny CPU run.py classic-collate training smoke"][2]
    assert "batch_shape (1, 8)" in rows["Tiny CPU run.py classic-collate training smoke"][2]
    assert "[7220, 25, 13816, 23748, 13, 198, 50256, 50256]" in rows["Tiny CPU run.py classic-collate training smoke"][2]
    assert "collate_type classic" in rows["Tiny CPU run.py classic-collate training smoke"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CPU run.py classic-collate training smoke"][2]
    assert "22c14fb304146274b424342dbace0e6a3691cf5924cbaf1ccd5c422a17614123" in rows["Tiny CPU run.py classic-collate training smoke"][2]
    assert "jondurbin/airoboros-2.2" in rows["Tiny CPU run.py classic-collate training smoke"][4]
    assert "pre-tokenized dataset schemas" in rows["Tiny CPU run.py classic-collate training smoke"][4]
    assert "tiny_cpu_kv_raw_packing.yaml" in rows["Tiny CPU run.py KV raw-packing training smoke"][2]
    assert "RoundRobinLoader -> KVRetrieval -> Tokenize(collate_type=raw) -> PackingDataLoader -> CycleDataLoader -> TorchDataLoader -> run.py" in rows["Tiny CPU run.py KV raw-packing training smoke"][2]
    assert "num_pairs: 1" in rows["Tiny CPU run.py KV raw-packing training smoke"][2]
    assert "seed: 0" in rows["Tiny CPU run.py KV raw-packing training smoke"][2]
    assert "batch_keys ['attention_mask', 'input_ids', 'position_ids']" in rows["Tiny CPU run.py KV raw-packing training smoke"][2]
    assert "input_shape (1, 8)" in rows["Tiny CPU run.py KV raw-packing training smoke"][2]
    assert "position_shape (1, 8)" in rows["Tiny CPU run.py KV raw-packing training smoke"][2]
    assert "attention_shape (1, 8)" in rows["Tiny CPU run.py KV raw-packing training smoke"][2]
    assert "input_dtype torch.int64" in rows["Tiny CPU run.py KV raw-packing training smoke"][2]
    assert "[13579, 273, 1096, 262, 1708, 22155, 25, 198]" in rows["Tiny CPU run.py KV raw-packing training smoke"][2]
    assert "position_ids [0, 1, 2, 3, 4, 5, 6, 7]" in rows["Tiny CPU run.py KV raw-packing training smoke"][2]
    assert "attention_sum 8" in rows["Tiny CPU run.py KV raw-packing training smoke"][2]
    assert "12/12 local teacher keys" in rows["Tiny CPU run.py KV raw-packing training smoke"][2]
    assert "mohawk_tiny_kv_raw_packing_cpu_run/save" in rows["Tiny CPU run.py KV raw-packing training smoke"][2]
    assert "nested_collate_type raw" in rows["Tiny CPU run.py KV raw-packing training smoke"][2]
    assert "packing_max_seq_len 8" in rows["Tiny CPU run.py KV raw-packing training smoke"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CPU run.py KV raw-packing training smoke"][2]
    assert "5f00fc0d077c74b11caa844cfdbae21102f95cf743973b3bc7cf1000115d157d" in rows["Tiny CPU run.py KV raw-packing training smoke"][2]
    assert "large KV runs" in rows["Tiny CPU run.py KV raw-packing training smoke"][4]
    assert "mixed real-data round-robin ratios" in rows["Tiny CPU run.py KV raw-packing training smoke"][4]
    assert "long-context packing" in rows["Tiny CPU run.py KV raw-packing training smoke"][4]
    assert "tiny_cpu_niah_dataset_supervised.yaml" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][2]
    assert "NeedleInHaystackDataset -> CycleDataLoader -> TorchDataLoader -> run.py" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][2]
    assert "NeedleInHaystackDataset.local_files_only" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][2]
    assert "local_files_only: true" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][2]
    assert "type_haystack: none" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][2]
    assert "batch_keys ['input_ids', 'response_ids']" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][2]
    assert "input_shape (1, 96)" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][2]
    assert "response_shape (1, 3)" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][2]
    assert "input_dtype torch.int64" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][2]
    assert "response_dtype torch.int64" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][2]
    assert "[32, 2041, 5536, 1271, 318, 7104, 1626, 262, 1708, 2420, 13, 6889]" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][2]
    assert "response tokens [2327, 21495, 1959]" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][2]
    assert "max_seq_len 96" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][2]
    assert "pad_count 23" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][2]
    assert "12/12 local teacher keys" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][2]
    assert "mohawk_tiny_niah_dataset_cpu_run/save" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][2]
    assert "train_n_tokens 192" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][2]
    assert "b229762d8c4f7555f8a640fd9fdd5cc186e18ab3366f76eaf077f89f7690ec1a" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][2]
    assert "essay haystacks" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][4]
    assert "long-context needle retrieval" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][4]
    assert "multi-query needles" in rows["Tiny CPU run.py NeedleInHaystackDataset training smoke"][4]
    assert "tiny_cpu_copying_task_supervised.yaml" in rows["Tiny CPU run.py CopyingTaskDataset training smoke"][2]
    assert "CopyingTaskDataset.local_files_only" in rows["Tiny CPU run.py CopyingTaskDataset training smoke"][2]
    assert "CopyingTaskDataset -> CycleDataLoader -> TorchDataLoader -> run.py" in rows["Tiny CPU run.py CopyingTaskDataset training smoke"][2]
    assert "local_files_only: true" in rows["Tiny CPU run.py CopyingTaskDataset training smoke"][2]
    assert "max_seq_len: 24" in rows["Tiny CPU run.py CopyingTaskDataset training smoke"][2]
    assert "batch_keys ['input_ids']" in rows["Tiny CPU run.py CopyingTaskDataset training smoke"][2]
    assert "batch_shape (1, 24)" in rows["Tiny CPU run.py CopyingTaskDataset training smoke"][2]
    assert "batch_dtype torch.int64" in rows["Tiny CPU run.py CopyingTaskDataset training smoke"][2]
    assert "[6310, 2762, 25, 11436, 262, 1708, 3146, 13, 198, 21947, 25, 718, 11, 21, 198, 31077, 25, 718, 11, 21, 50256, 50256, 50256, 50256]" in rows["Tiny CPU run.py CopyingTaskDataset training smoke"][2]
    assert "pad_count 4" in rows["Tiny CPU run.py CopyingTaskDataset training smoke"][2]
    assert "12/12 local teacher keys" in rows["Tiny CPU run.py CopyingTaskDataset training smoke"][2]
    assert "mohawk_tiny_copying_task_cpu_run/save" in rows["Tiny CPU run.py CopyingTaskDataset training smoke"][2]
    assert "train_n_tokens 48" in rows["Tiny CPU run.py CopyingTaskDataset training smoke"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CPU run.py CopyingTaskDataset training smoke"][2]
    assert "optimizer_groups 1" in rows["Tiny CPU run.py CopyingTaskDataset training smoke"][2]
    assert "411990a52399afe4302955bf8972e42471f5ee6e0169ed039e38a573a19100c9" in rows["Tiny CPU run.py CopyingTaskDataset training smoke"][2]
    assert "long copying contexts" in rows["Tiny CPU run.py CopyingTaskDataset training smoke"][4]
    assert "richer copying distributions" in rows["Tiny CPU run.py CopyingTaskDataset training smoke"][4]
    assert "tiny_cpu_sequential_loader_supervised.yaml" in rows["Tiny CPU run.py SequentialLoader training smoke"][2]
    assert "data_sequence_a/tiny_sequence_a.json" in rows["Tiny CPU run.py SequentialLoader training smoke"][2]
    assert "data_sequence_b/tiny_sequence_b.json" in rows["Tiny CPU run.py SequentialLoader training smoke"][2]
    assert "SequentialLoader -> PaddingDataLoader -> CycleDataLoader -> TorchDataLoader -> run.py" in rows["Tiny CPU run.py SequentialLoader training smoke"][2]
    assert "JSONIterableDataset -> Tokenize(collate_type=text)" in rows["Tiny CPU run.py SequentialLoader training smoke"][2]
    assert "max_samples: 1" in rows["Tiny CPU run.py SequentialLoader training smoke"][2]
    assert "batch1_shape (1, 8)" in rows["Tiny CPU run.py SequentialLoader training smoke"][2]
    assert "batch2_shape (1, 8)" in rows["Tiny CPU run.py SequentialLoader training smoke"][2]
    assert "batch_dtype torch.int64" in rows["Tiny CPU run.py SequentialLoader training smoke"][2]
    assert "[26591, 285, 1219, 19301, 8379, 50256, 50256, 50256]" in rows["Tiny CPU run.py SequentialLoader training smoke"][2]
    assert "[31361, 285, 1219, 19301, 8379, 50256, 50256, 50256]" in rows["Tiny CPU run.py SequentialLoader training smoke"][2]
    assert "different_batches True" in rows["Tiny CPU run.py SequentialLoader training smoke"][2]
    assert "dataloader 0 and 1" in rows["Tiny CPU run.py SequentialLoader training smoke"][2]
    assert "12/12 local teacher keys" in rows["Tiny CPU run.py SequentialLoader training smoke"][2]
    assert "mohawk_tiny_sequential_loader_cpu_run/save" in rows["Tiny CPU run.py SequentialLoader training smoke"][2]
    assert "local_files_only [True, True]" in rows["Tiny CPU run.py SequentialLoader training smoke"][2]
    assert "padding_max_seq_len 8" in rows["Tiny CPU run.py SequentialLoader training smoke"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CPU run.py SequentialLoader training smoke"][2]
    assert "d1e60333ba54ab9d2763b4c7e06700b24f438601aa3bdc2cd151bc6e97e75fa7" in rows["Tiny CPU run.py SequentialLoader training smoke"][2]
    assert "large multi-stage sequential curricula" in rows["Tiny CPU run.py SequentialLoader training smoke"][4]
    assert "remote datasets" in rows["Tiny CPU run.py SequentialLoader training smoke"][4]
    assert "long-context packing" in rows["Tiny CPU run.py SequentialLoader training smoke"][4]
    assert "tiny_cpu_aggregation_loader_supervised.yaml" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "data_aggregation_a/tiny_aggregation_a.json" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "data_aggregation_b/tiny_aggregation_b.json" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "RoundRobinLoader -> PackingDataLoader -> AggregationDataLoader -> CycleDataLoader -> TorchDataLoader -> run.py" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "JSONIterableDataset -> Tokenize(collate_type=text)" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "aggregation_size: 2" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "batch1_input_shape (1, 8)" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "batch2_input_shape (1, 8)" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "batch1_position_shape (1, 8)" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "batch2_position_shape (1, 8)" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "batch_dtype torch.int64" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "[9460, 43068, 17130, 285, 1219, 19301, 3047, 6291]" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "[9460, 43068, 12159, 285, 1219, 19301, 3047, 6291]" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "batch1_attention_sum 8" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "batch2_attention_sum 8" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "different_batches True" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "Aggregating 2 samples" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "12/12 local teacher keys" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "mohawk_tiny_aggregation_loader_cpu_run/save" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "packing_max_seq_len 8" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "55cb16302e332da1d06c9f1873af1d0c6bea4d48d13a0a082d4a6d389fb4d03d" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][2]
    assert "large aggregation buffers" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][4]
    assert "remote multi-source round-robin datasets" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][4]
    assert "long-context packing" in rows["Tiny CPU run.py AggregationDataLoader training smoke"][4]
    assert "public C4 streaming run.py training smoke" in rows["Large/gated remote training datasets"][2]
    assert "storage-backed tiny FineWeb-Edu default-runtime run.py smoke" in rows["Large/gated remote training datasets"][2]
    assert "bounded public-data paths" in rows["Large/gated remote training datasets"][2]
    assert "production PackingDataLoader.max_seq_len: 2048" in rows["Large/gated remote training datasets"][2]
    assert "optimizer-updating distributed FineWeb training" in rows["Large/gated remote training datasets"][2]
    assert "all four block combiners" in rows["Hybrid adapter/merger/vanilla/hymba configs"][2]
    assert "1.80B-parameter DoubleBlockAdapter" in rows["Hybrid adapter/merger/vanilla/hymba configs"][2]
    assert "real public Qwen weights/tokenizer" in rows["Hybrid adapter/merger/vanilla/hymba configs"][2]
    assert "production Llama scenario" in rows["Hybrid adapter/merger/vanilla/hymba configs"][2]
    assert "MOHAWK_ALLOW_CPU_TRAINING=1" in rows["Tiny CPU run.py supervised training smoke"][2]
    assert "tiny_cpu_gradient_accumulation.yaml" in rows["Tiny CPU run.py gradient accumulation smoke"][2]
    assert "effective_batch_size: 2" in rows["Tiny CPU run.py gradient accumulation smoke"][2]
    assert "TorchDataLoader.batch_size: 1" in rows["Tiny CPU run.py gradient accumulation smoke"][2]
    assert "accumulation_steps: 2" in rows["Tiny CPU run.py gradient accumulation smoke"][2]
    assert "n_batches: 4" in rows["Tiny CPU run.py gradient accumulation smoke"][2]
    assert "two micro-batches per optimizer step" in rows["Tiny CPU run.py gradient accumulation smoke"][2]
    assert "mohawk_tiny_gradient_accumulation_cpu_run/save" in rows["Tiny CPU run.py gradient accumulation smoke"][2]
    assert "db0807a38e96f60b84e2aa68ad3eb9d3a05f7b14091df541d9fe473e083b7b2c" in rows["Tiny CPU run.py gradient accumulation smoke"][2]
    assert "distributed no-sync coverage is recorded separately" in rows["Tiny CPU run.py gradient accumulation smoke"][4]
    assert "full-size checkpoints" in rows["Tiny CPU run.py gradient accumulation smoke"][4]
    assert "tiny_cuda_ddp_gradient_accumulation.yaml" in rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][2]
    assert "tiny_cuda_fsdp_gradient_accumulation.yaml" in rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][2]
    assert "configs/smoke/data_ddp" in rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][2]
    assert "TrainConfig.effective_batch_size: 4" in rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][2]
    assert "TorchDataLoader.batch_size: 1" in rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][2]
    assert "TrainConfig.n_tokens: 64" in rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][2]
    assert "FSDP wrapper now bridges no_sync()" in rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][2]
    assert "WORLD_SIZE=2" in rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][2]
    assert "FULL_SHARD" in rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][2]
    assert "accumulation_steps: 2" in rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][2]
    assert "n_batches: 4" in rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][2]
    assert "Total Grad Steps: 2" in rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][2]
    assert "Using DDP no_sync for gradient accumulation" in rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][2]
    assert "Using FSDP no_sync for gradient accumulation" in rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][2]
    assert "num_steps == 2" in rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][2]
    assert "torch.float32" in rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][2]
    assert "b75cb407af8b36f88a630ae2f4a9ddcc240edbf31548899075c4f8136c729e29" in rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][2]
    assert "mixed-precision and bfloat16 no_sync coverage are recorded separately" in rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][4]
    assert "multi-node DDP/FSDP" in rows["Tiny CUDA DDP/FSDP run.py gradient accumulation no_sync smokes"][4]
    assert "tiny_cuda_ddp_mixed_precision_gradient_accumulation.yaml" in rows["Tiny CUDA DDP/FSDP run.py mixed-precision gradient accumulation no_sync smokes"][2]
    assert "tiny_cuda_fsdp_mixed_precision_gradient_accumulation.yaml" in rows["Tiny CUDA DDP/FSDP run.py mixed-precision gradient accumulation no_sync smokes"][2]
    assert "Tiny-CUDA-DDP-Mixed-Precision-Gradient-Accumulation-Smoke" in rows["Tiny CUDA DDP/FSDP run.py mixed-precision gradient accumulation no_sync smokes"][2]
    assert "Tiny-CUDA-FSDP-Mixed-Precision-Gradient-Accumulation-Smoke" in rows["Tiny CUDA DDP/FSDP run.py mixed-precision gradient accumulation no_sync smokes"][2]
    assert "TrainConfig.mixed_precision: true" in rows["Tiny CUDA DDP/FSDP run.py mixed-precision gradient accumulation no_sync smokes"][2]
    assert "TeacherConfig.mixed_precision: true" in rows["Tiny CUDA DDP/FSDP run.py mixed-precision gradient accumulation no_sync smokes"][2]
    assert "effective_batch_size: 4" in rows["Tiny CUDA DDP/FSDP run.py mixed-precision gradient accumulation no_sync smokes"][2]
    assert "n_tokens: 64" in rows["Tiny CUDA DDP/FSDP run.py mixed-precision gradient accumulation no_sync smokes"][2]
    assert "Total Grad Steps: 2" in rows["Tiny CUDA DDP/FSDP run.py mixed-precision gradient accumulation no_sync smokes"][2]
    assert "Mixed Precision: True" in rows["Tiny CUDA DDP/FSDP run.py mixed-precision gradient accumulation no_sync smokes"][2]
    assert "Using DDP no_sync for gradient accumulation" in rows["Tiny CUDA DDP/FSDP run.py mixed-precision gradient accumulation no_sync smokes"][2]
    assert "Using FSDP no_sync for gradient accumulation" in rows["Tiny CUDA DDP/FSDP run.py mixed-precision gradient accumulation no_sync smokes"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CUDA DDP/FSDP run.py mixed-precision gradient accumulation no_sync smokes"][2]
    assert "num_steps == 2" in rows["Tiny CUDA DDP/FSDP run.py mixed-precision gradient accumulation no_sync smokes"][2]
    assert "69a62d39bd3931ba5ccfcdaa3a5d08e285a29a9fa359a15e16cb54646c6d5203" in rows["Tiny CUDA DDP/FSDP run.py mixed-precision gradient accumulation no_sync smokes"][2]
    assert "a8eb5ba1843710879037f98bb659cccf4d477f885337907172c057e1da62a97e" in rows["Tiny CUDA DDP/FSDP run.py mixed-precision gradient accumulation no_sync smokes"][2]
    assert "bfloat16 no_sync coverage is recorded separately" in rows["Tiny CUDA DDP/FSDP run.py mixed-precision gradient accumulation no_sync smokes"][4]
    assert "multi-node mixed precision coverage is recorded separately" in rows["Tiny CUDA DDP/FSDP run.py mixed-precision gradient accumulation no_sync smokes"][4]
    assert "tiny_cuda_ddp_bfloat16_gradient_accumulation.yaml" in rows["Tiny CUDA DDP/FSDP run.py bfloat16 gradient accumulation no_sync smokes"][2]
    assert "tiny_cuda_fsdp_bfloat16_gradient_accumulation.yaml" in rows["Tiny CUDA DDP/FSDP run.py bfloat16 gradient accumulation no_sync smokes"][2]
    assert "Tiny-CUDA-DDP-BFloat16-Gradient-Accumulation-Smoke" in rows["Tiny CUDA DDP/FSDP run.py bfloat16 gradient accumulation no_sync smokes"][2]
    assert "Tiny-CUDA-FSDP-BFloat16-Gradient-Accumulation-Smoke" in rows["Tiny CUDA DDP/FSDP run.py bfloat16 gradient accumulation no_sync smokes"][2]
    assert "TrainConfig.model_dtype: bfloat16" in rows["Tiny CUDA DDP/FSDP run.py bfloat16 gradient accumulation no_sync smokes"][2]
    assert "TeacherConfig.model_dtype: bfloat16" in rows["Tiny CUDA DDP/FSDP run.py bfloat16 gradient accumulation no_sync smokes"][2]
    assert "effective_batch_size: 4" in rows["Tiny CUDA DDP/FSDP run.py bfloat16 gradient accumulation no_sync smokes"][2]
    assert "n_tokens: 64" in rows["Tiny CUDA DDP/FSDP run.py bfloat16 gradient accumulation no_sync smokes"][2]
    assert "Total Grad Steps: 2" in rows["Tiny CUDA DDP/FSDP run.py bfloat16 gradient accumulation no_sync smokes"][2]
    assert "Cannot use mixed precision with bfloat16" in rows["Tiny CUDA DDP/FSDP run.py bfloat16 gradient accumulation no_sync smokes"][2]
    assert "Using DDP no_sync for gradient accumulation" in rows["Tiny CUDA DDP/FSDP run.py bfloat16 gradient accumulation no_sync smokes"][2]
    assert "Using FSDP no_sync for gradient accumulation" in rows["Tiny CUDA DDP/FSDP run.py bfloat16 gradient accumulation no_sync smokes"][2]
    assert "Model dtype: torch.bfloat16" in rows["Tiny CUDA DDP/FSDP run.py bfloat16 gradient accumulation no_sync smokes"][2]
    assert "Mixed Precision: False" in rows["Tiny CUDA DDP/FSDP run.py bfloat16 gradient accumulation no_sync smokes"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CUDA DDP/FSDP run.py bfloat16 gradient accumulation no_sync smokes"][2]
    assert "num_steps == 2" in rows["Tiny CUDA DDP/FSDP run.py bfloat16 gradient accumulation no_sync smokes"][2]
    assert "torch.bfloat16" in rows["Tiny CUDA DDP/FSDP run.py bfloat16 gradient accumulation no_sync smokes"][2]
    assert "84819bd613ccd234514e3b6aff202d5abdc69674963d7941328e80bf3ddd7282" in rows["Tiny CUDA DDP/FSDP run.py bfloat16 gradient accumulation no_sync smokes"][2]
    assert "multi-node bfloat16 coverage is recorded separately" in rows["Tiny CUDA DDP/FSDP run.py bfloat16 gradient accumulation no_sync smokes"][4]
    assert "configs/smoke/data_ddp_multinode" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "tiny_cuda_ddp_multinode_gradient_accumulation.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "tiny_cuda_fsdp_multinode_gradient_accumulation.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "tiny_cuda_ddp_default_runtime_multinode_gradient_accumulation.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "tiny_cuda_fsdp_default_runtime_multinode_gradient_accumulation.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "nvl72d091-T05" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "nvl72d091-T07" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "nvl72d066-T08" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "nvl72d066-T09" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "nvl72d063-T11" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "nvl72d063-T12" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "NVIDIA GB300" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "--nnodes=$SLURM_NNODES" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "--nproc_per_node=2" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "node_rank=0" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "node_rank=1" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "Tiny-CUDA-DDP-Multinode-Gradient-Accumulation-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "Tiny-CUDA-FSDP-Multinode-Gradient-Accumulation-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Gradient-Accumulation-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Gradient-Accumulation-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "Slurm jobs 692123 and 692150" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "WORLD_SIZE=4" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "wrappers ddp ddp" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "wrappers fsdp fsdp" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "FULL_SHARD FULL_SHARD" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "effective_batch_size: 8" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "n_tokens: 64" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "Total Grad Steps: 1" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "Model dtype: torch.float32" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "Mixed Precision: False" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "Activation Checkpointing: False" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "scheduler _step_count == 1" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "num_steps == 1" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "torch.float32" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][2]
    assert "multi-node mixed precision, multi-node bfloat16, multi-node resume, multi-node eval_hstates, multi-node eval_ppl, and multi-node benchmark coverage are recorded separately" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][4]
    assert "multi-node benchmark" in rows["Tiny CUDA multi-node DDP/FSDP run.py gradient accumulation smokes"][4]
    assert "tiny_cuda_ddp_multinode_mixed_precision_gradient_accumulation.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "tiny_cuda_fsdp_multinode_mixed_precision_gradient_accumulation.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "tiny_cuda_ddp_default_runtime_multinode_mixed_precision_gradient_accumulation.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "tiny_cuda_fsdp_default_runtime_multinode_mixed_precision_gradient_accumulation.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "nvl72d112-T11" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "nvl72d112-T13" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "nvl72d037-T15" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "nvl72d037-T18" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "nvl72d125-T07" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "nvl72d125-T08" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "nvl72d105-T17" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "nvl72d105-T18" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "Tiny-CUDA-DDP-Multinode-Mixed-Precision-Gradient-Accumulation-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "Tiny-CUDA-FSDP-Multinode-Mixed-Precision-Gradient-Accumulation-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Mixed-Precision-Gradient-Accumulation-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Mixed-Precision-Gradient-Accumulation-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "Slurm job 692716" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "Slurm job 692810" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "WORLD_SIZE=4" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "wrappers ddp ddp" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "wrappers fsdp fsdp" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "TrainConfig.mixed_precision: true" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "TeacherConfig.mixed_precision: true" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "FULL_SHARD FULL_SHARD" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "Model dtype: torch.float32" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "Mixed Precision: True" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "Activation Checkpointing: False" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "effective_batch_size: 8" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "n_tokens: 64" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "Total Grad Steps: 1" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "scheduler _step_count == 1" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "num_steps == 1" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "torch.float32" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][2]
    assert "multi-node bfloat16, multi-node resume, multi-node eval_hstates, multi-node eval_ppl, and multi-node benchmark coverage are recorded separately" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][4]
    assert "multi-node benchmark" in rows["Tiny CUDA multi-node DDP/FSDP run.py mixed-precision gradient accumulation smokes"][4]
    assert "tiny_cuda_ddp_multinode_bfloat16_gradient_accumulation.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "tiny_cuda_fsdp_multinode_bfloat16_gradient_accumulation.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "tiny_cuda_ddp_default_runtime_multinode_bfloat16_gradient_accumulation.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "tiny_cuda_fsdp_default_runtime_multinode_bfloat16_gradient_accumulation.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "nvl72d112-T11" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "nvl72d112-T13" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "nvl72d012-T01" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "nvl72d012-T02" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "nvl72d054-T01" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "nvl72d054-T02" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "nvl72d108-T15" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "nvl72d108-T18" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "Tiny-CUDA-DDP-Multinode-BFloat16-Gradient-Accumulation-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "Tiny-CUDA-FSDP-Multinode-BFloat16-Gradient-Accumulation-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-BFloat16-Gradient-Accumulation-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-BFloat16-Gradient-Accumulation-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "Slurm job 692872" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "Slurm job 692887" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "WORLD_SIZE=4" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "wrappers ddp ddp" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "wrappers fsdp fsdp" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "TrainConfig.model_dtype: bfloat16" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "TeacherConfig.model_dtype: bfloat16" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "model_dtype bfloat16 bfloat16" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "mixed precision True True" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "bfloat16 AMP fallback warning" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "FULL_SHARD FULL_SHARD" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "Model dtype: torch.bfloat16" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "Mixed Precision: False" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "Activation Checkpointing: False" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "effective_batch_size: 8" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "n_tokens: 64" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "Total Grad Steps: 1" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "scheduler _step_count == 1" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "num_steps == 1" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "torch.bfloat16" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "0bc9e79b22d4fce8a926de7260a144c0a5a0ff09a86328af5f35777ab655af15" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][2]
    assert "multi-node resume, multi-node eval_hstates, multi-node eval_ppl, and multi-node benchmark coverage are recorded separately" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][4]
    assert "multi-node benchmark" in rows["Tiny CUDA multi-node DDP/FSDP run.py bfloat16 gradient accumulation smokes"][4]
    assert "tiny_cuda_ddp_multinode_resume.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "tiny_cuda_fsdp_multinode_resume.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "tiny_cuda_ddp_default_runtime_multinode_resume.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "tiny_cuda_fsdp_default_runtime_multinode_resume.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "nvl72d094-T12" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "nvl72d094-T14" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "nvl72d016-T15" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "nvl72d016-T18" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "nvl72d053-T10" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "nvl72d053-T13" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "nvl72d053-T17" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "nvl72d053-T18" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "Tiny-CUDA-DDP-Multinode-Resume-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "Tiny-CUDA-FSDP-Multinode-Resume-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Resume-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Resume-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "Slurm job 692584" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "Slurm job 692601" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "WORLD_SIZE=4" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "wrappers ddp ddp" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "wrappers fsdp fsdp" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "FULL_SHARD FULL_SHARD" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "Model dtype: torch.float32" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "Mixed Precision: False" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "Activation Checkpointing: False" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "loaded 12/12 source model keys" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "tiny_cuda_ddp_default_runtime_multinode_gradient_accumulation/save" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "tiny_cuda_fsdp_default_runtime_multinode_gradient_accumulation/save" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "tiny_cuda_ddp_default_runtime_multinode_resume/save" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "tiny_cuda_fsdp_default_runtime_multinode_resume/save" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "DDP no-sync" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "FSDP no-sync" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "scheduler state advancing 1/1 -> 2/2" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "source optimizer state entries 0" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "target optimizer state entries 12" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "torch.float32" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "5faf34670d9ba7799cb14ed17fb29ec3679c01a003c1d9100783181ae2d6b2d9" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "hash_changed True" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][2]
    assert "multi-node eval_hstates or multi-node eval_ppl, which are recorded separately below" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][4]
    assert "multi-node benchmark" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][4]
    assert "mixed-precision or bfloat16 multi-node resume" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][4]
    assert "non-empty source optimizer-state restoration" in rows["Tiny CUDA multi-node DDP/FSDP run.py checkpoint resume smokes"][4]
    assert "tiny_cuda_ddp_multinode_eval_hstates.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "tiny_cuda_fsdp_multinode_eval_hstates.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "tiny_cuda_ddp_default_runtime_multinode_eval_hstates.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "tiny_cuda_fsdp_default_runtime_multinode_eval_hstates.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "Tiny-CUDA-DDP-Multinode-Eval-HStates-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "Tiny-CUDA-FSDP-Multinode-Eval-HStates-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Eval-HStates-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Eval-HStates-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "wrappers ddp ddp" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "and fsdp fsdp" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "NO_SHARD NO_SHARD" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "FULL_SHARD FULL_SHARD" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "nvl72d026-T08" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "nvl72d026-T10" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "nvl72d112-T11" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "nvl72d112-T13" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "Slurm jobs 692419 and 692430" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "nvl72d024-T10" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "nvl72d024-T11" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "nvl72d024-T12" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "nvl72d024-T13" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "WORLD_SIZE=4" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "configs/smoke/data_ddp_multinode" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "EvalConfig -> eval_hstates" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "eval_at_start True" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "eval_at_end True" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "save_latest True" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "n_batches 1" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "Model dtype: torch.float32" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "Mixed Precision: False" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "Activation Checkpointing: False" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "12/12 teacher keys" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "eval_score: 1.593651" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "hstates_distance: 1.593651" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "latest_eval_hstates.txt" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "1.5936508178710938" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "scheduler _step_count == 1" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "num_steps == 1" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "torch.float32" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "matching default-runtime final/latest hashes" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][2]
    assert "storage-backed default-runtime venv" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][4]
    assert "multi-node eval_ppl and multi-node benchmark coverage are recorded separately" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][4]
    assert "multi-node benchmark" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][4]
    assert "mixed-precision or bfloat16 multi-node eval callbacks" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][4]
    assert "representative eval data" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_hstates callback smokes"][4]
    assert "tiny_cuda_ddp_multinode_eval_ppl.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "tiny_cuda_fsdp_multinode_eval_ppl.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "tiny_cuda_ddp_default_runtime_multinode_eval_ppl.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "tiny_cuda_fsdp_default_runtime_multinode_eval_ppl.yaml" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "Tiny-CUDA-DDP-Multinode-Eval-PPL-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "Tiny-CUDA-FSDP-Multinode-Eval-PPL-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Eval-PPL-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Eval-PPL-Smoke" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "wrappers ddp ddp" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "wrappers fsdp fsdp" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "NO_SHARD NO_SHARD" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "FULL_SHARD FULL_SHARD" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "nvl72d012-T03" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "nvl72d012-T04" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "nvl72d103-T01" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "nvl72d103-T02" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "Slurm jobs 692462 and 692471" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "nvl72d066-T11" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "nvl72d066-T17" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "nvl72d024-T12" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "nvl72d024-T13" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "WORLD_SIZE=4" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "configs/smoke/data_ddp_multinode" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "EvalConfig -> eval_ppl" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "eval_at_start True" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "eval_at_end True" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "save_latest True" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "n_batches 1" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "Model dtype: torch.float32" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "Mixed Precision: False" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "Activation Checkpointing: False" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "12/12 teacher keys" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "eval_score: 44002.273438" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "perplexity: 44002.273438" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "accuracy: 0.000000" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "latest_Perplexity.txt" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "44002.2734375" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "scheduler _step_count == 1" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "num_steps == 1" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "torch.float32" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "matching default-runtime final/latest hashes" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][2]
    assert "storage-backed default-runtime venv" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][4]
    assert "multi-node benchmark" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][4]
    assert "mixed-precision or bfloat16 multi-node eval callbacks" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][4]
    assert "representative eval data" in rows["Tiny CUDA multi-node DDP/FSDP run.py eval_ppl callback smokes"][4]
    assert "tiny_cpu_wsd_scheduler.yaml" in rows["Tiny CPU run.py WSD scheduler smoke"][2]
    assert "OptimizerConfig.scheduler.name: wsd" in rows["Tiny CPU run.py WSD scheduler smoke"][2]
    assert "n_tokens: 32" in rows["Tiny CPU run.py WSD scheduler smoke"][2]
    assert "warmup_steps: 0.25" in rows["Tiny CPU run.py WSD scheduler smoke"][2]
    assert "decay_steps: 0.25" in rows["Tiny CPU run.py WSD scheduler smoke"][2]
    assert "Warmup Steps: 1" in rows["Tiny CPU run.py WSD scheduler smoke"][2]
    assert "Decay Steps: 1" in rows["Tiny CPU run.py WSD scheduler smoke"][2]
    assert "Total Grad Steps: 4" in rows["Tiny CPU run.py WSD scheduler smoke"][2]
    assert "12/12 keys" in rows["Tiny CPU run.py WSD scheduler smoke"][2]
    assert "mohawk_tiny_wsd_scheduler_cpu_run/save" in rows["Tiny CPU run.py WSD scheduler smoke"][2]
    assert "scheduler _step_count == 4" in rows["Tiny CPU run.py WSD scheduler smoke"][2]
    assert "num_steps == 4" in rows["Tiny CPU run.py WSD scheduler smoke"][2]
    assert "name: wsd" in rows["Tiny CPU run.py WSD scheduler smoke"][2]
    assert "6444608" in rows["Tiny CPU run.py WSD scheduler smoke"][2]
    assert "cbaff7bf8a10732de5205340eaa3943ffe06371c400984ef855ed50541a1751b" in rows["Tiny CPU run.py WSD scheduler smoke"][2]
    assert "cosine" in rows["Tiny CPU run.py WSD scheduler smoke"][4]
    assert "wsd_train" in rows["Tiny CPU run.py WSD scheduler smoke"][4]
    assert "tiny_cpu_adam_optimizer.yaml" in rows["Tiny CPU run.py Adam optimizer smoke"][2]
    assert "OptimizerConfig.optimizer: Adam" in rows["Tiny CPU run.py Adam optimizer smoke"][2]
    assert "configs/Llama/8B/llama/hstates.yaml" in rows["Tiny CPU run.py Adam optimizer smoke"][2]
    assert "torch.optim.Adam" in rows["Tiny CPU run.py Adam optimizer smoke"][2]
    assert "Optimizer configuration: Adam (" in rows["Tiny CPU run.py Adam optimizer smoke"][2]
    assert "decoupled_weight_decay: False" in rows["Tiny CPU run.py Adam optimizer smoke"][2]
    assert "mohawk_tiny_adam_optimizer_cpu_run/save" in rows["Tiny CPU run.py Adam optimizer smoke"][2]
    assert "optimizer Adam" in rows["Tiny CPU run.py Adam optimizer smoke"][2]
    assert "param_groups 1" in rows["Tiny CPU run.py Adam optimizer smoke"][2]
    assert "state_entries 12" in rows["Tiny CPU run.py Adam optimizer smoke"][2]
    assert "['exp_avg', 'exp_avg_sq', 'step']" in rows["Tiny CPU run.py Adam optimizer smoke"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CPU run.py Adam optimizer smoke"][2]
    assert "e153eb131e5bbbf7f1e1ad9913dcfc6082fe5a8c5c95822768d10a58ba653687" in rows["Tiny CPU run.py Adam optimizer smoke"][2]
    assert "DDP/FSDP optimizer wrapping" in rows["Tiny CPU run.py Adam optimizer smoke"][4]
    assert "tiny_cpu_optimize_weights_whitelist.yaml" in rows["Tiny CPU run.py optimize_weights whitelist/blacklist smokes"][2]
    assert "tiny_cpu_optimize_weights_blacklist.yaml" in rows["Tiny CPU run.py optimize_weights whitelist/blacklist smokes"][2]
    assert "white_list: [lm_head]" in rows["Tiny CPU run.py optimize_weights whitelist/blacklist smokes"][2]
    assert "black_list: [lm_head]" in rows["Tiny CPU run.py optimize_weights whitelist/blacklist smokes"][2]
    assert "804,112 / 1,610,832" in rows["Tiny CPU run.py optimize_weights whitelist/blacklist smokes"][2]
    assert "806,720 / 1,610,832" in rows["Tiny CPU run.py optimize_weights whitelist/blacklist smokes"][2]
    assert "changed only lm_head.weight" in rows["Tiny CPU run.py optimize_weights whitelist/blacklist smokes"][2]
    assert "kept only lm_head.weight unchanged" in rows["Tiny CPU run.py optimize_weights whitelist/blacklist smokes"][2]
    assert "6183032e556db32bb7e3ad59d801530f04116a4551c6efe0ede45681f889dbd2" in rows["Tiny CPU run.py optimize_weights whitelist/blacklist smokes"][2]
    assert "6351426c0d02e1e1471edc590934932daa404c8ee15dd0603f0bf74409cfa536" in rows["Tiny CPU run.py optimize_weights whitelist/blacklist smokes"][2]
    assert "DDP/FSDP flattened/original-parameter behavior" in rows["Tiny CPU run.py optimize_weights whitelist/blacklist smokes"][4]
    assert "tiny_cpu_load_model_whitelist.yaml" in rows["Tiny CPU run.py LoadConfig.model whitelist/blacklist smokes"][2]
    assert "tiny_cpu_load_model_blacklist.yaml" in rows["Tiny CPU run.py LoadConfig.model whitelist/blacklist smokes"][2]
    assert "LoadConfig.model.white_list: [lm_head]" in rows["Tiny CPU run.py LoadConfig.model whitelist/blacklist smokes"][2]
    assert "LoadConfig.model.black_list: [lm_head]" in rows["Tiny CPU run.py LoadConfig.model whitelist/blacklist smokes"][2]
    assert "loaded 1/12 student key" in rows["Tiny CPU run.py LoadConfig.model whitelist/blacklist smokes"][2]
    assert "loaded 11/12 student keys" in rows["Tiny CPU run.py LoadConfig.model whitelist/blacklist smokes"][2]
    assert "source_equal ['lm_head.weight']" in rows["Tiny CPU run.py LoadConfig.model whitelist/blacklist smokes"][2]
    assert "source_equal_count 11" in rows["Tiny CPU run.py LoadConfig.model whitelist/blacklist smokes"][2]
    assert "source_different ['lm_head.weight']" in rows["Tiny CPU run.py LoadConfig.model whitelist/blacklist smokes"][2]
    assert "expected_match True" in rows["Tiny CPU run.py LoadConfig.model whitelist/blacklist smokes"][2]
    assert "8d01e9141c1752f14212292b412825b9a8ff17dff7bbc076f5eb726f342780f3" in rows["Tiny CPU run.py LoadConfig.model whitelist/blacklist smokes"][2]
    assert "cfcfef52e5be35c99f25758f4bc17e72464245521ba45d2f946eb950a0fec015" in rows["Tiny CPU run.py LoadConfig.model whitelist/blacklist smokes"][2]
    assert "init_ssm_from_attention" in rows["Tiny CPU run.py LoadConfig.model whitelist/blacklist smokes"][4]
    assert "tiny_cpu_load_model_rename.yaml" in rows["Tiny CPU run.py LoadConfig.model rename smoke"][2]
    assert "--rename-key" in rows["Tiny CPU run.py LoadConfig.model rename smoke"][2]
    assert "lm_head.weight is absent" in rows["Tiny CPU run.py LoadConfig.model rename smoke"][2]
    assert "renamed_lm_head.weight" in rows["Tiny CPU run.py LoadConfig.model rename smoke"][2]
    assert "LoadConfig.model.rename: {renamed_lm_head: lm_head}" in rows["Tiny CPU run.py LoadConfig.model rename smoke"][2]
    assert "loaded 1/12 renamed student key" in rows["Tiny CPU run.py LoadConfig.model rename smoke"][2]
    assert "11 missing and 0 unexpected keys" in rows["Tiny CPU run.py LoadConfig.model rename smoke"][2]
    assert "source_has_lm_head False" in rows["Tiny CPU run.py LoadConfig.model rename smoke"][2]
    assert "source_has_renamed True" in rows["Tiny CPU run.py LoadConfig.model rename smoke"][2]
    assert "renamed_equal True" in rows["Tiny CPU run.py LoadConfig.model rename smoke"][2]
    assert "equal_original_names_count 0" in rows["Tiny CPU run.py LoadConfig.model rename smoke"][2]
    assert "6776e8e1f62c8bc6372e9afffbb21bd78bd38cc3f78e18c4e77f7fac216daaa6" in rows["Tiny CPU run.py LoadConfig.model rename smoke"][2]
    assert "multi-key rename collisions" in rows["Tiny CPU run.py LoadConfig.model rename smoke"][4]
    assert "init_ssm_from_attention" in rows["Tiny CPU run.py LoadConfig.model rename smoke"][4]
    assert "tiny_cpu_load_model_sequence.yaml" in rows["Tiny CPU run.py LoadConfig.model sequential-list smoke"][2]
    assert "two ordered LoadConfig.model entries" in rows["Tiny CPU run.py LoadConfig.model sequential-list smoke"][2]
    assert "mohawk_tiny_sequence_renamed_lm_head_ckpt" in rows["Tiny CPU run.py LoadConfig.model sequential-list smoke"][2]
    assert "head_equal_original False" in rows["Tiny CPU run.py LoadConfig.model sequential-list smoke"][2]
    assert "0.49912628531455994" in rows["Tiny CPU run.py LoadConfig.model sequential-list smoke"][2]
    assert "black_list: [lm_head]" in rows["Tiny CPU run.py LoadConfig.model sequential-list smoke"][2]
    assert "rename: {renamed_lm_head: lm_head}" in rows["Tiny CPU run.py LoadConfig.model sequential-list smoke"][2]
    assert "Loaded 12 keys in total" in rows["Tiny CPU run.py LoadConfig.model sequential-list smoke"][2]
    assert "OptimizerConfig.lr: 0.0" in rows["Tiny CPU run.py LoadConfig.model sequential-list smoke"][2]
    assert "load_entries 2" in rows["Tiny CPU run.py LoadConfig.model sequential-list smoke"][2]
    assert "backbone_equal_count 11" in rows["Tiny CPU run.py LoadConfig.model sequential-list smoke"][2]
    assert "head_equal_alt True" in rows["Tiny CPU run.py LoadConfig.model sequential-list smoke"][2]
    assert "head_equal_base False" in rows["Tiny CPU run.py LoadConfig.model sequential-list smoke"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CPU run.py LoadConfig.model sequential-list smoke"][2]
    assert "221881260f780b853e481ae250d48221d1ab120b8b367f8219588a8f7936ba12" in rows["Tiny CPU run.py LoadConfig.model sequential-list smoke"][2]
    assert "longer load chains" in rows["Tiny CPU run.py LoadConfig.model sequential-list smoke"][4]
    assert "multi-key rename collision behavior" in rows["Tiny CPU run.py LoadConfig.model sequential-list smoke"][4]
    assert "tiny_cpu_load_model_strict_missing.yaml" in rows["Tiny CPU run.py LoadConfig.model strict failure guards"][2]
    assert "tiny_cpu_load_model_strict_unexpected.yaml" in rows["Tiny CPU run.py LoadConfig.model strict failure guards"][2]
    assert "allow_missing_keys: false" in rows["Tiny CPU run.py LoadConfig.model strict failure guards"][2]
    assert "allow_unexpected_keys: false" in rows["Tiny CPU run.py LoadConfig.model strict failure guards"][2]
    assert "AssertionError: Missing keys" in rows["Tiny CPU run.py LoadConfig.model strict failure guards"][2]
    assert "AssertionError: Unexpected keys: ['renamed_lm_head.weight']" in rows["Tiny CPU run.py LoadConfig.model strict failure guards"][2]
    assert "only config.yaml" in rows["Tiny CPU run.py LoadConfig.model strict failure guards"][2]
    assert "expected-failure guard smokes do not validate successful training" in rows["Tiny CPU run.py LoadConfig.model strict failure guards"][4]
    assert "init_ssm_from_attention" in rows["Tiny CPU run.py LoadConfig.model strict failure guards"][4]
    assert "tiny_cpu_mixed_precision.yaml" in rows["Tiny CPU run.py centralized mixed-precision smoke"][2]
    assert "TrainConfig.mixed_precision: true" in rows["Tiny CPU run.py centralized mixed-precision smoke"][2]
    assert "TeacherConfig.mixed_precision: true" in rows["Tiny CPU run.py centralized mixed-precision smoke"][2]
    assert "torch.amp.autocast" in rows["Tiny CPU run.py centralized mixed-precision smoke"][2]
    assert "train-mode scaler path" in rows["Tiny CPU run.py centralized mixed-precision smoke"][2]
    assert "12/12 keys" in rows["Tiny CPU run.py centralized mixed-precision smoke"][2]
    assert "mohawk_tiny_mixed_precision_cpu_run/save" in rows["Tiny CPU run.py centralized mixed-precision smoke"][2]
    assert "3375940e29d65712f3426bf2cac70ff85c2365d1b8a590c6554d1edd9206f163" in rows["Tiny CPU run.py centralized mixed-precision smoke"][2]
    assert "CUDA autocast/scaler behavior" in rows["Tiny CPU run.py centralized mixed-precision smoke"][4]
    assert "DDP/FSDP mixed precision" in rows["Tiny CPU run.py centralized mixed-precision smoke"][4]
    assert "configs/smoke/tiny_cuda_mixed_precision.yaml" in rows["Centralized CUDA mixed precision"][2]
    assert "TrainConfig.mixed_precision: true" in rows["Centralized CUDA mixed precision"][2]
    assert "TeacherConfig.mixed_precision: true" in rows["Centralized CUDA mixed precision"][2]
    assert "cuda:0" in rows["Centralized CUDA mixed precision"][2]
    assert "optimizer state entries 12" in rows["Centralized CUDA mixed precision"][2]
    assert "bd20786dba84dfb0975db85ea79cebeb1929e1658318fdbfa6e0d18ec91cb42f" in rows["Centralized CUDA mixed precision"][2]
    assert "DDP/FSDP resume" in rows["Centralized CUDA mixed precision"][4]
    assert "tiny_cpu_mixed_precision_gradient_accumulation.yaml" in rows["Tiny CPU run.py mixed-precision gradient accumulation smoke"][2]
    assert "TrainConfig.mixed_precision: true" in rows["Tiny CPU run.py mixed-precision gradient accumulation smoke"][2]
    assert "TeacherConfig.mixed_precision: true" in rows["Tiny CPU run.py mixed-precision gradient accumulation smoke"][2]
    assert "effective_batch_size: 2" in rows["Tiny CPU run.py mixed-precision gradient accumulation smoke"][2]
    assert "TorchDataLoader.batch_size: 1" in rows["Tiny CPU run.py mixed-precision gradient accumulation smoke"][2]
    assert "n_tokens: 32" in rows["Tiny CPU run.py mixed-precision gradient accumulation smoke"][2]
    assert "accumulation_steps: 2" in rows["Tiny CPU run.py mixed-precision gradient accumulation smoke"][2]
    assert "n_batches: 4" in rows["Tiny CPU run.py mixed-precision gradient accumulation smoke"][2]
    assert "centralized mixed-precision wrapper path" in rows["Tiny CPU run.py mixed-precision gradient accumulation smoke"][2]
    assert "mohawk_tiny_mixed_precision_gradient_accumulation_cpu_run/save" in rows["Tiny CPU run.py mixed-precision gradient accumulation smoke"][2]
    assert "524e21de7874c2a5a29f99ac5dbfe0ec85583b516af001aa205cdd14708b233a" in rows["Tiny CPU run.py mixed-precision gradient accumulation smoke"][2]
    assert "distributed mixed-precision no_sync coverage is recorded separately" in rows["Tiny CPU run.py mixed-precision gradient accumulation smoke"][4]
    assert "tiny_cpu_bfloat16_mixed_precision_fallback.yaml" in rows["Tiny CPU run.py bfloat16 mixed-precision fallback smoke"][2]
    assert "model_dtype: bfloat16" in rows["Tiny CPU run.py bfloat16 mixed-precision fallback smoke"][2]
    assert "mixed_precision: true" in rows["Tiny CPU run.py bfloat16 mixed-precision fallback smoke"][2]
    assert "Cannot use mixed precision with bfloat16" in rows["Tiny CPU run.py bfloat16 mixed-precision fallback smoke"][2]
    assert "float != c10::BFloat16" in rows["Tiny CPU run.py bfloat16 mixed-precision fallback smoke"][2]
    assert "honor factory dtype/device" in rows["Tiny CPU run.py bfloat16 mixed-precision fallback smoke"][2]
    assert "without moving meta tensors to CPU" in rows["Tiny CPU run.py bfloat16 mixed-precision fallback smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py bfloat16 mixed-precision fallback smoke"][2]
    assert "a2943247863988b07f553af2359eaaab936674d8c76148ffa7801a27390b4754" in rows["Tiny CPU run.py bfloat16 mixed-precision fallback smoke"][2]
    assert "CUDA bfloat16" in rows["Tiny CPU run.py bfloat16 mixed-precision fallback smoke"][4]
    assert "representative configs" in rows["Tiny CPU run.py bfloat16 mixed-precision fallback smoke"][4]
    assert "tiny_cuda_bfloat16_supervised.yaml" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 supervised smokes"][2]
    assert "tiny_cuda_ddp_bfloat16_supervised.yaml" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 supervised smokes"][2]
    assert "tiny_cuda_fsdp_bfloat16_supervised.yaml" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 supervised smokes"][2]
    assert "TrainConfig.model_dtype: bfloat16" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 supervised smokes"][2]
    assert "TeacherConfig.model_dtype: bfloat16" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 supervised smokes"][2]
    assert "Cannot use mixed precision with bfloat16" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 supervised smokes"][2]
    assert "broadcast teacher weights" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 supervised smokes"][2]
    assert "FULL_SHARD" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 supervised smokes"][2]
    assert "torch.bfloat16" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 supervised smokes"][2]
    assert "52fe58e628de69a168b351d41c8d01a52017300497a0675770ae75653ccf1ef1" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 supervised smokes"][2]
    assert "0bc9e79b22d4fce8a926de7260a144c0a5a0ff09a86328af5f35777ab655af15" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 supervised smokes"][2]
    assert "representative configs" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 supervised smokes"][4]
    assert "multi-node DDP/FSDP" in rows["Tiny CUDA/DDP/FSDP run.py bfloat16 supervised smokes"][4]
    assert "tiny_cpu_bfloat16_gradient_accumulation_fallback.yaml" in rows["Tiny CPU run.py bfloat16 gradient accumulation fallback smoke"][2]
    assert "model_dtype: bfloat16" in rows["Tiny CPU run.py bfloat16 gradient accumulation fallback smoke"][2]
    assert "mixed_precision: true" in rows["Tiny CPU run.py bfloat16 gradient accumulation fallback smoke"][2]
    assert "effective_batch_size: 2" in rows["Tiny CPU run.py bfloat16 gradient accumulation fallback smoke"][2]
    assert "TorchDataLoader.batch_size: 1" in rows["Tiny CPU run.py bfloat16 gradient accumulation fallback smoke"][2]
    assert "n_tokens: 32" in rows["Tiny CPU run.py bfloat16 gradient accumulation fallback smoke"][2]
    assert "Cannot use mixed precision with bfloat16" in rows["Tiny CPU run.py bfloat16 gradient accumulation fallback smoke"][2]
    assert "12/12 keys" in rows["Tiny CPU run.py bfloat16 gradient accumulation fallback smoke"][2]
    assert "accumulation_steps: 2" in rows["Tiny CPU run.py bfloat16 gradient accumulation fallback smoke"][2]
    assert "n_batches: 4" in rows["Tiny CPU run.py bfloat16 gradient accumulation fallback smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py bfloat16 gradient accumulation fallback smoke"][2]
    assert "0b1885aa63aa315e61d7c244fa188375bdc5d0405a54466f1b1ade0f8322a1d1" in rows["Tiny CPU run.py bfloat16 gradient accumulation fallback smoke"][2]
    assert "CUDA bfloat16" in rows["Tiny CPU run.py bfloat16 gradient accumulation fallback smoke"][4]
    assert "representative configs" in rows["Tiny CPU run.py bfloat16 gradient accumulation fallback smoke"][4]
    assert "tiny_cpu_qwen2_bfloat16_gradient_accumulation_fallback.yaml" in rows["Tiny CPU run.py Qwen2/Phi/Falcon bfloat16 gradient accumulation fallback smokes"][2]
    assert "tiny_cpu_phi_bfloat16_gradient_accumulation_fallback.yaml" in rows["Tiny CPU run.py Qwen2/Phi/Falcon bfloat16 gradient accumulation fallback smokes"][2]
    assert "tiny_cpu_falcon_bfloat16_gradient_accumulation_fallback.yaml" in rows["Tiny CPU run.py Qwen2/Phi/Falcon bfloat16 gradient accumulation fallback smokes"][2]
    assert "15/15" in rows["Tiny CPU run.py Qwen2/Phi/Falcon bfloat16 gradient accumulation fallback smokes"][2]
    assert "18/18" in rows["Tiny CPU run.py Qwen2/Phi/Falcon bfloat16 gradient accumulation fallback smokes"][2]
    assert "13/13" in rows["Tiny CPU run.py Qwen2/Phi/Falcon bfloat16 gradient accumulation fallback smokes"][2]
    assert "accumulation_steps: 2" in rows["Tiny CPU run.py Qwen2/Phi/Falcon bfloat16 gradient accumulation fallback smokes"][2]
    assert "n_batches: 4" in rows["Tiny CPU run.py Qwen2/Phi/Falcon bfloat16 gradient accumulation fallback smokes"][2]
    assert "mohawk_tiny_qwen2_bfloat16_gradient_accumulation_cpu_run/save" in rows["Tiny CPU run.py Qwen2/Phi/Falcon bfloat16 gradient accumulation fallback smokes"][2]
    assert "mohawk_tiny_phi_bfloat16_gradient_accumulation_cpu_run/save" in rows["Tiny CPU run.py Qwen2/Phi/Falcon bfloat16 gradient accumulation fallback smokes"][2]
    assert "mohawk_tiny_falcon_bfloat16_gradient_accumulation_cpu_run/save" in rows["Tiny CPU run.py Qwen2/Phi/Falcon bfloat16 gradient accumulation fallback smokes"][2]
    assert "de79eb48388fc3e1fc2f0d51090287bf56432378819edb401638977063e98c3b" in rows["Tiny CPU run.py Qwen2/Phi/Falcon bfloat16 gradient accumulation fallback smokes"][2]
    assert "b33c9800b8805dac077546f9352be1850d1a16358a5000a4aafcaed15b7cb254" in rows["Tiny CPU run.py Qwen2/Phi/Falcon bfloat16 gradient accumulation fallback smokes"][2]
    assert "caa827ffba2775c917dae416fe5cf660d925c03f43bb3fbd4af977f7a9b4a25e" in rows["Tiny CPU run.py Qwen2/Phi/Falcon bfloat16 gradient accumulation fallback smokes"][2]
    assert "fast Mamba kernels" in rows["Tiny CPU run.py Qwen2/Phi/Falcon bfloat16 gradient accumulation fallback smokes"][4]
    assert "tiny_cpu_qwen2_bfloat16_mixed_precision_fallback.yaml" in rows["Tiny CPU run.py Qwen2 bfloat16 mixed-precision fallback smoke"][2]
    assert "Qwen2Model -> Qwen2Block -> Qwen2Attention" in rows["Tiny CPU run.py Qwen2 bfloat16 mixed-precision fallback smoke"][2]
    assert "model_dtype: bfloat16" in rows["Tiny CPU run.py Qwen2 bfloat16 mixed-precision fallback smoke"][2]
    assert "mixed_precision: true" in rows["Tiny CPU run.py Qwen2 bfloat16 mixed-precision fallback smoke"][2]
    assert "15/15 keys" in rows["Tiny CPU run.py Qwen2 bfloat16 mixed-precision fallback smoke"][2]
    assert "Float and BFloat16" in rows["Tiny CPU run.py Qwen2 bfloat16 mixed-precision fallback smoke"][2]
    assert "honor factory dtype/device" in rows["Tiny CPU run.py Qwen2 bfloat16 mixed-precision fallback smoke"][2]
    assert "without moving meta tensors to CPU" in rows["Tiny CPU run.py Qwen2 bfloat16 mixed-precision fallback smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py Qwen2 bfloat16 mixed-precision fallback smoke"][2]
    assert "34fba47d457715fc90c45cc81cc3f4a187ba9e55661f0352abef53057395b9e0" in rows["Tiny CPU run.py Qwen2 bfloat16 mixed-precision fallback smoke"][2]
    assert "DDP/FSDP" in rows["Tiny CPU run.py Qwen2 bfloat16 mixed-precision fallback smoke"][4]
    assert "representative configs" in rows["Tiny CPU run.py Qwen2 bfloat16 mixed-precision fallback smoke"][4]
    assert rows["Tiny CPU run.py Phi bfloat16 mixed-precision fallback smoke"][3] == "FAILED_FIXED"
    assert "tiny_cpu_phi_bfloat16_mixed_precision_fallback.yaml" in rows["Tiny CPU run.py Phi bfloat16 mixed-precision fallback smoke"][2]
    assert "MambaPhi -> PhiAttention" in rows["Tiny CPU run.py Phi bfloat16 mixed-precision fallback smoke"][2]
    assert "model_dtype: bfloat16" in rows["Tiny CPU run.py Phi bfloat16 mixed-precision fallback smoke"][2]
    assert "mixed_precision: true" in rows["Tiny CPU run.py Phi bfloat16 mixed-precision fallback smoke"][2]
    assert "18/18 keys" in rows["Tiny CPU run.py Phi bfloat16 mixed-precision fallback smoke"][2]
    assert "BFloat16 and Float" in rows["Tiny CPU run.py Phi bfloat16 mixed-precision fallback smoke"][2]
    assert "PhiAttention" in rows["Tiny CPU run.py Phi bfloat16 mixed-precision fallback smoke"][2]
    assert "PhiMLP" in rows["Tiny CPU run.py Phi bfloat16 mixed-precision fallback smoke"][2]
    assert "honor factory dtype/device" in rows["Tiny CPU run.py Phi bfloat16 mixed-precision fallback smoke"][2]
    assert "without moving meta tensors to CPU" in rows["Tiny CPU run.py Phi bfloat16 mixed-precision fallback smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py Phi bfloat16 mixed-precision fallback smoke"][2]
    assert "bf5232b363bc71eb665857afafc2c644c469291e7ccfa25611778b5999ddba8d" in rows["Tiny CPU run.py Phi bfloat16 mixed-precision fallback smoke"][2]
    assert "DDP/FSDP" in rows["Tiny CPU run.py Phi bfloat16 mixed-precision fallback smoke"][4]
    assert "representative configs" in rows["Tiny CPU run.py Phi bfloat16 mixed-precision fallback smoke"][4]
    assert rows["Tiny CPU run.py Falcon bfloat16 mixed-precision fallback smoke"][3] == "FAILED_FIXED"
    assert "tiny_cpu_falcon_bfloat16_mixed_precision_fallback.yaml" in rows["Tiny CPU run.py Falcon bfloat16 mixed-precision fallback smoke"][2]
    assert "FalconBlock -> FalconMambaMixer" in rows["Tiny CPU run.py Falcon bfloat16 mixed-precision fallback smoke"][2]
    assert "Transformers' sequential Mamba fallback" in rows["Tiny CPU run.py Falcon bfloat16 mixed-precision fallback smoke"][2]
    assert "model_dtype: bfloat16" in rows["Tiny CPU run.py Falcon bfloat16 mixed-precision fallback smoke"][2]
    assert "mixed_precision: true" in rows["Tiny CPU run.py Falcon bfloat16 mixed-precision fallback smoke"][2]
    assert "13/13 keys" in rows["Tiny CPU run.py Falcon bfloat16 mixed-precision fallback smoke"][2]
    assert "float != c10::BFloat16" in rows["Tiny CPU run.py Falcon bfloat16 mixed-precision fallback smoke"][2]
    assert "fp32 residual path returned float32 hidden states" in rows["Tiny CPU run.py Falcon bfloat16 mixed-precision fallback smoke"][2]
    assert "honor factory dtype/device" in rows["Tiny CPU run.py Falcon bfloat16 mixed-precision fallback smoke"][2]
    assert "back to the input dtype" in rows["Tiny CPU run.py Falcon bfloat16 mixed-precision fallback smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py Falcon bfloat16 mixed-precision fallback smoke"][2]
    assert "6493528660f88e271dda8b25f649b49581b5b1852c70ac7498aa17137e3a32be" in rows["Tiny CPU run.py Falcon bfloat16 mixed-precision fallback smoke"][2]
    assert "fast Mamba kernels" in rows["Tiny CPU run.py Falcon bfloat16 mixed-precision fallback smoke"][4]
    assert "tiny_cuda_qwen2_bfloat16_eval_callbacks.yaml" in rows["Tiny CUDA run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "tiny_cuda_phi_bfloat16_eval_callbacks.yaml" in rows["Tiny CUDA run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "tiny_cuda_falcon_bfloat16_eval_callbacks.yaml" in rows["Tiny CUDA run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "Qwen2Model -> Qwen2Block -> Qwen2Attention" in rows["Tiny CUDA run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "LlamaModel -> MambaPhi -> PhiAttention" in rows["Tiny CUDA run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "LlamaModel -> FalconBlock -> FalconMambaMixer" in rows["Tiny CUDA run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "15/15 teacher keys" in rows["Tiny CUDA run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "18/18 keys" in rows["Tiny CUDA run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "13/13 keys" in rows["Tiny CUDA run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "latest_eval_hstates" in rows["Tiny CUDA run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "latest_Perplexity" in rows["Tiny CUDA run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "59874.144531" in rows["Tiny CUDA run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "40134.855469" in rows["Tiny CUDA run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "62943.968750" in rows["Tiny CUDA run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "8414e24632d042ced981bc7f4457205e643195d41f9ff7d6b4a0f37da58d3a19" in rows["Tiny CUDA run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "e84464330dba07b4ef35e0f538d923fd4c188cae2cb1458b6cb18f93b60c66f0" in rows["Tiny CUDA run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "a0297d015d195ccda5cb633c16624df99e256053a09f14d31a140073bc73d790" in rows["Tiny CUDA run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "fast Mamba kernels" in rows["Tiny CUDA run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][4]
    assert "tiny_cuda_ddp_qwen2_bfloat16_eval_callbacks.yaml" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "tiny_cuda_fsdp_qwen2_bfloat16_eval_callbacks.yaml" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "tiny_cuda_ddp_phi_bfloat16_eval_callbacks.yaml" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "tiny_cuda_fsdp_phi_bfloat16_eval_callbacks.yaml" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "tiny_cuda_ddp_falcon_bfloat16_eval_callbacks.yaml" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "tiny_cuda_fsdp_falcon_bfloat16_eval_callbacks.yaml" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "single-process CUDA callback configs appended two stale eval entries" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "15/15 teacher keys" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "18/18 keys" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "13/13 keys" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "1.4765625" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "69068.7421875" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "1.6484375" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "50082.56640625" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "0.99609375" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "53790.79296875" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "c95790cea5ec1fe21235972050d52293b981a21fb4f50bfd1acb3e9f308511ba" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "82d94929797662c9abc5b7ecb20962f82dfef0c41d252be6f5a4be8e31045e3b" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "ef48a53d4ba3941435096b197361619301b7560ef96322543ab0ece75b3c3dde" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "scheduler _step_count == 1" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][2]
    assert "fast Mamba kernels" in rows["Tiny CUDA DDP/FSDP run.py Qwen2/Phi/Falcon bfloat16 eval callback smokes"][4]
    assert "tiny_cpu_compile_model.yaml" in rows["Tiny CPU run.py compile_model smoke"][2]
    assert "TrainConfig.compile_model: true" in rows["Tiny CPU run.py compile_model smoke"][2]
    assert "TeacherConfig.compile_model: false" in rows["Tiny CPU run.py compile_model smoke"][2]
    assert "torch.compile" in rows["Tiny CPU run.py compile_model smoke"][2]
    assert "fullgraph: true" in rows["Tiny CPU run.py compile_model smoke"][2]
    assert "MixerModel" in rows["Tiny CPU run.py compile_model smoke"][2]
    assert "no _orig_mod. prefixes" in rows["Tiny CPU run.py compile_model smoke"][2]
    assert "no missing or unexpected keys" in rows["Tiny CPU run.py compile_model smoke"][2]
    assert "44640dc7493a12ffd96e65774835a60a466383b921a0ff81831c69e9d059c439" in rows["Tiny CPU run.py compile_model smoke"][2]
    assert "DDP/FSDP compile paths" in rows["Tiny CPU run.py compile_model smoke"][4]
    assert "tiny_cpu_bfloat16_compile_model.yaml" in rows["Tiny CPU run.py bfloat16 compile_model smoke"][2]
    assert "TrainConfig.compile_model: true" in rows["Tiny CPU run.py bfloat16 compile_model smoke"][2]
    assert "TrainConfig.model_dtype: bfloat16" in rows["Tiny CPU run.py bfloat16 compile_model smoke"][2]
    assert "bfloat16 AMP fallback" in rows["Tiny CPU run.py bfloat16 compile_model smoke"][2]
    assert "Inductor" in rows["Tiny CPU run.py bfloat16 compile_model smoke"][2]
    assert "fullgraph: true" in rows["Tiny CPU run.py bfloat16 compile_model smoke"][2]
    assert "12/12 teacher keys" in rows["Tiny CPU run.py bfloat16 compile_model smoke"][2]
    assert "mohawk_tiny_bfloat16_compile_model_cpu_run/save" in rows["Tiny CPU run.py bfloat16 compile_model smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py bfloat16 compile_model smoke"][2]
    assert "no _orig_mod. prefixes" in rows["Tiny CPU run.py bfloat16 compile_model smoke"][2]
    assert "uncompiled bfloat16 local model" in rows["Tiny CPU run.py bfloat16 compile_model smoke"][2]
    assert "3e606413e6aa7465dd7dc6ea0677f3ecb89f449185e7768ef5b7fac96c454695" in rows["Tiny CPU run.py bfloat16 compile_model smoke"][2]
    assert "DDP/FSDP compile paths" in rows["Tiny CPU run.py bfloat16 compile_model smoke"][4]
    assert "tiny_cpu_teacher_compile_model.yaml" in rows["Tiny CPU run.py teacher compile_model smoke"][2]
    assert "TrainConfig.compile_model: false" in rows["Tiny CPU run.py teacher compile_model smoke"][2]
    assert "TeacherConfig.compile_model: true" in rows["Tiny CPU run.py teacher compile_model smoke"][2]
    assert "inference parameters non-trainable" in rows["Tiny CPU run.py teacher compile_model smoke"][2]
    assert "Inductor" in rows["Tiny CPU run.py teacher compile_model smoke"][2]
    assert "fullgraph: true" in rows["Tiny CPU run.py teacher compile_model smoke"][2]
    assert "12/12 keys" in rows["Tiny CPU run.py teacher compile_model smoke"][2]
    assert "mohawk_tiny_teacher_compile_model_cpu_run/save" in rows["Tiny CPU run.py teacher compile_model smoke"][2]
    assert "train_compile False" in rows["Tiny CPU run.py teacher compile_model smoke"][2]
    assert "teacher_compile True" in rows["Tiny CPU run.py teacher compile_model smoke"][2]
    assert "torch.float32" in rows["Tiny CPU run.py teacher compile_model smoke"][2]
    assert "8c4c51efd1bdd1966139a863087e7b5d726eb42e6d942f090dcc8f50ddac9227" in rows["Tiny CPU run.py teacher compile_model smoke"][2]
    assert "tiny_cpu_student_teacher_compile_model.yaml" in rows["Tiny CPU run.py student+teacher compile_model smoke"][2]
    assert "TrainConfig.compile_model: true" in rows["Tiny CPU run.py student+teacher compile_model smoke"][2]
    assert "TeacherConfig.compile_model: true" in rows["Tiny CPU run.py student+teacher compile_model smoke"][2]
    assert "compiled both student and teacher" in rows["Tiny CPU run.py student+teacher compile_model smoke"][2]
    assert "12/12 keys" in rows["Tiny CPU run.py student+teacher compile_model smoke"][2]
    assert "mohawk_tiny_student_teacher_compile_model_cpu_run/save" in rows["Tiny CPU run.py student+teacher compile_model smoke"][2]
    assert "train_compile True" in rows["Tiny CPU run.py student+teacher compile_model smoke"][2]
    assert "teacher_compile True" in rows["Tiny CPU run.py student+teacher compile_model smoke"][2]
    assert "torch.float32" in rows["Tiny CPU run.py student+teacher compile_model smoke"][2]
    assert "4a8fb367c41906c5fee49cd786c47844157fe6ccaf9a5fb7005e5f9010a64b7d" in rows["Tiny CPU run.py student+teacher compile_model smoke"][2]
    assert "tiny_cpu_bfloat16_teacher_compile_model.yaml" in rows["Tiny CPU run.py bfloat16 teacher compile_model smoke"][2]
    assert "TrainConfig.compile_model: false" in rows["Tiny CPU run.py bfloat16 teacher compile_model smoke"][2]
    assert "TeacherConfig.compile_model: true" in rows["Tiny CPU run.py bfloat16 teacher compile_model smoke"][2]
    assert "bfloat16 AMP fallback" in rows["Tiny CPU run.py bfloat16 teacher compile_model smoke"][2]
    assert "compiled the bfloat16 teacher" in rows["Tiny CPU run.py bfloat16 teacher compile_model smoke"][2]
    assert "mohawk_tiny_bfloat16_teacher_compile_model_cpu_run/save" in rows["Tiny CPU run.py bfloat16 teacher compile_model smoke"][2]
    assert "train_compile False" in rows["Tiny CPU run.py bfloat16 teacher compile_model smoke"][2]
    assert "teacher_compile True" in rows["Tiny CPU run.py bfloat16 teacher compile_model smoke"][2]
    assert "train True bfloat16" in rows["Tiny CPU run.py bfloat16 teacher compile_model smoke"][2]
    assert "teacher True bfloat16" in rows["Tiny CPU run.py bfloat16 teacher compile_model smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py bfloat16 teacher compile_model smoke"][2]
    assert "d92f42c1d8bf9b50810ec3369ab911f89688db9491f1d8a00e6b2c7976ab5c5a" in rows["Tiny CPU run.py bfloat16 teacher compile_model smoke"][2]
    assert "tiny_cpu_bfloat16_student_teacher_compile_model.yaml" in rows["Tiny CPU run.py bfloat16 student+teacher compile_model smoke"][2]
    assert "TrainConfig.compile_model: true" in rows["Tiny CPU run.py bfloat16 student+teacher compile_model smoke"][2]
    assert "TeacherConfig.compile_model: true" in rows["Tiny CPU run.py bfloat16 student+teacher compile_model smoke"][2]
    assert "bfloat16 AMP fallback" in rows["Tiny CPU run.py bfloat16 student+teacher compile_model smoke"][2]
    assert "compiled both student and teacher" in rows["Tiny CPU run.py bfloat16 student+teacher compile_model smoke"][2]
    assert "mohawk_tiny_bfloat16_student_teacher_compile_model_cpu_run/save" in rows["Tiny CPU run.py bfloat16 student+teacher compile_model smoke"][2]
    assert "train_compile True" in rows["Tiny CPU run.py bfloat16 student+teacher compile_model smoke"][2]
    assert "teacher_compile True" in rows["Tiny CPU run.py bfloat16 student+teacher compile_model smoke"][2]
    assert "train True bfloat16" in rows["Tiny CPU run.py bfloat16 student+teacher compile_model smoke"][2]
    assert "teacher True bfloat16" in rows["Tiny CPU run.py bfloat16 student+teacher compile_model smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py bfloat16 student+teacher compile_model smoke"][2]
    assert "1637d7e9f7f2822aafc77b631f21b8b073ec08169a4c28a4d182509d33bbf5f1" in rows["Tiny CPU run.py bfloat16 student+teacher compile_model smoke"][2]
    assert "TeacherConfig.dir: sshleifer/tiny-gpt2" in rows["Tiny CPU run.py public Hugging Face teacher smoke"][2]
    assert "isolated /tmp HF cache was empty" in rows["Tiny CPU run.py public Hugging Face teacher smoke"][2]
    assert "loaded 29/29 real HF teacher keys" in rows["Tiny CPU run.py public Hugging Face teacher smoke"][2]
    assert "mohawk_tiny_hf_teacher_cpu_run/save" in rows["Tiny CPU run.py public Hugging Face teacher smoke"][2]
    assert "6a45a7b408356f127bf72d89ea33e2a7ea045627fa0b92475f86345c601e9a9e" in rows["Tiny CPU run.py public Hugging Face teacher smoke"][2]
    assert "does not validate large/gated teachers" in rows["Tiny CPU run.py public Hugging Face teacher smoke"][4]
    assert "output_hidden_states=True" in rows["Tiny CPU run.py hstates training smoke"][2]
    assert "Teacher model should have at least 2 hidden states" in rows["Tiny CPU run.py hstates training smoke"][2]
    assert "remaining objective types" in rows["Tiny CPU run.py hstates training smoke"][4]
    assert "tiny_cpu_bfloat16_hstates.yaml" in rows["Tiny CPU run.py bfloat16 hstates training smoke"][2]
    assert "DistillConfig.type: hstates" in rows["Tiny CPU run.py bfloat16 hstates training smoke"][2]
    assert "TrainConfig.model_dtype bfloat16" in rows["Tiny CPU run.py bfloat16 hstates training smoke"][2]
    assert "TeacherConfig.model_dtype bfloat16" in rows["Tiny CPU run.py bfloat16 hstates training smoke"][2]
    assert "12/12 keys" in rows["Tiny CPU run.py bfloat16 hstates training smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py bfloat16 hstates training smoke"][2]
    assert "42ba2930c56370fcdbbc3ac5a9aa03c02bc11b2c47f671a1272cde227df2a70c" in rows["Tiny CPU run.py bfloat16 hstates training smoke"][2]
    assert "representative hidden-state data" in rows["Tiny CPU run.py bfloat16 hstates training smoke"][4]
    assert "output_hidden_states=True" in rows["Tiny CPU run.py sequential_hstates training smoke"][2]
    assert "mohawk_tiny_sequential_hstates_cpu_run/save" in rows["Tiny CPU run.py sequential_hstates training smoke"][2]
    assert "tiny_cpu_bfloat16_sequential_hstates.yaml" in rows["Tiny CPU run.py bfloat16 sequential_hstates training smoke"][2]
    assert "DistillConfig.type: sequential_hstates" in rows["Tiny CPU run.py bfloat16 sequential_hstates training smoke"][2]
    assert "output_hidden_states=True" in rows["Tiny CPU run.py bfloat16 sequential_hstates training smoke"][2]
    assert "hstates_gap 1" in rows["Tiny CPU run.py bfloat16 sequential_hstates training smoke"][2]
    assert "TrainConfig.model_dtype bfloat16" in rows["Tiny CPU run.py bfloat16 sequential_hstates training smoke"][2]
    assert "TeacherConfig.model_dtype bfloat16" in rows["Tiny CPU run.py bfloat16 sequential_hstates training smoke"][2]
    assert "12/12 keys" in rows["Tiny CPU run.py bfloat16 sequential_hstates training smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py bfloat16 sequential_hstates training smoke"][2]
    assert "bb3c68f0d13a1da992fd61f86d7d5348fe0990ac9b9c3d8537a248c152972957" in rows["Tiny CPU run.py bfloat16 sequential_hstates training smoke"][2]
    assert "representative hidden-state data" in rows["Tiny CPU run.py bfloat16 sequential_hstates training smoke"][4]
    assert "output_attentions=True" in rows["Tiny CPU run.py matrices training smoke"][2]
    assert "mohawk_tiny_matrices_cpu_run/save" in rows["Tiny CPU run.py matrices training smoke"][2]
    assert "DPO/instruct objectives" in rows["Tiny CPU run.py matrices training smoke"][4]
    assert "tiny_cpu_bfloat16_matrices.yaml" in rows["Tiny CPU run.py bfloat16 matrices training smoke"][2]
    assert "DistillConfig.type: matrices" in rows["Tiny CPU run.py bfloat16 matrices training smoke"][2]
    assert "output_attentions=True" in rows["Tiny CPU run.py bfloat16 matrices training smoke"][2]
    assert "TrainConfig.model_dtype bfloat16" in rows["Tiny CPU run.py bfloat16 matrices training smoke"][2]
    assert "TeacherConfig.model_dtype bfloat16" in rows["Tiny CPU run.py bfloat16 matrices training smoke"][2]
    assert "12/12 keys" in rows["Tiny CPU run.py bfloat16 matrices training smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py bfloat16 matrices training smoke"][2]
    assert "1e206bf68e4ea570c0b324ad314bc3fbf9d51e331fa179342808c17cbede0270" in rows["Tiny CPU run.py bfloat16 matrices training smoke"][2]
    assert "representative attention-matrix data" in rows["Tiny CPU run.py bfloat16 matrices training smoke"][4]
    assert "collate_type=preference" in rows["Tiny CPU run.py DPO training smoke"][2]
    assert "Dimension specified as 0" in rows["Tiny CPU run.py DPO training smoke"][2]
    assert "mohawk_tiny_dpo_cpu_run/save" in rows["Tiny CPU run.py DPO training smoke"][2]
    assert "tiny_cpu_bfloat16_dpo.yaml" in rows["Tiny CPU run.py bfloat16 DPO training smoke"][2]
    assert "DistillConfig.type: dpo" in rows["Tiny CPU run.py bfloat16 DPO training smoke"][2]
    assert "DistillConfig.beta 0.1" in rows["Tiny CPU run.py bfloat16 DPO training smoke"][2]
    assert "collate_type=preference" in rows["Tiny CPU run.py bfloat16 DPO training smoke"][2]
    assert "chosen_ids" in rows["Tiny CPU run.py bfloat16 DPO training smoke"][2]
    assert "rejected_ids" in rows["Tiny CPU run.py bfloat16 DPO training smoke"][2]
    assert "TrainConfig.model_dtype bfloat16" in rows["Tiny CPU run.py bfloat16 DPO training smoke"][2]
    assert "TeacherConfig.model_dtype bfloat16" in rows["Tiny CPU run.py bfloat16 DPO training smoke"][2]
    assert "12/12 keys" in rows["Tiny CPU run.py bfloat16 DPO training smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py bfloat16 DPO training smoke"][2]
    assert "6cc7796660ade9e861d3115d20a351ed8e03d6fc7d7256d1cee490746539d475" in rows["Tiny CPU run.py bfloat16 DPO training smoke"][2]
    assert "representative preference datasets" in rows["Tiny CPU run.py bfloat16 DPO training smoke"][4]
    assert "return_dict=true" in rows["Tiny CPU run.py supervised_instruct training smoke"][2]
    assert "tokenizer.chat_template is not set" in rows["Tiny CPU run.py supervised_instruct training smoke"][2]
    assert "mohawk_tiny_supervised_instruct_cpu_run/save" in rows["Tiny CPU run.py supervised_instruct training smoke"][2]
    assert "8c0b64ea9bc99724503935f143d7a66d93e990a3f62c6948948c7f41b2c78294" in rows["Tiny CPU run.py supervised_instruct training smoke"][2]
    assert "tiny_cpu_supervised_instruct_teacher_logits.yaml" in rows["Tiny CPU run.py supervised_instruct teacher-logits supervision smoke"][2]
    assert "generate False" in rows["Tiny CPU run.py supervised_instruct teacher-logits supervision smoke"][2]
    assert "supervision True" in rows["Tiny CPU run.py supervised_instruct teacher-logits supervision smoke"][2]
    assert "response span" in rows["Tiny CPU run.py supervised_instruct teacher-logits supervision smoke"][2]
    assert "torch.float32" in rows["Tiny CPU run.py supervised_instruct teacher-logits supervision smoke"][2]
    assert "3b6aa68bc5a25a6bfe23ec768b38f643044d110cd369468f1e453c49d1a63f8c" in rows["Tiny CPU run.py supervised_instruct teacher-logits supervision smoke"][2]
    assert "tiny_cpu_supervised_instruct_generate.yaml" in rows["Tiny CPU run.py supervised_instruct generate-supervision smoke"][2]
    assert "generate True" in rows["Tiny CPU run.py supervised_instruct generate-supervision smoke"][2]
    assert "supervision True" in rows["Tiny CPU run.py supervised_instruct generate-supervision smoke"][2]
    assert "generation_max_new_tokens 2" in rows["Tiny CPU run.py supervised_instruct generate-supervision smoke"][2]
    assert "teacher wrapper generation path" in rows["Tiny CPU run.py supervised_instruct generate-supervision smoke"][2]
    assert "a298f6bfb00563b3e932c3de47022f97e241c1fcb47bdd9d1e4d52df94dc5198" in rows["Tiny CPU run.py supervised_instruct generate-supervision smoke"][2]
    assert "tiny_cpu_bfloat16_supervised_instruct.yaml" in rows["Tiny CPU run.py bfloat16 supervised_instruct training smoke"][2]
    assert "DistillConfig.type: supervised_instruct" in rows["Tiny CPU run.py bfloat16 supervised_instruct training smoke"][2]
    assert "generate False" in rows["Tiny CPU run.py bfloat16 supervised_instruct training smoke"][2]
    assert "supervision False" in rows["Tiny CPU run.py bfloat16 supervised_instruct training smoke"][2]
    assert "collate_type=instruction" in rows["Tiny CPU run.py bfloat16 supervised_instruct training smoke"][2]
    assert "return_dict=true" in rows["Tiny CPU run.py bfloat16 supervised_instruct training smoke"][2]
    assert "input_ids" in rows["Tiny CPU run.py bfloat16 supervised_instruct training smoke"][2]
    assert "response_ids" in rows["Tiny CPU run.py bfloat16 supervised_instruct training smoke"][2]
    assert "TrainConfig.model_dtype bfloat16" in rows["Tiny CPU run.py bfloat16 supervised_instruct training smoke"][2]
    assert "TeacherConfig.model_dtype bfloat16" in rows["Tiny CPU run.py bfloat16 supervised_instruct training smoke"][2]
    assert "12/12 keys" in rows["Tiny CPU run.py bfloat16 supervised_instruct training smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py bfloat16 supervised_instruct training smoke"][2]
    assert "9c5e08cfc440165badaf7844aeb9fcfb10c8100980be9e580d51909aa8d01d6e" in rows["Tiny CPU run.py bfloat16 supervised_instruct training smoke"][2]
    assert "bfloat16 teacher-logits/generation-supervision modes" in rows["Tiny CPU run.py bfloat16 supervised_instruct training smoke"][4]
    assert "representative instruction datasets" in rows["Tiny CPU run.py bfloat16 supervised_instruct training smoke"][4]
    assert "torchrun --standalone --nproc_per_node=1" in rows["Tiny CPU torchrun single-process training smoke"][2]
    assert "scheduler _step_count == 4" in rows["Tiny CPU run.py checkpoint resume smoke"][2]
    assert "hash different from the source checkpoint" in rows["Tiny CPU run.py checkpoint resume smoke"][2]
    assert "tiny_cpu_mixed_precision_resume.yaml" in rows["Tiny CPU run.py mixed-precision checkpoint resume smoke"][2]
    assert "TrainConfig.mixed_precision: true" in rows["Tiny CPU run.py mixed-precision checkpoint resume smoke"][2]
    assert "TeacherConfig.mixed_precision: true" in rows["Tiny CPU run.py mixed-precision checkpoint resume smoke"][2]
    assert "scheduler step 2 to 4" in rows["Tiny CPU run.py mixed-precision checkpoint resume smoke"][2]
    assert "n_batches: 4" in rows["Tiny CPU run.py mixed-precision checkpoint resume smoke"][2]
    assert "db56f67881c62d6bcb076b67adb34e38fea076855be99c7c0d928c07272e1e8a" in rows["Tiny CPU run.py mixed-precision checkpoint resume smoke"][2]
    assert "3375940e29d65712f3426bf2cac70ff85c2365d1b8a590c6554d1edd9206f163" in rows["Tiny CPU run.py mixed-precision checkpoint resume smoke"][2]
    assert "DDP/FSDP resume" in rows["Tiny CPU run.py mixed-precision checkpoint resume smoke"][4]
    assert "tiny_cpu_bfloat16_resume.yaml" in rows["Tiny CPU run.py bfloat16 checkpoint resume smoke"][2]
    assert "TrainConfig.model_dtype: bfloat16" in rows["Tiny CPU run.py bfloat16 checkpoint resume smoke"][2]
    assert "TeacherConfig.model_dtype: bfloat16" in rows["Tiny CPU run.py bfloat16 checkpoint resume smoke"][2]
    assert "TrainConfig.mixed_precision: true" in rows["Tiny CPU run.py bfloat16 checkpoint resume smoke"][2]
    assert "TeacherConfig.mixed_precision: true" in rows["Tiny CPU run.py bfloat16 checkpoint resume smoke"][2]
    assert "bfloat16 AMP fallback" in rows["Tiny CPU run.py bfloat16 checkpoint resume smoke"][2]
    assert "scheduler step 2 to 4" in rows["Tiny CPU run.py bfloat16 checkpoint resume smoke"][2]
    assert "n_batches: 4" in rows["Tiny CPU run.py bfloat16 checkpoint resume smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py bfloat16 checkpoint resume smoke"][2]
    assert "10057436f48a3989cee46b70fb6798a05e8fdb7bd65d08b2b8f6f207075c7ce2" in rows["Tiny CPU run.py bfloat16 checkpoint resume smoke"][2]
    assert "a2943247863988b07f553af2359eaaab936674d8c76148ffa7801a27390b4754" in rows["Tiny CPU run.py bfloat16 checkpoint resume smoke"][2]
    assert "CUDA bfloat16" in rows["Tiny CPU run.py bfloat16 checkpoint resume smoke"][4]
    assert "DDP/FSDP resume" in rows["Tiny CPU run.py bfloat16 checkpoint resume smoke"][4]
    assert "configs/smoke/tiny_cuda_resume.yaml" in rows["Tiny CUDA run.py checkpoint resume smoke"][2]
    assert "tiny_cuda_supervised/save" in rows["Tiny CUDA run.py checkpoint resume smoke"][2]
    assert "source scheduler 2/2" in rows["Tiny CUDA run.py checkpoint resume smoke"][2]
    assert "resumed scheduler 4/4" in rows["Tiny CUDA run.py checkpoint resume smoke"][2]
    assert "287b32e35195cb73a9a0f35493802b98f510bff5269543d6ca721750a8a2bb46" in rows["Tiny CUDA run.py checkpoint resume smoke"][2]
    assert "17e40ae4debe2dac0f42d643bc662fc4618f921b106a72c80d542a30bcad046f" in rows["Tiny CUDA run.py checkpoint resume smoke"][2]
    assert "DDP/FSDP resume" in rows["Tiny CUDA run.py checkpoint resume smoke"][4]
    assert "configs/smoke/tiny_cuda_mixed_precision_resume.yaml" in rows["Tiny CUDA run.py mixed-precision checkpoint resume smoke"][2]
    assert "tiny_cuda_mixed_precision/save" in rows["Tiny CUDA run.py mixed-precision checkpoint resume smoke"][2]
    assert "TrainConfig.mixed_precision: true" in rows["Tiny CUDA run.py mixed-precision checkpoint resume smoke"][2]
    assert "source scheduler 2/2" in rows["Tiny CUDA run.py mixed-precision checkpoint resume smoke"][2]
    assert "resumed scheduler 4/4" in rows["Tiny CUDA run.py mixed-precision checkpoint resume smoke"][2]
    assert "bd20786dba84dfb0975db85ea79cebeb1929e1658318fdbfa6e0d18ec91cb42f" in rows["Tiny CUDA run.py mixed-precision checkpoint resume smoke"][2]
    assert "6e23c54372d768965deab8fc8442d8e842938a52f7c858075d7411d25fd82f4e" in rows["Tiny CUDA run.py mixed-precision checkpoint resume smoke"][2]
    assert "DDP/FSDP resume" in rows["Tiny CUDA run.py mixed-precision checkpoint resume smoke"][4]
    assert "LoadConfig.dataloader.path" in rows["Tiny CPU run.py public HFDataset dataloader resume smoke"][2]
    assert "tiny_cpu_hfdata_dataloader_resume.yaml" in rows["Tiny CPU run.py public HFDataset dataloader resume smoke"][2]
    assert "Starting NeelNanda/pile-10k from index 2" in rows["Tiny CPU run.py public HFDataset dataloader resume smoke"][2]
    assert "_index: 4" in rows["Tiny CPU run.py public HFDataset dataloader resume smoke"][2]
    assert "1f44aff34235041ac68bc914800030c0f2e23a8e50eaf263e9664cd001cee4d5" in rows["Tiny CPU run.py public HFDataset dataloader resume smoke"][2]
    assert "model/optimizer/scheduler resume together with dataloader state" in rows["Tiny CPU run.py public HFDataset dataloader resume smoke"][4]
    assert "tiny_cpu_hfdata_full_resume.yaml" in rows["Tiny CPU run.py public HFDataset full-resume smoke"][2]
    assert "model, optimizer, scheduler, and dataloader" in rows["Tiny CPU run.py public HFDataset full-resume smoke"][2]
    assert "TrainConfig.n_tokens: 32" in rows["Tiny CPU run.py public HFDataset full-resume smoke"][2]
    assert "Starting NeelNanda/pile-10k from index 2" in rows["Tiny CPU run.py public HFDataset full-resume smoke"][2]
    assert "12/12 student checkpoint keys" in rows["Tiny CPU run.py public HFDataset full-resume smoke"][2]
    assert "29/29 cached" in rows["Tiny CPU run.py public HFDataset full-resume smoke"][2]
    assert "scheduler step 2 to 4" in rows["Tiny CPU run.py public HFDataset full-resume smoke"][2]
    assert "_index: 5" in rows["Tiny CPU run.py public HFDataset full-resume smoke"][2]
    assert "00376f568ca2fdb6abd25f26a8f66ddff1370592787f5bfb518632116cb97eee" in rows["Tiny CPU run.py public HFDataset full-resume smoke"][2]
    assert "model/optimizer/scheduler resume together with dataloader state" in rows["Tiny CPU run.py public HFDataset full-resume smoke"][4]
    assert "tiny_cpu_bfloat16_hfdata_full_resume.yaml" in rows["Tiny CPU run.py bfloat16 public HFDataset full-resume smoke"][2]
    assert "model, optimizer, scheduler, and dataloader" in rows["Tiny CPU run.py bfloat16 public HFDataset full-resume smoke"][2]
    assert "TrainConfig.n_tokens: 32" in rows["Tiny CPU run.py bfloat16 public HFDataset full-resume smoke"][2]
    assert "TrainConfig.model_dtype: bfloat16" in rows["Tiny CPU run.py bfloat16 public HFDataset full-resume smoke"][2]
    assert "TeacherConfig.model_dtype: bfloat16" in rows["Tiny CPU run.py bfloat16 public HFDataset full-resume smoke"][2]
    assert "Starting NeelNanda/pile-10k from index 2" in rows["Tiny CPU run.py bfloat16 public HFDataset full-resume smoke"][2]
    assert "only the bfloat16 source checkpoint" in rows["Tiny CPU run.py bfloat16 public HFDataset full-resume smoke"][2]
    assert "12/12 student keys" in rows["Tiny CPU run.py bfloat16 public HFDataset full-resume smoke"][2]
    assert "29/29 cached" in rows["Tiny CPU run.py bfloat16 public HFDataset full-resume smoke"][2]
    assert "scheduler step 2 to 4" in rows["Tiny CPU run.py bfloat16 public HFDataset full-resume smoke"][2]
    assert "_index: 5" in rows["Tiny CPU run.py bfloat16 public HFDataset full-resume smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py bfloat16 public HFDataset full-resume smoke"][2]
    assert "393eb34c4f142454c1134f918595da7a997d01732109da3df810e8b49fcc3e7d" in rows["Tiny CPU run.py bfloat16 public HFDataset full-resume smoke"][2]
    assert "CUDA bfloat16" in rows["Tiny CPU run.py bfloat16 public HFDataset full-resume smoke"][4]
    assert "TrainConfig.init_fn: lazy" in rows["Tiny CPU run.py lazy-load checkpoint training smoke"][2]
    assert "loaded 12/12 checkpoint keys" in rows["Tiny CPU run.py lazy-load checkpoint training smoke"][2]
    assert "mohawk_tiny_lazy_cpu_run/save" in rows["Tiny CPU run.py lazy-load checkpoint training smoke"][2]
    assert "tiny_cpu_mixed_precision_lazy_load.yaml" in rows["Tiny CPU run.py mixed-precision lazy-load checkpoint smoke"][2]
    assert "TrainConfig.init_fn: lazy" in rows["Tiny CPU run.py mixed-precision lazy-load checkpoint smoke"][2]
    assert "TrainConfig.mixed_precision: true" in rows["Tiny CPU run.py mixed-precision lazy-load checkpoint smoke"][2]
    assert "TeacherConfig.mixed_precision: true" in rows["Tiny CPU run.py mixed-precision lazy-load checkpoint smoke"][2]
    assert "loaded 12/12 checkpoint keys" in rows["Tiny CPU run.py mixed-precision lazy-load checkpoint smoke"][2]
    assert "mohawk_tiny_mixed_precision_lazy_cpu_run/save" in rows["Tiny CPU run.py mixed-precision lazy-load checkpoint smoke"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CPU run.py mixed-precision lazy-load checkpoint smoke"][2]
    assert "e9dd521c6fef505708167e2a5f048ca8feccefbdbbf482709c52b71e887f2016" in rows["Tiny CPU run.py mixed-precision lazy-load checkpoint smoke"][2]
    assert "3375940e29d65712f3426bf2cac70ff85c2365d1b8a590c6554d1edd9206f163" in rows["Tiny CPU run.py mixed-precision lazy-load checkpoint smoke"][2]
    assert "tiny_cpu_bfloat16_lazy_load.yaml" in rows["Tiny CPU run.py bfloat16 lazy-load checkpoint smoke"][2]
    assert "TrainConfig.init_fn: lazy" in rows["Tiny CPU run.py bfloat16 lazy-load checkpoint smoke"][2]
    assert "TrainConfig.model_dtype: bfloat16" in rows["Tiny CPU run.py bfloat16 lazy-load checkpoint smoke"][2]
    assert "TeacherConfig.model_dtype: bfloat16" in rows["Tiny CPU run.py bfloat16 lazy-load checkpoint smoke"][2]
    assert "TrainConfig.mixed_precision: true" in rows["Tiny CPU run.py bfloat16 lazy-load checkpoint smoke"][2]
    assert "TeacherConfig.mixed_precision: true" in rows["Tiny CPU run.py bfloat16 lazy-load checkpoint smoke"][2]
    assert "loaded 12/12 checkpoint keys" in rows["Tiny CPU run.py bfloat16 lazy-load checkpoint smoke"][2]
    assert "bfloat16 AMP fallback" in rows["Tiny CPU run.py bfloat16 lazy-load checkpoint smoke"][2]
    assert "mohawk_tiny_bfloat16_lazy_cpu_run/save" in rows["Tiny CPU run.py bfloat16 lazy-load checkpoint smoke"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CPU run.py bfloat16 lazy-load checkpoint smoke"][2]
    assert "torch.bfloat16" in rows["Tiny CPU run.py bfloat16 lazy-load checkpoint smoke"][2]
    assert "a279c41e3c27638b1b658be50e25343e754d54d0c536b223808dc3b3115c38eb" in rows["Tiny CPU run.py bfloat16 lazy-load checkpoint smoke"][2]
    assert "a2943247863988b07f553af2359eaaab936674d8c76148ffa7801a27390b4754" in rows["Tiny CPU run.py bfloat16 lazy-load checkpoint smoke"][2]
    assert "tiny_cuda_lazy_load.yaml" in rows["Tiny CUDA run.py lazy-load checkpoint smoke"][2]
    assert "Tiny-CUDA-Lazy-Load-Smoke" in rows["Tiny CUDA run.py lazy-load checkpoint smoke"][2]
    assert "Slurm command" in rows["Tiny CUDA run.py lazy-load checkpoint smoke"][2]
    assert "job 688316" in rows["Tiny CUDA run.py lazy-load checkpoint smoke"][2]
    assert "lazy-initialized the student on meta" in rows["Tiny CUDA run.py lazy-load checkpoint smoke"][2]
    assert "loaded 12/12 student checkpoint keys" in rows["Tiny CUDA run.py lazy-load checkpoint smoke"][2]
    assert "cuda:0" in rows["Tiny CUDA run.py lazy-load checkpoint smoke"][2]
    assert "tiny_cuda_lazy_load/save" in rows["Tiny CUDA run.py lazy-load checkpoint smoke"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CUDA run.py lazy-load checkpoint smoke"][2]
    assert "torch.float32" in rows["Tiny CUDA run.py lazy-load checkpoint smoke"][2]
    assert "287b32e35195cb73a9a0f35493802b98f510bff5269543d6ca721750a8a2bb46" in rows["Tiny CUDA run.py lazy-load checkpoint smoke"][2]
    assert "52729570d2fecd2c714de7933b52ea805d1c1ab49d0c8f6cb02e27ca046360e1" in rows["Tiny CUDA run.py lazy-load checkpoint smoke"][2]
    assert "representative configs" in rows["Tiny CUDA run.py lazy-load checkpoint smoke"][4]
    assert "configs/smoke/tiny_cuda_ddp_lazy_load.yaml" in rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][2]
    assert "configs/smoke/tiny_cuda_fsdp_lazy_load.yaml" in rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][2]
    assert "Tiny-CUDA-DDP-Lazy-Load-Smoke" in rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][2]
    assert "Tiny-CUDA-FSDP-Lazy-Load-Smoke" in rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][2]
    assert "job 688353" in rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][2]
    assert "job 688357" in rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][2]
    assert "WORLD_SIZE=2" in rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][2]
    assert "rank 0 lazy-initialized the student" in rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][2]
    assert "loaded 12/12 student checkpoint keys" in rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][2]
    assert "broadcasted weights from rank 0" in rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][2]
    assert "FULL_SHARD" in rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][2]
    assert "scheduler _step_count == 2" in rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][2]
    assert "num_steps == 2" in rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][2]
    assert "all 12 tensors changed" in rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][2]
    assert "01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][2]
    assert "14d447df4d4d3c6bfff71734e78cb36c71909b38ffb41be77c8b698739e5401b" in rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][2]
    assert "hash_changed True" in rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][2]
    assert "multi-node distributed lazy init" in rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][4]
    assert "mixed-precision distributed lazy init" in rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][4]
    assert "representative configs" in rows["Tiny CUDA DDP/FSDP run.py lazy-load checkpoint smokes"][4]
    assert "configs/smoke/tiny_cuda_supervised.yaml" in rows["Single-process CUDA training"][2]
    assert "configs/smoke/tiny_cuda_default_runtime_supervised.yaml" in rows["Single-process CUDA training"][2]
    assert "/home/abick/storage/mohawk/venvs/ssm-cuda-20260626/bin/python" in rows["Single-process CUDA training"][2]
    assert "Tiny-CUDA-Default-Runtime-Supervised-Smoke" in rows["Single-process CUDA training"][2]
    assert "[TRAINING_WRAPPER] Device: cuda:0" in rows["Single-process CUDA training"][2]
    assert "BaseDataWrapper.close()" in rows["Single-process CUDA training"][2]
    assert "BaseDataGenerator" in rows["Single-process CUDA training"][2]
    assert "13 passed" in rows["Single-process CUDA training"][2]
    assert "optimizer state entries 12" in rows["Single-process CUDA training"][2]
    assert "scheduler _step_count == 2" in rows["Single-process CUDA training"][2]
    assert "287b32e35195cb73a9a0f35493802b98f510bff5269543d6ca721750a8a2bb46" in rows["Single-process CUDA training"][2]
    assert "WORLD_SIZE=1" in rows["Single-process CUDA torchrun training"][2]
    assert "287b32e35195cb73a9a0f35493802b98f510bff5269543d6ca721750a8a2bb46" in rows["Single-process CUDA torchrun training"][2]
    assert "does not validate CUDA, DDP, FSDP" in rows["Tiny CPU run.py supervised training smoke"][4]
    assert "configs/smoke/data_ddp" in rows["DDP multi-GPU training"][2]
    assert "rank 1 received an empty shard" in rows["DDP multi-GPU training"][2]
    assert "explicit cuda:<local_rank> device" in rows["DDP multi-GPU training"][2]
    assert "WORLD_SIZE=2" in rows["DDP multi-GPU training"][2]
    assert "wrapper_type: ddp" in rows["DDP multi-GPU training"][2]
    assert "teacher weight broadcast" in rows["DDP multi-GPU training"][2]
    assert "tiny_cuda_ddp_default_runtime_supervised.yaml" in rows["DDP multi-GPU training"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Supervised-Smoke" in rows["DDP multi-GPU training"][2]
    assert "job 690730" in rows["DDP multi-GPU training"][2]
    assert "/home/abick/storage/mohawk/venvs/ssm-cuda-20260626" in rows["DDP multi-GPU training"][2]
    assert "01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in rows["DDP multi-GPU training"][2]
    assert "separate tiny multi-node DDP/FSDP row" in rows["DDP multi-GPU training"][2]
    assert "default-runtime multi-node DDP" not in rows["DDP multi-GPU training"][4]
    assert "mixed-precision multi-node DDP" in rows["DDP multi-GPU training"][4]
    assert "bfloat16 multi-node DDP" in rows["DDP multi-GPU training"][4]
    assert "configs/smoke/tiny_cuda_fsdp_supervised.yaml" in rows["FSDP multi-GPU training"][2]
    assert "Sharding Strategy: FULL_SHARD" in rows["FSDP multi-GPU training"][2]
    assert "Activation Checkpointing: False" in rows["FSDP multi-GPU training"][2]
    assert "tiny_cuda_fsdp_default_runtime_supervised.yaml" in rows["FSDP multi-GPU training"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Supervised-Smoke" in rows["FSDP multi-GPU training"][2]
    assert "job 690735" in rows["FSDP multi-GPU training"][2]
    assert "/home/abick/storage/mohawk/venvs/ssm-cuda-20260626" in rows["FSDP multi-GPU training"][2]
    assert "01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in rows["FSDP multi-GPU training"][2]
    assert "separate tiny multi-node DDP/FSDP row" in rows["FSDP multi-GPU training"][2]
    assert "default-runtime multi-node FSDP" not in rows["FSDP multi-GPU training"][4]
    assert "mixed-precision multi-node FSDP" in rows["FSDP multi-GPU training"][4]
    assert "bfloat16 multi-node FSDP" in rows["FSDP multi-GPU training"][4]
    assert "configs/smoke/tiny_cuda_fsdp_activation_checkpointing.yaml" in rows["Real FSDP activation checkpointing"][2]
    assert "Activation Checkpointing: True" in rows["Real FSDP activation checkpointing"][2]
    assert "01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in rows["Real FSDP activation checkpointing"][2]
    assert "DDPTrainingWrapper previously ignored mixed_precision" in rows["DDP/FSDP mixed precision"][2]
    assert "mixed_precision False False" in rows["DDP/FSDP mixed precision"][2]
    assert "configs/smoke/tiny_cuda_ddp_mixed_precision.yaml" in rows["DDP/FSDP mixed precision"][2]
    assert "configs/smoke/tiny_cuda_fsdp_mixed_precision.yaml" in rows["DDP/FSDP mixed precision"][2]
    assert "tiny_cuda_ddp_default_runtime_mixed_precision.yaml" in rows["DDP/FSDP mixed precision"][2]
    assert "tiny_cuda_fsdp_default_runtime_mixed_precision.yaml" in rows["DDP/FSDP mixed precision"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Mixed-Precision-Smoke" in rows["DDP/FSDP mixed precision"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Mixed-Precision-Smoke" in rows["DDP/FSDP mixed precision"][2]
    assert "Slurm job 690963" in rows["DDP/FSDP mixed precision"][2]
    assert "Slurm job 690971" in rows["DDP/FSDP mixed precision"][2]
    assert "/home/abick/storage/mohawk/venvs/ssm-cuda-20260626" in rows["DDP/FSDP mixed precision"][2]
    assert "Mixed Precision: True" in rows["DDP/FSDP mixed precision"][2]
    assert "wrappers ddp ddp" in rows["DDP/FSDP mixed precision"][2]
    assert "wrappers fsdp fsdp" in rows["DDP/FSDP mixed precision"][2]
    assert "mixed-precision flags True True" in rows["DDP/FSDP mixed precision"][2]
    assert "01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in rows["DDP/FSDP mixed precision"][2]
    assert "mixed-precision DDP/FSDP resume" in rows["DDP/FSDP mixed precision"][4]
    assert "configs/smoke/tiny_cuda_ddp_resume.yaml" in rows["Tiny CUDA DDP checkpoint resume smoke"][2]
    assert "configs/smoke/tiny_cuda_ddp_default_runtime_resume.yaml" in rows["Tiny CUDA DDP checkpoint resume smoke"][2]
    assert "DDP optimizer/scheduler loading" in rows["Tiny CUDA DDP checkpoint resume smoke"][2]
    assert "WORLD_SIZE=2" in rows["Tiny CUDA DDP checkpoint resume smoke"][2]
    assert "source scheduler 1/1 to resumed scheduler 2/2" in rows["Tiny CUDA DDP checkpoint resume smoke"][2]
    assert "wrappers ddp ddp" in rows["Tiny CUDA DDP checkpoint resume smoke"][2]
    assert "Slurm job 691369" in rows["Tiny CUDA DDP checkpoint resume smoke"][2]
    assert "Tiny-CUDA-DDP-Default-Runtime-Resume-Smoke" in rows["Tiny CUDA DDP checkpoint resume smoke"][2]
    assert "tiny_cuda_ddp_default_runtime_supervised/save" in rows["Tiny CUDA DDP checkpoint resume smoke"][2]
    assert "tiny_cuda_ddp_default_runtime_resume/save" in rows["Tiny CUDA DDP checkpoint resume smoke"][2]
    assert "source optimizer state entries 0" in rows["Tiny CUDA DDP checkpoint resume smoke"][2]
    assert "target optimizer state entries 12" in rows["Tiny CUDA DDP checkpoint resume smoke"][2]
    assert "14d447df4d4d3c6bfff71734e78cb36c71909b38ffb41be77c8b698739e5401b" in rows["Tiny CUDA DDP checkpoint resume smoke"][2]
    assert "all 12 tensors changed" in rows["Tiny CUDA DDP checkpoint resume smoke"][2]
    assert "non-empty DDP optimizer-state restore" in rows["Tiny CUDA DDP checkpoint resume smoke"][4]
    assert "mixed-precision DDP resume" in rows["Tiny CUDA DDP checkpoint resume smoke"][4]
    assert "storage-backed default-runtime venv" in rows["Tiny CUDA DDP checkpoint resume smoke"][4]
    assert "configs/smoke/tiny_cuda_fsdp_resume.yaml" in rows["Tiny CUDA FSDP checkpoint resume smoke"][2]
    assert "configs/smoke/tiny_cuda_fsdp_default_runtime_resume.yaml" in rows["Tiny CUDA FSDP checkpoint resume smoke"][2]
    assert "configured FSDP scheduler total steps" in rows["Tiny CUDA FSDP checkpoint resume smoke"][2]
    assert "FULL_SHARD" in rows["Tiny CUDA FSDP checkpoint resume smoke"][2]
    assert "source scheduler 1/1 to resumed scheduler 2/2" in rows["Tiny CUDA FSDP checkpoint resume smoke"][2]
    assert "wrappers fsdp fsdp" in rows["Tiny CUDA FSDP checkpoint resume smoke"][2]
    assert "Slurm job 691375" in rows["Tiny CUDA FSDP checkpoint resume smoke"][2]
    assert "Tiny-CUDA-FSDP-Default-Runtime-Resume-Smoke" in rows["Tiny CUDA FSDP checkpoint resume smoke"][2]
    assert "tiny_cuda_fsdp_default_runtime_supervised/save" in rows["Tiny CUDA FSDP checkpoint resume smoke"][2]
    assert "tiny_cuda_fsdp_default_runtime_resume/save" in rows["Tiny CUDA FSDP checkpoint resume smoke"][2]
    assert "Model dtype: torch.float32" in rows["Tiny CUDA FSDP checkpoint resume smoke"][2]
    assert "Mixed Precision: False" in rows["Tiny CUDA FSDP checkpoint resume smoke"][2]
    assert "Activation Checkpointing: False" in rows["Tiny CUDA FSDP checkpoint resume smoke"][2]
    assert "source optimizer state entries 0" in rows["Tiny CUDA FSDP checkpoint resume smoke"][2]
    assert "target optimizer state entries 12" in rows["Tiny CUDA FSDP checkpoint resume smoke"][2]
    assert "14d447df4d4d3c6bfff71734e78cb36c71909b38ffb41be77c8b698739e5401b" in rows["Tiny CUDA FSDP checkpoint resume smoke"][2]
    assert "all 12 tensors changed" in rows["Tiny CUDA FSDP checkpoint resume smoke"][2]
    assert "non-empty FSDP optimizer-state restore" in rows["Tiny CUDA FSDP checkpoint resume smoke"][4]
    assert "mixed-precision FSDP resume" in rows["Tiny CUDA FSDP checkpoint resume smoke"][4]
    assert "storage-backed default-runtime venv" in rows["Tiny CUDA FSDP checkpoint resume smoke"][4]
    assert "tiny CUDA resume" in rows["Wrapper resume"][2]
    assert "tiny CUDA mixed-precision resume" in rows["Wrapper resume"][2]
    assert "tiny CUDA DDP checkpoint resume smoke" in rows["Wrapper resume"][2]
    assert "tiny CUDA FSDP checkpoint resume smoke" in rows["Wrapper resume"][2]
    assert "mixed-precision resume smoke" in rows["Wrapper resume"][2]
    assert "bfloat16 resume smoke" in rows["Wrapper resume"][2]
    assert "scheduler step 2 to 4" in rows["Wrapper resume"][2]
    assert "source scheduler 1/1 to resumed scheduler 2/2" in rows["Wrapper resume"][2]
    assert "default-runtime DDP/FSDP resume smokes" in rows["Wrapper resume"][2]
    assert "691369" in rows["Wrapper resume"][2]
    assert "691375" in rows["Wrapper resume"][2]
    assert "0 -> 12" in rows["Wrapper resume"][2]
    assert "representative/full-size checkpoint resume" in rows["Wrapper resume"][4]
    assert "non-empty distributed optimizer-state restore" in rows["Wrapper resume"][4]
    assert "meta-initialized student" in rows["Lazy/eager initialization"][2]
    assert "mixed-precision lazy-load smoke" in rows["Lazy/eager initialization"][2]
    assert "bfloat16 lazy-load smoke" in rows["Lazy/eager initialization"][2]
    assert "tiny CUDA lazy-load smoke" in rows["Lazy/eager initialization"][2]
    assert "tiny CUDA DDP/FSDP lazy-load smokes" in rows["Lazy/eager initialization"][2]
    assert "DDP rank-0 broadcast" in rows["Lazy/eager initialization"][2]
    assert "FSDP FULL_SHARD" in rows["Lazy/eager initialization"][2]
    assert "configs/Llama/8B/llama/hstates.yaml" in rows["Lazy/eager initialization"][2]
    assert "RoundRobinLoader -> HFDataset" in rows["Lazy/eager initialization"][2]
    assert "HuggingFaceFW/fineweb-edu" in rows["Lazy/eager initialization"][2]
    assert "before model initialization" in rows["Lazy/eager initialization"][2]
    assert "representative lazy/eager checkpoint init therefore remains resource-blocked" in rows["Lazy/eager initialization"][2]
    assert "required datasets and checkpoints accessible" in rows["Lazy/eager initialization"][4]
    assert "two real supervised training runs sequentially" in rows["Tiny CPU comma-separated run.py sequential training smoke"][2]
    assert "tiny_cpu_supervised.yaml,configs/smoke/tiny_cpu_eval_ppl.yaml" in rows["Tiny CPU comma-separated run.py train+eval callback chain smoke"][2]
    assert "EvalConfig -> eval_ppl" in rows["Tiny CPU comma-separated run.py train+eval callback chain smoke"][2]
    assert "loaded 12/12 eval teacher keys" in rows["Tiny CPU comma-separated run.py train+eval callback chain smoke"][2]
    assert "eval_score: 46137.343750" in rows["Tiny CPU comma-separated run.py train+eval callback chain smoke"][2]
    assert "latest_Perplexity.txt" in rows["Tiny CPU comma-separated run.py train+eval callback chain smoke"][2]
    assert "perplexity: 46137.34375" in rows["Tiny CPU comma-separated run.py train+eval callback chain smoke"][2]
    assert "e153eb131e5bbbf7f1e1ad9913dcfc6082fe5a8c5c95822768d10a58ba653687" in rows["Tiny CPU comma-separated run.py train+eval callback chain smoke"][2]
    assert "ada2086c37b7c885e663df2589d4aa4a318178e2163906772a2efd4eaa8c36b0" in rows["Tiny CPU comma-separated run.py train+eval callback chain smoke"][2]
    assert "external eval commands" in rows["Tiny CPU comma-separated run.py train+eval callback chain smoke"][4]
    assert "tiny_cuda_comma_supervised.yaml,configs/smoke/tiny_cuda_comma_eval_ppl.yaml" in rows["Tiny CUDA comma-separated run.py train+eval callback chain smoke"][2]
    assert "Slurm job 689748" in rows["Tiny CUDA comma-separated run.py train+eval callback chain smoke"][2]
    assert "cuda:0" in rows["Tiny CUDA comma-separated run.py train+eval callback chain smoke"][2]
    assert "eval_score: 46136.378906" in rows["Tiny CUDA comma-separated run.py train+eval callback chain smoke"][2]
    assert "latest_Perplexity.txt" in rows["Tiny CUDA comma-separated run.py train+eval callback chain smoke"][2]
    assert "perplexity: 46136.37890625" in rows["Tiny CUDA comma-separated run.py train+eval callback chain smoke"][2]
    assert "4c5a6e0dcdd8b17cba44e0bd7b364518dca115d046f8753ae058cea651b88f29" in rows["Tiny CUDA comma-separated run.py train+eval callback chain smoke"][2]
    assert "external evaluation commands" in rows["Tiny CUDA comma-separated run.py train+eval callback chain smoke"][4]
    assert rows["Representative comma-separated config execution"][3] == "BLOCKED_RESOURCE"
    assert "tiny CUDA train+eval callback smokes" in rows["Representative comma-separated config execution"][2]
    assert "parsed and loaded both representative config entries" in rows["Representative comma-separated config execution"][2]
    assert "HuggingFaceTB/finemath" in rows["Representative comma-separated config execution"][2]
    assert "before model init, teacher load, optimizer construction, training, or execution of the second config" in rows["Representative comma-separated config execution"][2]
    assert "do not validate representative sequential execution" in rows["Representative comma-separated config execution"][2]
    assert "configs/Llama/8B/llama/supervised.yaml" in rows["Llama 8B representative config loadability"][2]
    assert "missing ./configs/components/llama.yaml" in rows["Llama 8B representative config loadability"][2]
    assert "configs/Llama/8B/bases/_matrices.yaml" in rows["Llama 8B representative config loadability"][2]
    assert "LlamaBlock" in rows["Llama 8B representative config loadability"][2]
    assert "This is only config loadability" in rows["Llama 8B representative config loadability"][4]
    assert "configs/Phi/1.5B/phi/supervised.yaml" in rows["Phi 1.5B representative config loadability"][2]
    assert "loaded without TrainDataConfig" in rows["Phi 1.5B representative config loadability"][2]
    assert "missing ./configs/Phi/1.5B/base/test15_phi_from_phi/components.yaml" in rows["Phi 1.5B representative config loadability"][2]
    assert "legacy TrainConfig.DataConfig" in rows["Phi 1.5B representative config loadability"][2]
    assert "loader: C4DataLoader" in rows["Phi 1.5B representative config loadability"][2]
    assert "TrainDataConfig: [HFDataset, Tokenize, PackingDataLoader, TorchDataLoader]" in rows["Phi 1.5B representative config loadability"][2]
    assert "This is only config loadability" in rows["Phi 1.5B representative config loadability"][4]
    assert "configs/Llama/1B/llama/matrices.yaml" in rows["Additional Llama/Qwen2/Falcon representative config loadability"][2]
    assert "configs/Qwen2/1.5B/bases/_matrices.yaml" in rows["Additional Llama/Qwen2/Falcon representative config loadability"][2]
    assert "Falcon pure/matrices.yaml and pure/supervised.yaml loaded without ComponentsConfig" in rows["Additional Llama/Qwen2/Falcon representative config loadability"][2]
    assert "n_batches: 100" in rows["Additional Llama/Qwen2/Falcon representative config loadability"][2]
    assert "configs/Llama/3B/hybrid/adapter.yaml" in rows["Additional Llama/Qwen2/Falcon representative config loadability"][2]
    assert "This is only config loadability" in rows["Additional Llama/Qwen2/Falcon representative config loadability"][4]
    assert "configs/Llama/1B/hybrid/architecture_0.yaml" in rows["Generated hybrid architecture fragments"][2]
    assert "configs/Qwen2/1.5B/hybrid/architecture_10.yaml" in rows["Generated hybrid architecture fragments"][2]
    assert "ROPE_SCALING" in rows["Generated hybrid architecture fragments"][2]
    assert "BLOCK_NAME" in rows["Generated hybrid architecture fragments"][2]
    assert "configs/Llama/3B/hybrid/adapter.yaml" in rows["Generated hybrid architecture fragments"][2]
    assert "configs/Llama/1B/hybrid/mohawk_8.yaml" in rows["Generated hybrid architecture fragments"][2]
    assert "not standalone public entrypoint configs" in rows["Generated hybrid architecture fragments"][2]
    assert "Keep public docs pointing to wrapper configs only" in rows["Generated hybrid architecture fragments"][4]
    assert "Production job 770711" in rows["tools/hybrid_weights_transfer.py execution"][2]
    assert "339/339 teacher tensors" in rows["tools/hybrid_weights_transfer.py execution"][2]
    assert "40 q/k/v/o attention heads" in rows["tools/hybrid_weights_transfer.py execution"][2]
    assert "7c1b48a3b8665ee3f5d665520d79e59c0623a4fecc3252907787ff47b2eaed27" in rows["tools/hybrid_weights_transfer.py execution"][2]
    production_qwen = rows["Production Qwen2.5-1.5B ARCH40 hybrid transfer, optimizer update, and generation"]
    assert "1,799,668,560" in production_qwen[2]
    assert "Job 770115" in production_qwen[2]
    assert "Job 770711" in production_qwen[2]
    assert "job 771203" in production_qwen[2]
    assert "404/462 tensors changed" in production_qwen[2]
    assert "Job 771456" in production_qwen[2]
    assert "production Llama support" in production_qwen[4]
    production_llama = rows[
        "Production Llama-3.2-3B ARCH20 random-weight hybrid forward/backward"
    ]
    assert "production vocab 128256" in production_llama[2]
    assert "20 retained attention heads" in production_llama[2]
    assert "job 772052" in production_llama[2]
    assert "4,066,033,536" in production_llama[2]
    assert "354 finite gradient tensors" in production_llama[2]
    assert "gated pretrained weights/tokenizer" in production_llama[4]
    assert "configs/smoke/tiny_cpu_hybrid_transfer.yaml" in rows["Tiny CPU tools/hybrid_weights_transfer.py public teacher smoke"][2]
    assert "hf-internal-testing/tiny-random-LlamaForCausalLM" in rows["Tiny CPU tools/hybrid_weights_transfer.py public teacher smoke"][2]
    assert "DoubleBlockAdapter" in rows["Tiny CPU tools/hybrid_weights_transfer.py public teacher smoke"][2]
    assert "explicit --heads" in rows["Tiny CPU tools/hybrid_weights_transfer.py public teacher smoke"][2]
    assert "loaded 7 matching student keys" in rows["Tiny CPU tools/hybrid_weights_transfer.py public teacher smoke"][2]
    assert "21/21 teacher keys" in rows["Tiny CPU tools/hybrid_weights_transfer.py public teacher smoke"][2]
    assert "q/k/v/o max diffs 0.0" in rows["Tiny CPU tools/hybrid_weights_transfer.py public teacher smoke"][2]
    assert "config-seeded initialization" in rows["Tiny CPU tools/hybrid_weights_transfer.py public teacher smoke"][2]
    assert "f27c6a8cc74fa11e918d8169ece052ccb0489337a3b60d08f6b0cf4ed805d4b6" in rows["Tiny CPU tools/hybrid_weights_transfer.py public teacher smoke"][2]
    assert "does not validate SSM/Mamba transfer" in rows["Tiny CPU tools/hybrid_weights_transfer.py public teacher smoke"][4]
    assert "--model hf" in rows["Public tiny tools/benchmark_throughput.py CUDA smoke"][2]
    assert "--hf-model-name" in rows["Public tiny tools/benchmark_throughput.py CUDA smoke"][2]
    assert "--local_files_only" in rows["Public tiny tools/benchmark_throughput.py CUDA smoke"][2]
    assert "hf-internal-testing/tiny-random-LlamaForCausalLM" in rows["Public tiny tools/benchmark_throughput.py CUDA smoke"][2]
    assert "Hugging Face StaticCache" in rows["Public tiny tools/benchmark_throughput.py CUDA smoke"][2]
    assert "Batch size; Generated tokens; Throughput" in rows["Public tiny tools/benchmark_throughput.py CUDA smoke"][2]
    assert "asynchronous graph enqueue" in rows["Public tiny tools/benchmark_throughput.py CUDA smoke"][2]
    assert "Slurm job 764327" in rows["Public tiny tools/benchmark_throughput.py CUDA smoke"][2]
    assert "1; 128; 4484.36; 0.10" in rows["Public tiny tools/benchmark_throughput.py CUDA smoke"][2]
    production_throughput = rows["Production-size Codestral/Falcon/Llamba architecture throughput"]
    assert "Slurm job 762836" in production_throughput[2]
    assert "Slurm job 762994" in production_throughput[2]
    assert "Slurm job 763929" in production_throughput[2]
    assert "277.43, 531.49, 938.83" in production_throughput[2]
    assert "255.19, 475.21, 952.45" in production_throughput[2]
    assert "203.95, 354.15, 688.33" in production_throughput[2]
    assert "a0e121ebed3d2324c6d762b0e211a08d62583681" in production_throughput[2]
    assert "synchronized CUDA-event timing" in production_throughput[2]
    assert "random weights" in production_throughput[4]
    long_context = rows["Production-size long-context SSM architecture throughput"]
    long_context_evidence = long_context[2].lower()
    assert "num_last_tokens: 1" in long_context[2]
    assert "Slurm job 764932" in long_context[2]
    assert "10.724348s" in long_context[2]
    assert "3055.48" in long_context[2]
    assert "job 765057" in long_context_evidence
    assert "3.520817s" in long_context[2]
    assert "9306.93" in long_context[2]
    assert "job 766886" in long_context_evidence
    assert "718.051062s" in long_context[2]
    assert "job 767486" in long_context_evidence
    assert "7.905415s" in long_context[2]
    assert "1036.25" in long_context[2]
    assert "job 767682" in long_context_evidence
    assert "timed out" in long_context_evidence
    assert "Llamba 32K completion" in long_context[4]
    full_throughput = rows["Full tools/benchmark_throughput.py research workflow"]
    assert "get_llamba_model_cls" in full_throughput[2]
    assert "No module named 'cartesia_pytorch'" in full_throughput[2]
    assert "job 706431" in full_throughput[2]
    assert "LocalEntryNotFoundError" in full_throughput[2]
    assert "Mamba 2.3 runtime" in full_throughput[2]
    assert "config-only Llamba" in full_throughput[2]
    assert "full pretrained workflow remains resource-gated" in full_throughput[2]
    assert "gated Llama tokenizer" in full_throughput[4]


def test_docs_scope_default_and_ssm_requirements_installs():
    readme = (ROOT / "README.md").read_text()
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    ssm_requirements = [
        line
        for line in (ROOT / "requirements-ssm-cuda.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]

    assert ssm_requirements == [
        "mamba-ssm==2.3.2.post1",
        "causal-conv1d==1.6.2.post1",
        "quack-kernels==0.5.3",
    ]

    assert "python -m pip install -r requirements.txt" in readme
    assert "python -m pip install -r requirements-cpu.txt" in readme
    assert "python -m pip install --no-build-isolation -r requirements-ssm-cuda.txt" in readme
    assert "requirements-cpu.txt" in readme
    assert "requirements-ssm-cuda.txt" in readme
    assert "deliberately excludes the CUDA-built SSM/Mamba kernels" in readme
    assert "`mamba_ssm` and `causal_conv1d` were absent" in readme
    assert "storage-backed default-runtime venv using the same default\nmanifest" in readme
    assert "configs/smoke/tiny_cuda_default_runtime_supervised.yaml" in readme
    assert "Tiny-CUDA-Default-Runtime-Supervised-Smoke" in readme
    assert "Tiny-CUDA-DDP-Default-Runtime-Supervised-Smoke" in readme
    assert "Tiny-CUDA-FSDP-Default-Runtime-Supervised-Smoke" in readme
    assert "Tiny-CUDA-DDP-Default-Runtime-Mixed-Precision-Smoke" in readme
    assert "Tiny-CUDA-FSDP-Default-Runtime-Mixed-Precision-Smoke" in readme
    assert "Tiny-CUDA-DDP-Default-Runtime-BFloat16-Smoke" in readme
    assert "Tiny-CUDA-FSDP-Default-Runtime-BFloat16-Smoke" in readme
    assert "Tiny-CUDA-DDP-Default-Runtime-Resume-Smoke" in readme
    assert "Tiny-CUDA-FSDP-Default-Runtime-Resume-Smoke" in readme
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Gradient-Accumulation-Smoke" in readme
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Gradient-Accumulation-Smoke" in readme
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Mixed-Precision-Gradient-Accumulation-Smoke" in readme
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Mixed-Precision-Gradient-Accumulation-Smoke" in readme
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-BFloat16-Gradient-Accumulation-Smoke" in readme
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-BFloat16-Gradient-Accumulation-Smoke" in readme
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Resume-Smoke" in readme
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Resume-Smoke" in readme
    assert "one-node and multi-node DDP/FSDP\nmixed-precision, one-node and multi-node DDP/FSDP bfloat16" in readme
    assert "multi-node DDP/FSDP checkpoint-resume" in readme
    assert "two-node four-rank DDP/FSDP `eval_hstates`, `eval_ppl`, and\nbenchmark callback paths" in readme
    assert "multi-node\ndefault-runtime evaluation" not in readme
    assert "multi-node\ndefault-runtime training/evaluation" not in readme
    assert "representative/full-scale SSM kernels" in readme
    assert "distributed eval, or SSM kernels" not in readme
    assert "distributed eval/resume" not in readme
    assert "distributed eval/resume, bfloat16 paths" not in readme
    assert "/home/abick/storage/mohawk/venvs/ssm-cuda-20260626" in readme
    assert "/home/abick/storage/mohawk/venvs/ssm-cuda-mamba23-20260630" in readme
    assert "mamba-ssm==2.3.2.post1" in readme
    assert "causal-conv1d==1.6.2.post1" in readme
    assert "quack-kernels==0.5.3" in readme
    assert "Mamba 2.3 adds TileLang/Quack/CUTLASS\ndependencies" in readme
    assert "Mohawk guards direct component imports by making the local `utils` package\nexplicit" in readme
    assert "e3e6115fdb969130881fff5504c20b58398bf6a200a0d3359a5ae0ab9949c70e" in readme
    assert "completed a real two-step float32 `run.py` training smoke" in readme
    assert "representative Mamba 2.3 sequence lengths, multi-node Mamba 2.3" in readme
    assert "TORCH_CUDA_ARCH_LIST=10.0" in readme
    assert "Official compiler, CRT, NVVM, and CCCL wheels produced aarch64 `sm_100` wheels" in readme
    assert "ran finite bfloat16 fast\nforward/backward" in readme
    assert "tiny_cuda_doubleblock_discrete_mamba2_fast_supervised.yaml" in readme
    assert "tiny_cuda_ddp_doubleblock_discrete_mamba2_fast_supervised.yaml" in readme
    assert "tiny_cuda_fsdp_doubleblock_discrete_mamba2_fast_supervised.yaml" in readme
    assert "tiny_cuda_doubleblock_vanilla_discrete_mamba2_fast_supervised.yaml" in readme
    assert "0041dc4ab3b24ccff2d5368840b72f6499ca1385d6e1c63dfa96c22bc8588f6f" in readme
    assert "c50c1bfa8e56246bff13e2e3b51d72e118d11f4a74b8f5d0bc0b7405f6ef0ce6" in readme
    assert "tiny one-node DoubleBlock-family fast Mamba2 paths" in readme
    assert "tiny\nCUDA/DDP/FSDP paths currently have real smoke evidence" in readme
    assert "representative distributed or SSM training" in readme
    assert "full research dependency install" not in readme
    assert "full-requirements install" not in readme
    assert "requirements-pinned\nlm-eval Git revision remains `FAILED_UNFIXED`" not in readme
    assert "release-readiness-before-claiming-full-advertised-support" not in readme
    assert "PUBLIC_SUPPORT_MATRIX.md#release-readiness-gates" in readme
    assert "configs/Llama/8B/llama/supervised.yaml" in readme
    assert "configs/Llama/8B/llama/hstates.yaml" in readme
    assert "configs/Llama/8B/llama/matrices.yaml" in readme
    assert "CONSTANTS={'slurm_job_id': '0'}" in readme
    assert "only\nvalidates config loadability" in readme

    assert "python -m pip install -r requirements.txt" in contributing
    assert "python -m pip install -r requirements-cpu.txt" in contributing
    assert "python -m pip install --no-build-isolation -r requirements-ssm-cuda.txt" in contributing
    assert "requirements-cpu.txt" in contributing
    assert "requirements-ssm-cuda.txt" in contributing
    assert "103 passed, 1 skipped" in contributing
    assert "does\nnot install the CUDA-built SSM/Mamba kernels" in contributing
    assert "PASSED_END_TO_END" in contributing
    assert "FAILED_UNFIXED" in contributing
    assert "PUBLIC_SUPPORT_MATRIX.md#release-readiness-gates" in contributing
    assert "Do not use the CPU opt-in as a substitute for distributed validation" in readme
    assert "configs/smoke/tiny_cpu_gradient_accumulation.yaml" in readme
    assert "effective_batch_size: 2" in readme
    assert "accumulation_steps: 2" in readme
    assert "Tiny CUDA DDP/FSDP gradient-accumulation `no_sync` smokes" in readme
    assert "configs/smoke/tiny_cuda_ddp_gradient_accumulation.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_gradient_accumulation.yaml" in readme
    assert "Using DDP no_sync for gradient accumulation" in readme
    assert "Using FSDP no_sync for gradient\naccumulation" in readme
    assert "scheduler `_step_count == 2`" in readme
    assert "`num_steps == 2`" in readme
    assert "b75cb407af8b36f88a630ae2f4a9ddcc240edbf31548899075c4f8136c729e29" in readme
    assert "mixed-precision or bfloat16 no-sync" in readme
    assert "bfloat16 no-sync" in readme
    assert "Tiny CUDA DDP/FSDP mixed-precision gradient-accumulation `no_sync` smokes" in readme
    assert "configs/smoke/tiny_cuda_ddp_mixed_precision_gradient_accumulation.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_mixed_precision_gradient_accumulation.yaml" in readme
    assert "`mixed_precision: true`" in readme
    assert "Mixed Precision: True" in readme
    assert "69a62d39bd3931ba5ccfcdaa3a5d08e285a29a9fa359a15e16cb54646c6d5203" in readme
    assert "a8eb5ba1843710879037f98bb659cccf4d477f885337907172c057e1da62a97e" in readme
    assert "bfloat16 no-sync, which is covered separately" in readme
    assert "Tiny CUDA DDP/FSDP bfloat16 gradient-accumulation `no_sync` smokes" in readme
    assert "configs/smoke/tiny_cuda_ddp_bfloat16_gradient_accumulation.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_bfloat16_gradient_accumulation.yaml" in readme
    assert "`model_dtype: bfloat16`" in readme
    assert "Cannot use mixed precision with bfloat16" in readme
    assert "Model dtype:\ntorch.bfloat16" in readme
    assert "Mixed Precision: False" in readme
    assert "84819bd613ccd234514e3b6aff202d5abdc69674963d7941328e80bf3ddd7282" in readme
    assert "Tiny CUDA multi-node DDP/FSDP gradient-accumulation smokes across two Slurm\nnodes" in readme
    assert "configs/smoke/tiny_cuda_ddp_multinode_gradient_accumulation.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_multinode_gradient_accumulation.yaml" in readme
    assert "configs/smoke/data_ddp_multinode" in readme
    assert "--nodes=2 --ntasks-per-node=1 --gpus-per-node=4" in readme
    assert "--nnodes=$SLURM_NNODES" in readme
    assert "--node_rank=$SLURM_NODEID" in readme
    assert "nvl72d091-T05" in readme
    assert "nvl72d091-T07" in readme
    assert "WORLD_SIZE=4" in readme
    assert "effective_batch_size: 8" in readme
    assert "Total Grad Steps: 1" in readme
    assert "scheduler `_step_count == 1`" in readme
    assert "`num_steps == 1`" in readme
    assert "multi-node mixed precision" in readme
    assert "multi-node bfloat16" in readme
    assert "Multi-node resume, `eval_hstates`, `eval_ppl`, and benchmark are covered\nseparately below" in readme
    assert "Tiny CUDA multi-node DDP/FSDP mixed-precision gradient-accumulation smokes\nacross two Slurm nodes" in readme
    assert "configs/smoke/tiny_cuda_ddp_multinode_mixed_precision_gradient_accumulation.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_multinode_mixed_precision_gradient_accumulation.yaml" in readme
    assert "nvl72d112-T11" in readme
    assert "nvl72d112-T13" in readme
    assert "nvl72d037-T15" in readme
    assert "nvl72d037-T18" in readme
    assert "configs/smoke/tiny_cuda_ddp_default_runtime_multinode_mixed_precision_gradient_accumulation.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_default_runtime_multinode_mixed_precision_gradient_accumulation.yaml" in readme
    assert "Slurm jobs `692716` and `692810`" in readme
    assert "nvl72d125-T07" in readme
    assert "nvl72d125-T08" in readme
    assert "nvl72d105-T17" in readme
    assert "nvl72d105-T18" in readme
    assert "`mixed_precision: true`" in readme
    assert "Mixed Precision: True" in readme
    assert "tiny float32 AMP DDP/FSDP training across two Slurm nodes" in readme
    assert "Tiny CUDA multi-node DDP/FSDP bfloat16 gradient-accumulation smokes across two\nSlurm nodes" in readme
    assert "configs/smoke/tiny_cuda_ddp_multinode_bfloat16_gradient_accumulation.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_multinode_bfloat16_gradient_accumulation.yaml" in readme
    assert "nvl72d012-T01" in readme
    assert "nvl72d012-T02" in readme
    assert "configs/smoke/tiny_cuda_ddp_default_runtime_multinode_bfloat16_gradient_accumulation.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_default_runtime_multinode_bfloat16_gradient_accumulation.yaml" in readme
    assert "Slurm jobs `692872` and `692887`" in readme
    assert "nvl72d054-T01" in readme
    assert "nvl72d054-T02" in readme
    assert "nvl72d108-T15" in readme
    assert "nvl72d108-T18" in readme
    assert "tiny bfloat16 DDP/FSDP training across two Slurm nodes" in readme
    assert "`model_dtype: bfloat16`" in readme
    assert "bfloat16 AMP\nfallback warning" in readme
    assert "tiny_cuda_ddp_multinode_bfloat16_gradient_accumulation/save" in readme
    assert "tiny_cuda_fsdp_multinode_bfloat16_gradient_accumulation/save" in readme
    assert "0bc9e79b22d4fce8a926de7260a144c0a5a0ff09a86328af5f35777ab655af15" in readme
    assert "Tiny CUDA multi-node DDP/FSDP checkpoint resume smokes across two Slurm nodes" in readme
    assert "configs/smoke/tiny_cuda_ddp_multinode_resume.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_multinode_resume.yaml" in readme
    assert "nvl72d094-T12" in readme
    assert "nvl72d094-T14" in readme
    assert "nvl72d016-T15" in readme
    assert "nvl72d016-T18" in readme
    assert "tiny float32 DDP/FSDP checkpoint resume across two Slurm" in readme
    assert "`n_tokens: 128`" in readme
    assert "advance scheduler state from `1/1` to `2/2`" in readme
    assert "source\noptimizer state entries `0` and resumed optimizer state entries `12`" in readme
    assert "tiny_cuda_ddp_multinode_resume/save" in readme
    assert "tiny_cuda_fsdp_multinode_resume/save" in readme
    assert "configs/smoke/tiny_cuda_ddp_default_runtime_multinode_resume.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_default_runtime_multinode_resume.yaml" in readme
    assert "Slurm jobs `692584` and `692601`" in readme
    assert "nvl72d053-T10" in readme
    assert "nvl72d053-T13" in readme
    assert "nvl72d053-T17" in readme
    assert "nvl72d053-T18" in readme
    assert "tiny_cuda_ddp_default_runtime_multinode_gradient_accumulation/save" in readme
    assert "tiny_cuda_fsdp_default_runtime_multinode_gradient_accumulation/save" in readme
    assert "tiny_cuda_ddp_default_runtime_multinode_resume/save" in readme
    assert "tiny_cuda_fsdp_default_runtime_multinode_resume/save" in readme
    assert "5faf34670d9ba7799cb14ed17fb29ec3679c01a003c1d9100783181ae2d6b2d9" in readme
    assert "`hash_changed True`" in readme
    assert "mixed-precision or bfloat16 multi-node resume" in readme
    assert "Tiny CUDA multi-node DDP/FSDP `eval_hstates` callback smokes across two Slurm\nnodes" in readme
    assert "configs/smoke/tiny_cuda_ddp_multinode_eval_hstates.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_multinode_eval_hstates.yaml" in readme
    assert "nvl72d026-T08" in readme
    assert "nvl72d026-T10" in readme
    assert "tiny float32 DDP/FSDP `EvalConfig -> eval_hstates`\ncallbacks" in readme
    assert "EvalConfig -> eval_hstates" in readme
    assert "`eval_at_start: true`" in readme
    assert "`eval_at_end: true`" in readme
    assert "`save_latest: true`" in readme
    assert "`n_batches: 1`" in readme
    assert "`hstates_distance: 1.593651`" in readme
    assert "tiny_cuda_ddp_multinode_eval_hstates/save" in readme
    assert "tiny_cuda_fsdp_multinode_eval_hstates/save" in readme
    assert "latest score `{'eval_score':\n1.5936508178710938, 'hstates_distance': 1.5936508178710938}`" in readme
    assert "matching final/latest DDP/FSDP hash" in readme
    assert "smokes do not validate multi-node `eval_ppl`, which is covered separately\nbelow. Multi-node benchmark is covered separately below" in readme
    assert "mixed-precision or bfloat16 multi-node eval callbacks" in readme
    assert "representative eval data" in readme
    assert "Tiny CUDA multi-node DDP/FSDP `eval_ppl` callback smokes across two Slurm\nnodes" in readme
    assert "configs/smoke/tiny_cuda_ddp_multinode_eval_ppl.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_multinode_eval_ppl.yaml" in readme
    assert "nvl72d012-T03" in readme
    assert "nvl72d012-T04" in readme
    assert "nvl72d103-T01" in readme
    assert "nvl72d103-T02" in readme
    assert "tiny float32 DDP/FSDP `EvalConfig -> eval_ppl` callbacks" in readme
    assert "eval_score: 44002.273438" in readme
    assert "perplexity:\n44002.273438" in readme
    assert "accuracy: 0.000000" in readme
    assert "latest_Perplexity" in readme
    assert "tiny_cuda_ddp_multinode_eval_ppl/save" in readme
    assert "tiny_cuda_fsdp_multinode_eval_ppl/save" in readme
    assert "latest score `{'eval_score':\n44002.2734375, 'perplexity': 44002.2734375, 'accuracy': 0.0}`" in readme
    assert "Tiny CUDA multi-node DDP/FSDP benchmark callback smokes across two Slurm nodes" in readme
    assert "configs/smoke/tiny_cuda_ddp_multinode_eval_benchmark.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_multinode_eval_benchmark.yaml" in readme
    assert "tiny float32 DDP/FSDP `EvalConfig -> benchmark` callbacks" in readme
    assert "cached public\n`wikitext`" in readme
    assert "benchmark `limit: 4`" in readme
    assert "Slurm job `688414`" in readme
    assert "Slurm job `688406`" in readme
    assert "eval_score:\n172057.421054" in readme
    assert "wikitext 172057.421054" in readme
    assert "tiny_cuda_ddp_multinode_eval_benchmark/save" in readme
    assert "tiny_cuda_fsdp_multinode_eval_benchmark/save" in readme
    assert "matching DDP/FSDP hash\n`01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55`" in readme
    assert "mixed-precision or bfloat16 distributed benchmark callbacks" in readme
    assert "configs/smoke/tiny_cpu_wsd_scheduler.yaml" in readme
    assert "OptimizerConfig.scheduler.name: wsd" in readme
    assert "warmup_steps: 0.25" in readme
    assert "decay_steps: 0.25" in readme
    assert "one\nwarmup step and one decay step" in readme
    assert "mohawk_tiny_wsd_scheduler_cpu_run/save" in readme
    assert "`cosine`, `wsd_train`" in readme
    assert "configs/smoke/tiny_cpu_adam_optimizer.yaml" in readme
    assert "OptimizerConfig.optimizer: Adam" in readme
    assert "`torch.optim.Adam`" in readme
    assert "mohawk_tiny_adam_optimizer_cpu_run/save" in readme
    assert "DDP/FSDP optimizer wrapping" in readme
    assert "configs/smoke/tiny_cpu_optimize_weights_whitelist.yaml" in readme
    assert "configs/smoke/tiny_cpu_optimize_weights_blacklist.yaml" in readme
    assert "OptimizerConfig.optimize_weights" in readme
    assert "The whitelist smoke trains\nonly `lm_head.weight`" in readme
    assert "blacklist smoke freezes only `lm_head.weight`" in readme
    assert "mohawk_tiny_optimize_weights_whitelist_cpu_run/save" in readme
    assert "mohawk_tiny_optimize_weights_blacklist_cpu_run/save" in readme
    assert "configs/smoke/tiny_cpu_load_model_whitelist.yaml" in readme
    assert "configs/smoke/tiny_cpu_load_model_blacklist.yaml" in readme
    assert "LoadConfig.model` partial checkpoint" in readme
    assert "whitelist smoke loads\nonly `lm_head.weight`" in readme
    assert "blacklist smoke loads the eleven non-`lm_head` tensors" in readme
    assert "mohawk_tiny_load_model_whitelist_cpu_run/save" in readme
    assert "mohawk_tiny_load_model_blacklist_cpu_run/save" in readme
    assert "configs/smoke/tiny_cpu_load_model_rename.yaml" in readme
    assert "--rename-key lm_head.weight=renamed_lm_head.weight" in readme
    assert "mohawk_tiny_renamed_teacher_ckpt" in readme
    assert "renamed_lm_head: lm_head" in readme
    assert "mohawk_tiny_load_model_rename_cpu_run/save" in readme
    assert "LoadConfig.model.rename" in readme
    assert "configs/smoke/tiny_cpu_load_model_sequence.yaml" in readme
    assert "mohawk_tiny_sequence_renamed_lm_head_ckpt" in readme
    assert "two `LoadConfig.model` entries are applied in order" in readme
    assert "eleven backbone tensors from `/tmp/mohawk_tiny_teacher_ckpt`" in readme
    assert "distinct renamed `lm_head.weight`" in readme
    assert "mohawk_tiny_load_model_sequence_cpu_run/save" in readme
    assert "list ordering beyond this two-entry case" in readme
    assert "init_ssm_from_attention" in readme
    assert "strict missing-key failure modes" not in readme
    assert "multi-key rename collision behavior" in readme
    assert "configs/smoke/tiny_cpu_mixed_precision.yaml" in readme
    assert "mixed_precision: true" in readme
    assert "CUDA\nautocast/scaler behavior" in readme
    assert "configs/smoke/tiny_cpu_mixed_precision_gradient_accumulation.yaml" in readme
    assert "AMP/scaler mode and\nmicro-batch accumulation are enabled together" in readme
    assert "mohawk_tiny_mixed_precision_gradient_accumulation_cpu_run/save" in readme
    assert "DDP/FSDP `no_sync` with mixed\nprecision" in readme
    assert "configs/smoke/tiny_cpu_bfloat16_mixed_precision_fallback.yaml" in readme
    assert "model_dtype: bfloat16" in readme
    assert "Tiny CUDA/DDP/FSDP bfloat16 supervised smokes" in readme
    assert "configs/smoke/tiny_cuda_bfloat16_supervised.yaml" in readme
    assert "configs/smoke/tiny_cuda_ddp_bfloat16_supervised.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_bfloat16_supervised.yaml" in readme
    assert "TrainConfig.model_dtype: bfloat16" in readme
    assert "TeacherConfig.model_dtype: bfloat16" in readme
    assert "supervised-training smokes only" in readme
    assert "AMP\nis disabled for bfloat16" in readme
    assert "configs/smoke/tiny_cpu_bfloat16_gradient_accumulation_fallback.yaml" in readme
    assert "micro-batch accumulation" in readme
    assert "mohawk_tiny_bfloat16_gradient_accumulation_cpu_run/save" in readme
    assert "configs/smoke/tiny_cpu_qwen2_bfloat16_gradient_accumulation_fallback.yaml" in readme
    assert "configs/smoke/tiny_cpu_phi_bfloat16_gradient_accumulation_fallback.yaml" in readme
    assert "configs/smoke/tiny_cpu_falcon_bfloat16_gradient_accumulation_fallback.yaml" in readme
    assert "mohawk_tiny_qwen2_bfloat16_gradient_accumulation_cpu_run/save" in readme
    assert "mohawk_tiny_phi_bfloat16_gradient_accumulation_cpu_run/save" in readme
    assert "mohawk_tiny_falcon_bfloat16_gradient_accumulation_cpu_run/save" in readme
    assert "fast Mamba kernels" in readme
    assert "configs/smoke/tiny_cuda_qwen2_bfloat16_eval_callbacks.yaml" in readme
    assert "configs/smoke/tiny_cuda_phi_bfloat16_eval_callbacks.yaml" in readme
    assert "configs/smoke/tiny_cuda_falcon_bfloat16_eval_callbacks.yaml" in readme
    assert "tiny_cuda_qwen2_teacher_ckpt" in readme
    assert "tiny_cuda_phi_teacher_ckpt" in readme
    assert "tiny_cuda_falcon_teacher_ckpt" in readme
    assert "Qwen2Model -> Qwen2Block -> Qwen2Attention" in readme
    assert "LlamaModel -> MambaPhi ->\nPhiAttention" in readme
    assert "LlamaModel -> FalconBlock -> FalconMambaMixer" in readme
    assert "latest_eval_hstates" in readme
    assert "latest_Perplexity" in readme
    assert "59874.144531" in readme
    assert "40134.855469" in readme
    assert "62943.968750" in readme
    assert "8414e24632d042ced981bc7f4457205e643195d41f9ff7d6b4a0f37da58d3a19" in readme
    assert "e84464330dba07b4ef35e0f538d923fd4c188cae2cb1458b6cb18f93b60c66f0" in readme
    assert "a0297d015d195ccda5cb633c16624df99e256053a09f14d31a140073bc73d790" in readme
    assert "Tiny CUDA DDP/FSDP Qwen2/Phi/Falcon bfloat16 eval callback smokes" in readme
    assert "configs/smoke/tiny_cuda_ddp_qwen2_bfloat16_eval_callbacks.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_qwen2_bfloat16_eval_callbacks.yaml" in readme
    assert "configs/smoke/tiny_cuda_ddp_phi_bfloat16_eval_callbacks.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_phi_bfloat16_eval_callbacks.yaml" in readme
    assert "configs/smoke/tiny_cuda_ddp_falcon_bfloat16_eval_callbacks.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_falcon_bfloat16_eval_callbacks.yaml" in readme
    assert "configs/smoke/data_ddp" in readme
    assert "_step_count == 1" in readme
    assert "69068.7421875" in readme
    assert "50082.56640625" in readme
    assert "53790.79296875" in readme
    assert "c95790cea5ec1fe21235972050d52293b981a21fb4f50bfd1acb3e9f308511ba" in readme
    assert "82d94929797662c9abc5b7ecb20962f82dfef0c41d252be6f5a4be8e31045e3b" in readme
    assert "ef48a53d4ba3941435096b197361619301b7560ef96322543ab0ece75b3c3dde" in readme
    assert "configs/smoke/tiny_cpu_qwen2_bfloat16_mixed_precision_fallback.yaml" in readme
    assert "tiny CPU Qwen2 path" in readme
    assert "mohawk_tiny_qwen2_bfloat16_fallback_cpu_run/save" in readme
    assert "configs/smoke/tiny_cpu_phi_bfloat16_mixed_precision_fallback.yaml" in readme
    assert "tiny CPU PhiAttention path" in readme
    assert "mohawk_tiny_phi_bfloat16_fallback_cpu_run/save" in readme
    assert "configs/smoke/tiny_cpu_falcon_bfloat16_mixed_precision_fallback.yaml" in readme
    assert "tiny CPU FalconBlock path" in readme
    assert "Transformers' sequential Mamba fallback" in readme
    assert "mohawk_tiny_falcon_bfloat16_fallback_cpu_run/save" in readme
    assert "Tiny CPU DiscreteMamba2 reference `run.py` smoke" in readme
    assert "configs/smoke/tiny_cpu_discrete_mamba2_ref_supervised.yaml" in readme
    assert "LayeredMambaLM -> LlamaModel -> LlamaBlock -> DiscreteMamba2" in readme
    assert "use_ref_impl: true" in readme
    assert "mohawk_tiny_discrete_mamba2_ref_cpu_run/save" in readme
    assert "d0c1a65b9aa684fda86f08e60addbaaaf22e28fac02faaa32fd433d7d852c633" in readme
    assert "A direct GB300 probe with `mamba-ssm==2.2.6.post3`" in readme
    assert "Tiny fast-kernel `run.py` smokes" in readme
    assert "fast path separately" in readme
    assert "Tiny two-node fast-kernel DDP/FSDP smokes" in readme
    assert "tiny_cuda_ddp_multinode_doubleblock_discrete_mamba2_fast_supervised.yaml" in readme
    assert "tiny_cuda_fsdp_multinode_doubleblock_discrete_mamba2_fast_supervised.yaml" in readme
    assert "Current-code Slurm jobs `747896` and `747933`" in readme
    assert "two local ranks per node (`WORLD_SIZE=4`)" in readme
    assert "same\nmodel SHA-256" in readme
    assert "0fe968211b13f8e38a20391db9fe7aca14698f159f215848ed9dbedaae792272" in readme
    assert "Slurm job `748021` also loaded the DDP checkpoint" in readme
    assert "Tokenizer discovery now examines each available local\n  config" in readme
    assert "Tiny CPU DoubleBlock DiscreteMamba2 reference `run.py` smokes" in readme
    assert "configs/smoke/tiny_cpu_doubleblock_discrete_mamba2_ref_supervised.yaml" in readme
    assert "configs/smoke/tiny_cpu_doubleblock_vanilla_discrete_mamba2_ref_supervised.yaml" in readme
    assert "configs/smoke/tiny_cpu_doubleblock_hymba_discrete_mamba2_ref_supervised.yaml" in readme
    assert "configs/smoke/tiny_cpu_doubleblock_merger_discrete_mamba2_ref_supervised.yaml" in readme
    assert "Tiny CUDA DoubleBlock DiscreteMamba2 reference `run.py` smokes" in readme
    assert "configs/smoke/tiny_cuda_doubleblock_discrete_mamba2_ref_supervised.yaml" in readme
    assert "configs/smoke/tiny_cuda_doubleblock_vanilla_discrete_mamba2_ref_supervised.yaml" in readme
    assert "configs/smoke/tiny_cuda_doubleblock_hymba_discrete_mamba2_ref_supervised.yaml" in readme
    assert "configs/smoke/tiny_cuda_doubleblock_merger_discrete_mamba2_ref_supervised.yaml" in readme
    assert "DoubleBlockAdapter/Vanilla/Hymba/Merger` paths with `DiscreteMamba2" in readme
    assert "artifacts/gpu_smoke/doubleblock_ref_checkpoints" in readme
    assert "Slurm jobs `689221`, `689237`, `689254`, and `689262`" in readme
    assert "teacher keys `19/19`, `17/17`, `23/23`, and `21/21` on `cuda:0`" in readme
    assert "1,612,596, 1,612,324, 1,612,660, and 1,612,628 parameters" in readme
    assert "tiny_cuda_doubleblock_discrete_mamba2_ref/save" in readme
    assert "tiny_cuda_doubleblock_vanilla_discrete_mamba2_ref/save" in readme
    assert "tiny_cuda_doubleblock_hymba_discrete_mamba2_ref/save" in readme
    assert "tiny_cuda_doubleblock_merger_discrete_mamba2_ref/save" in readme
    assert "optimizer state entries `19`, `17`, `23`, and `21`" in readme
    assert "26410b59df6bd130e8b87ca6ce4f35a4040fb7dad89edc6e4392afc47c85eb0c" in readme
    assert "688309cc0e531b8335568d7bcc5739087fcbf9ad82729bea0efa5711b1465053" in readme
    assert "668075227edf9868305ccbbcfcd287f81a51c3325acc7bbc3eb25efe57aca8fd" in readme
    assert "282710e0946b6830ef1d3026e4a456d572e797c2742911053293df786dd85459" in readme
    assert "Hymba tiny run changed 10 of 23 tensors" in readme
    assert "Tiny CUDA DDP/FSDP DoubleBlock DiscreteMamba2 reference `run.py` smokes" in readme
    assert "configs/smoke/tiny_cuda_ddp_doubleblock_discrete_mamba2_ref_supervised.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_doubleblock_discrete_mamba2_ref_supervised.yaml" in readme
    assert "configs/smoke/tiny_cuda_{ddp,fsdp}_doubleblock_vanilla_discrete_mamba2_ref_supervised.yaml" in readme
    assert "configs/smoke/tiny_cuda_{ddp,fsdp}_doubleblock_hymba_discrete_mamba2_ref_supervised.yaml" in readme
    assert "configs/smoke/tiny_cuda_{ddp,fsdp}_doubleblock_merger_discrete_mamba2_ref_supervised.yaml" in readme
    assert "configs/smoke/data_ddp" in readme
    assert "effective_batch_size: 2" in readme
    assert "n_tokens: 32" in readme
    assert "DDP jobs `689318`, `689363`, `689378`, and `689404`" in readme
    assert "FSDP jobs `689324`, `689371`, `689396`, and `689411`" in readme
    assert "wrapper `ddp`, `NO_SHARD`" in readme
    assert "wrapper `fsdp`, `FULL_SHARD`" in readme
    assert "tiny_cuda_ddp_doubleblock*_discrete_mamba2_ref/save" in readme
    assert "tiny_cuda_fsdp_doubleblock*_discrete_mamba2_ref/save" in readme
    assert "86be1ef437abce55cc58f7bffd6f3fdeecab0639a4014967314935dc30041bf4" in readme
    assert "ebd61d3b2f1d789631a25612b27c51ca0ed8a5ef0ab516e32aa54a85fde2df8a" in readme
    assert "9ec928a4d5863363f05235ba63eb7a065f3152f4e5a0867f34feb85217614d98" in readme
    assert "12a16f0b147becab00aa2e6e3a69895368f215bb3c871b0e903c21e4faf2f6e7" in readme
    assert "/home/abick/storage/mohawk/artifacts/gpu_smoke" in readme
    assert "/home/abick/mohawk/artifacts/gpu_smoke` is a symlink" in readme
    assert "multi-node distributed hybrid\ntraining" in readme
    assert "LayeredMambaLM -> LlamaModel -> DoubleBlock*" in readme
    assert "`mixer1` as\n`DiscreteMamba2 use_ref_impl: true` and `mixer2` as `LlamaAttention`" in readme
    assert "mohawk_tiny_doubleblock_discrete_mamba2_ref_cpu_run/save" in readme
    assert "mohawk_tiny_doubleblock_vanilla_discrete_mamba2_ref_cpu_run/save" in readme
    assert "mohawk_tiny_doubleblock_hymba_discrete_mamba2_ref_cpu_run/save" in readme
    assert "mohawk_tiny_doubleblock_merger_discrete_mamba2_ref_cpu_run/save" in readme
    assert "fee4e7c14156ce0404efa63f631a89d6b1effb895f4f5c77e40696a4035ddc46" in readme
    assert "1c779255a7d2b4920e3c09d87f1e51efc6f7a388ff44c5bd4f513d0bebb648a7" in readme
    assert "36da178993d39692d8c1cba3163fdc78b0f6ad330c0bbd7fa6b5065297c8d449" in readme
    assert "56bb1d2f4226b9f1fa5ce9b945052204e4254a8bc9aeebdddae868e75525b3cc" in readme
    assert "Hymba tiny run changed 17 of 23 tensors" in readme
    assert "production Qwen/Llama hybrid configs" in readme
    assert "transfer tooling" in readme
    assert "configs/smoke/tiny_cpu_compile_model.yaml" in readme
    assert "TrainConfig.compile_model: true" in readme
    assert "no `_orig_mod.` prefixes" in readme
    assert "DDP/FSDP compile paths" in readme
    assert "configs/smoke/tiny_cpu_teacher_compile_model.yaml" in readme
    assert "TeacherConfig.compile_model: true" in readme
    assert "mohawk_tiny_teacher_compile_model_cpu_run/save" in readme
    assert "configs/smoke/tiny_cpu_student_teacher_compile_model.yaml" in readme
    assert "mohawk_tiny_student_teacher_compile_model_cpu_run/save" in readme
    assert "configs/smoke/tiny_cpu_bfloat16_compile_model.yaml" in readme
    assert "model_dtype: bfloat16" in readme
    assert "mohawk_tiny_bfloat16_compile_model_cpu_run/save" in readme
    assert "uncompiled bfloat16 local model" in readme
    assert "configs/smoke/tiny_cpu_bfloat16_teacher_compile_model.yaml" in readme
    assert "mohawk_tiny_bfloat16_teacher_compile_model_cpu_run/save" in readme
    assert "configs/smoke/tiny_cpu_bfloat16_student_teacher_compile_model.yaml" in readme
    assert "mohawk_tiny_bfloat16_student_teacher_compile_model_cpu_run/save" in readme
    assert "torchrun --standalone --nproc_per_node=2" in readme
    assert "Mohawk training requires CUDA" in readme
    assert "configs/smoke/tiny_cuda_supervised.yaml" in readme
    assert "Tiny CUDA supervised smoke on a Slurm GPU allocation" in readme
    assert "/home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun" in readme
    assert "/home/abick/nemotron_abick/conda/envs/fla-raven/bin/python run.py" in readme
    assert "Device: cuda:0" in readme
    assert "287b32e35195cb73a9a0f35493802b98f510bff5269543d6ca721750a8a2bb46" in readme
    assert "configs/smoke/tiny_cuda_mixed_precision.yaml" in readme
    assert "configs/smoke/tiny_cuda_resume.yaml" in readme
    assert "configs/smoke/tiny_cuda_mixed_precision_resume.yaml" in readme
    assert "bd20786dba84dfb0975db85ea79cebeb1929e1658318fdbfa6e0d18ec91cb42f" in readme
    assert "17e40ae4debe2dac0f42d643bc662fc4618f921b106a72c80d542a30bcad046f" in readme
    assert "6e23c54372d768965deab8fc8442d8e842938a52f7c858075d7411d25fd82f4e" in readme
    assert "configs/smoke/tiny_cuda_ddp_supervised.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_supervised.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_activation_checkpointing.yaml" in readme
    assert "configs/smoke/tiny_cuda_ddp_mixed_precision.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_mixed_precision.yaml" in readme
    assert "configs/smoke/tiny_cuda_ddp_resume.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_resume.yaml" in readme
    assert "configs/smoke/data_ddp" in readme
    assert "rank 1 received an\nempty shard" in readme
    assert "01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in readme
    assert "14d447df4d4d3c6bfff71734e78cb36c71909b38ffb41be77c8b698739e5401b" in readme
    assert "FULL_SHARD" in readme
    assert "scheduler `2/2`" in readme
    assert "mixed-precision distributed resume" in readme
    assert "honor `mixed_precision`" in readme
    assert "configs/smoke/tiny_cuda_supervised.yaml" in contributing
    assert "/home/abick/nemotron_abick/conda/envs/fla-raven/bin/torchrun" in contributing
    assert "/home/abick/nemotron_abick/conda/envs/fla-raven/bin/python run.py" in contributing
    assert "configs/smoke/tiny_cuda_mixed_precision.yaml" in contributing
    assert "configs/smoke/tiny_cuda_resume.yaml" in contributing
    assert "configs/smoke/tiny_cuda_mixed_precision_resume.yaml" in contributing
    assert "configs/smoke/tiny_cuda_ddp_supervised.yaml" in contributing
    assert "configs/smoke/tiny_cuda_fsdp_supervised.yaml" in contributing
    assert "configs/smoke/tiny_cuda_fsdp_activation_checkpointing.yaml" in contributing
    assert "configs/smoke/tiny_cuda_ddp_mixed_precision.yaml" in contributing
    assert "configs/smoke/tiny_cuda_fsdp_mixed_precision.yaml" in contributing
    assert "configs/smoke/tiny_cuda_ddp_resume.yaml" in contributing
    assert "configs/smoke/tiny_cuda_fsdp_resume.yaml" in contributing
    assert "tiny_cuda_comma_supervised.yaml,configs/smoke/tiny_cuda_comma_eval_ppl.yaml" in contributing
    assert "configs/smoke/data_ddp" in contributing
    assert "tiny one-node CUDA/DDP/FSDP paths" in contributing
    assert "mixed-precision\n     distributed resume" in contributing
    assert "configs/smoke/tiny_cpu_supervised.yaml,configs/smoke/tiny_cpu_eval_ppl.yaml" in readme
    assert "EvalConfig -> eval_ppl" in readme
    assert "mohawk_tiny_eval_ppl_cpu_run/save/latest_Perplexity" in readme
    assert "external\nevaluation commands" in readme
    assert "configs/smoke/tiny_cuda_comma_supervised.yaml,configs/smoke/tiny_cuda_comma_eval_ppl.yaml" in readme
    assert "tiny_cuda_comma_eval_ppl/save/latest_Perplexity" in readme
    assert "single-process CUDA sequence" in readme
    assert "configs/Llama/8B/llama/hstates.yaml,configs/Llama/8B/llama/hstates.yaml" in readme
    assert "does not reach model initialization, training, or the second config" in readme
    assert "configs/Llama/8B/llama/hstates.yaml" in readme
    assert "fails before model initialization" in readme
    assert "HuggingFaceFW/fineweb-edu" in readme
    assert "Raw `architecture_*.yaml` files under\n`hybrid/` are include fragments" in readme
    assert "`BLOCK_NAME` and `ROPE_SCALING`" in readme
    assert "not standalone public entrypoint\nconfigs" in readme
    assert "configs/smoke/tiny_cpu_hf_teacher_supervised.yaml" in readme
    assert "TeacherConfig.dir: sshleifer/tiny-gpt2" in readme
    assert "large or gated teachers" in readme
    assert "configs/smoke/tiny_cpu_public_hfdata_supervised.yaml" in readme
    assert "NeelNanda/pile-10k" in readme
    assert "HF_DATASETS_OFFLINE=1" in readme
    assert "configs/smoke/tiny_cpu_c4_streaming_supervised.yaml" in readme
    assert "allenai/c4" in readme
    assert "name: en.noclean" in readme
    assert "streaming: true" in readme
    assert "shuffle_buffer_size: 2" in readme
    assert "mohawk_tiny_c4_streaming_cpu_run/save" in readme
    assert "default\n`10000` shuffle buffer" in readme
    assert "configs/smoke/tiny_cpu_fineweb_edu_streaming_supervised_default_runtime.yaml" in readme
    assert "HuggingFaceFW/fineweb-edu" in readme
    assert "sample-100BT" in readme
    assert "/home/abick/storage/mohawk/fineweb_edu_streaming_smoke_default_runtime" in readme
    assert "exits with status 0" in readme
    assert "9e447f7af4d9d10fbd0bab04b7078d39bc6239f1681a73cdebc2c8c7762676d2" in readme
    assert "Tokenize.local_files_only: false" in readme
    assert "PyGILState_Release" in readme
    assert "429 Too Many Requests" in readme
    assert "without an available `HF_TOKEN`" in readme
    assert "default-runtime CPU result as a bounded tiny public-data smoke only" in readme
    assert "tiny_cuda_fineweb_edu_production_loader_supervised.yaml" in readme
    assert "runtime default\n`10000`" in readme
    assert "max_seq_len: 2048" in readme
    assert "Slurm job `706699`" in readme
    assert "55d4b4188067a61a6fcd6799b460e2059536b6b92f265c9e74e773ea96f600a1" in readme
    assert "tiny_cuda_ddp_fineweb_edu_production_loader_supervised.yaml" in readme
    assert "tiny_cuda_fsdp_fineweb_edu_production_loader_supervised.yaml" in readme
    assert "corrected optimizer-update retry hit Hugging Face `429 Too Many Requests`" in readme
    assert "Legacy `C4DataLoader` note" in readme
    assert "data_dir: <path-to-c4-dataset>" in readme
    assert "Those references have been removed or replaced by\ncurrent public `HFDataset` C4 configs" in readme
    assert "No module named 'streaming'" in readme
    assert "configs/smoke/tiny_cpu_random_data_supervised.yaml" in readme
    assert "RandomDataLoader -> CycleDataLoader -> run.py" in readme
    assert "TorchDataLoader.batch_size: 1" in readme
    assert "mohawk_tiny_random_data_cpu_run/save" in readme
    assert "synthetic random-token path" in readme
    assert "TorchDataLoader wrapping" in readme
    assert "configs/smoke/tiny_cpu_shuffle_loader_supervised.yaml" in readme
    assert "ShuffleLoader -> CycleDataLoader -> run.py" in readme
    assert "configs/smoke/tiny_shuffle_source_a.yaml" in readme
    assert "configs/smoke/tiny_shuffle_source_b.yaml" in readme
    assert "JSONIterableDataset -> Tokenize(collate_type=text) -> PaddingDataLoader ->\nTorchDataLoader" in readme
    assert "mohawk_tiny_shuffle_loader_cpu_run/save" in readme
    assert "large\nshuffle pools" in readme
    assert "source weighting" in readme
    assert "configs/smoke/tiny_cpu_conversation_collate.yaml" in readme
    assert "Tokenize(collate_type=conversation)" in readme
    assert "configs/smoke/data_conversation/tiny_conversation.json" in readme
    assert "configs/smoke/tiny_chat_template.jinja" in readme
    assert "mohawk_tiny_conversation_collate_cpu_run/save" in readme
    assert "chat-tokenizer default templates" in readme
    assert "configs/smoke/tiny_cpu_classic_collate.yaml" in readme
    assert "Tokenize(collate_type=classic)" in readme
    assert "configs/smoke/data_classic/tiny_classic.json" in readme
    assert "mohawk_tiny_classic_collate_cpu_run/save" in readme
    assert "jondurbin/airoboros-2.2" in readme
    assert "configs/smoke/tiny_cpu_kv_raw_packing.yaml" in readme
    assert "KVRetrieval -> Tokenize(collate_type=raw)" in readme
    assert "mohawk_tiny_kv_raw_packing_cpu_run/save" in readme
    assert "mixed real-data round-robin\nratios" in readme
    assert "configs/smoke/tiny_cpu_niah_dataset_supervised.yaml" in readme
    assert "NeedleInHaystackDataset -> CycleDataLoader -> TorchDataLoader -> run.py" in readme
    assert "local_files_only: true" in readme
    assert "mohawk_tiny_niah_dataset_cpu_run/save" in readme
    assert "essay\nhaystacks" in readme
    assert "configs/smoke/tiny_cpu_copying_task_supervised.yaml" in readme
    assert "CopyingTaskDataset -> CycleDataLoader -> TorchDataLoader -> run.py" in readme
    assert "max_seq_len: 24" in readme
    assert "mohawk_tiny_copying_task_cpu_run/save" in readme
    assert "long copying contexts" in readme
    assert "configs/smoke/tiny_cpu_sequential_loader_supervised.yaml" in readme
    assert "SequentialLoader -> PaddingDataLoader -> CycleDataLoader -> TorchDataLoader ->\nrun.py" in readme
    assert "configs/smoke/data_sequence_a/tiny_sequence_a.json" in readme
    assert "configs/smoke/data_sequence_b/tiny_sequence_b.json" in readme
    assert "mohawk_tiny_sequential_loader_cpu_run/save" in readme
    assert "multi-stage sequential curricula" in readme
    assert "configs/smoke/tiny_cpu_aggregation_loader_supervised.yaml" in readme
    assert "RoundRobinLoader -> PackingDataLoader -> AggregationDataLoader ->\nCycleDataLoader -> TorchDataLoader -> run.py" in readme
    assert "configs/smoke/data_aggregation_a/tiny_aggregation_a.json" in readme
    assert "configs/smoke/data_aggregation_b/tiny_aggregation_b.json" in readme
    assert "aggregation_size: 2" in readme
    assert "mohawk_tiny_aggregation_loader_cpu_run/save" in readme
    assert "large aggregation buffers" in readme
    assert "configs/smoke/tiny_cpu_hybrid_transfer.yaml" in readme
    assert "hf-internal-testing/tiny-random-LlamaForCausalLM" in readme
    assert "tiny\n  public-teacher attention-transfer path" in readme
    assert "Production-size architecture benchmarks also run" in readme
    assert "277.43,531.49,938.83" in readme
    assert "255.19,475.21,952.45" in readme
    assert "203.95,354.15,688.33" in readme
    assert "--llamba-config-only" in readme
    assert "constructs\n  random weights" in readme
    assert "mohawk_viz_llama" in readme
    assert "593bb14787d14db3ca6f631649aaf8cf3858cd9a455a8115e7253210a1328806" in readme
    assert "Llama-family CPU smoke" in readme
    assert "visualize_attention_cuda" in readme
    assert "db3c010fae59d7418ba0cd40ed91eb5c5a483e58d00d3682e991c46e5edd18ef" in readme
    assert "--device cuda" in readme
    assert "meta-llama/Llama-3.2-3B-Instruct" in readme
    assert "gated model files were not in the local cache" in readme
    assert "Representative custom Transformers checkpoint" in readme
    assert "--model-registration-module MODULE" in readme
    assert "AvivBick/raven-nslots256-topk64" in readme
    assert "1,697,254,264 bytes" in readme
    assert "424,303,808 parameters" in readme
    assert "Slurm jobs `751419`, `751508`, `751635`, and `752591`" in readme
    assert "97.02066040039062" in readme
    assert "61.629803" in readme
    assert "--mode hidden_similarity --layers 0,12,24" in readme
    assert "512 x 906" in readme
    assert "afaa1b151a6434f7bf24424241b236959062a085e15de242d30fda239a12e50f" in readme
    assert "Raven deliberately does not return attention tensors" in readme
    assert "not a\nquality benchmark" in readme
    assert "representative Mohawk\nhybrid checkpoint" in readme
    assert "configs/smoke/tiny_cpu_doubleblock_adapter_supervised.yaml" in readme
    assert "DoubleBlockVanilla" in readme
    assert "attention-only `run.py` smokes" in readme
    assert "configs/smoke/tiny_cpu_eval_hstates.yaml" in readme
    assert "eval_hstates` callback" in readme
    assert "wrapper-driven `eval_hstates` callback" in readme
    assert "configs/smoke/tiny_cpu_bfloat16_eval_hstates.yaml" in readme
    assert "bfloat16 wrapper-driven `eval_hstates` callback" in readme
    assert "mohawk_tiny_bfloat16_eval_hstates_cpu_run/save/latest_eval_hstates" in readme
    assert "Tiny CUDA/DDP/FSDP wrapper-driven `eval_hstates` callback smokes" in readme
    assert "configs/smoke/tiny_cuda_eval_hstates.yaml" in readme
    assert "configs/smoke/tiny_cuda_ddp_eval_hstates.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_eval_hstates.yaml" in readme
    assert "FSDP FULL_SHARD" in readme
    assert "torch.inference_mode()" in readme
    assert "torch.no_grad()" in readme
    assert "multi-node distributed eval" in readme
    assert "configs/smoke/tiny_cuda_bfloat16_eval_hstates.yaml" in readme
    assert "configs/smoke/tiny_cuda_ddp_bfloat16_eval_hstates.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_bfloat16_eval_hstates.yaml" in readme
    assert "tiny_cuda_bfloat16_eval_hstates/save/latest_eval_hstates" in readme
    assert "eval_score: 1.640625" in readme
    assert "eval_score: 1.664062" in readme
    assert "configs/smoke/tiny_cpu_bfloat16_hstates.yaml" in readme
    assert "bfloat16 `hstates` objective smoke" in readme
    assert "mohawk_tiny_bfloat16_hstates_cpu_run/save" in readme
    assert "configs/smoke/tiny_cpu_bfloat16_sequential_hstates.yaml" in readme
    assert "bfloat16 `sequential_hstates` objective smoke" in readme
    assert "mohawk_tiny_bfloat16_sequential_hstates_cpu_run/save" in readme
    assert "configs/smoke/tiny_cpu_bfloat16_matrices.yaml" in readme
    assert "bfloat16 `matrices` objective smoke" in readme
    assert "mohawk_tiny_bfloat16_matrices_cpu_run/save" in readme
    assert "configs/smoke/tiny_cpu_bfloat16_dpo.yaml" in readme
    assert "bfloat16 DPO objective smoke" in readme
    assert "mohawk_tiny_bfloat16_dpo_cpu_run/save" in readme
    assert "configs/smoke/tiny_cpu_supervised_instruct_teacher_logits.yaml" in readme
    assert "teacher-logits supervision smoke" in readme
    assert "supervision: true` and `generate: false" in readme
    assert "mohawk_tiny_supervised_instruct_teacher_logits_cpu_run/save" in readme
    assert "configs/smoke/tiny_cpu_supervised_instruct_generate.yaml" in readme
    assert "generate-supervision smoke" in readme
    assert "supervision: true` and `generate: true" in readme
    assert "generation_max_new_tokens: 2" in readme
    assert "mohawk_tiny_supervised_instruct_generate_cpu_run/save" in readme
    assert "configs/smoke/tiny_cpu_bfloat16_supervised_instruct.yaml" in readme
    assert "bfloat16 supervised-instruction objective smoke" in readme
    assert "mohawk_tiny_bfloat16_supervised_instruct_cpu_run/save" in readme
    assert "bfloat16 teacher-logits/generation-supervision modes" in readme
    assert "configs/smoke/tiny_cpu_eval_ppl.yaml" in readme
    assert "EvalConfig -> eval_ppl" in readme
    assert "Tiny CPU bfloat16 Mohawk checkpoint PPL" in readme
    assert "PPL JSON includes `model_dtype`" in readme
    assert "`--backend mohawk`, `--model` must be a local Mohawk checkpoint directory with\n`config.json`" in readme
    assert "passing a Hugging Face model ID is not a representative Mohawk\ncheckpoint PPL run" in readme
    assert "ran as `torch.bfloat16`" in readme
    assert "default `--backend auto` also routes this local\nMohawk checkpoint directory" in readme
    assert "Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint PPL" in readme
    assert "mohawk_tiny_doubleblock_discrete_mamba2_ref_cpu_run/save" in readme
    assert "mohawk_tiny_doubleblock_vanilla_discrete_mamba2_ref_cpu_run/save" in readme
    assert "mohawk_tiny_doubleblock_hymba_discrete_mamba2_ref_cpu_run/save" in readme
    assert "mohawk_tiny_doubleblock_merger_discrete_mamba2_ref_cpu_run/save" in readme
    assert "66080.9453125" in readme
    assert "42200.625" in readme
    assert "64897.3828125" in readme
    assert "70790.328125" in readme
    assert "fast Mamba kernels, CUDA eval" in readme
    assert "Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint PPL" in readme
    assert "Slurm compute nodes cannot see login-host `/tmp`" in readme
    assert "artifacts/gpu_smoke/doubleblock_ref_checkpoints/adapter" in readme
    assert "42200.62890625" in readme
    assert "64897.38671875" in readme
    assert "70790.3203125" in readme
    assert "tiny single-process CUDA DoubleBlock-family checkpoint PPL" in readme
    assert "save_latest" in readme
    assert "skips `dataloader_state_dict.pth`" in readme
    assert "configs/smoke/tiny_cpu_bfloat16_eval_ppl.yaml" in readme
    assert "bfloat16 wrapper-driven `eval_ppl` callback" in readme
    assert "mohawk_tiny_bfloat16_eval_ppl_cpu_run/save/latest_Perplexity" in readme
    assert "CUDA bfloat16" in readme
    assert "configs/smoke/tiny_cuda_bfloat16_eval_ppl.yaml" in readme
    assert "configs/smoke/tiny_cuda_ddp_bfloat16_eval_ppl.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_bfloat16_eval_ppl.yaml" in readme
    assert "tiny_cuda_bfloat16_eval_ppl/save/latest_Perplexity" in readme
    assert "eval_score: 49020.808594" in readme
    assert "eval_score: 55746.601562" in readme
    assert "configs/smoke/tiny_cpu_eval_ppl_frequency.yaml" in readme
    assert "frequency: 1" in readme
    assert "eval_at_start: false" in readme
    assert "eval_at_end: false" in readme
    assert "configs/smoke/tiny_cpu_eval_ppl_save_best.yaml" in readme
    assert "save_best" in readme
    assert "best_Perplexity" in readme
    assert "configs/smoke/tiny_cpu_bfloat16_eval_ppl_save_best.yaml" in readme
    assert "bfloat16 `save_best` checkpoint smoke" in readme
    assert "mohawk_tiny_bfloat16_eval_ppl_save_best_cpu_run/save/best_Perplexity" in readme
    assert "configs/smoke/tiny_cpu_eval_multi.yaml" in readme
    assert "multiple `EvalConfig` entries" in readme
    assert "cached `wikitext`" in readme
    assert "configs/smoke/tiny_cpu_eval_ppl_hfdata_save_latest.yaml" in readme
    assert "dataloader_state_dict.pth` into the evaluator latest checkpoint" in readme
    assert "configs/smoke/tiny_cpu_bfloat16_eval_ppl_hfdata_save_latest.yaml" in readme
    assert "bfloat16 public HFDataset `save_latest` state smoke" in readme
    assert "mohawk_tiny_bfloat16_eval_ppl_hfdata_cpu_run/save/latest_Perplexity" in readme
    assert "configs/smoke/tiny_cpu_bfloat16_eval_ppl_hfdata_save_best.yaml" in readme
    assert "bfloat16 public HFDataset `save_best` state smoke" in readme
    assert "mohawk_tiny_bfloat16_eval_ppl_hfdata_save_best_cpu_run/save/best_Perplexity" in readme
    assert "configs/smoke/tiny_cpu_eval_ppl_hfdata_save_best.yaml" in readme
    assert "dataloader_state_dict.pth` into the evaluator best checkpoint" in readme
    assert "configs/smoke/tiny_cpu_hfdata_dataloader_resume.yaml" in readme
    assert "LoadConfig.dataloader" in readme
    assert "saved `_index: 2`" in readme
    assert "new latest checkpoint with `_index: 4`" in readme
    assert "configs/smoke/tiny_cpu_hfdata_full_resume.yaml" in readme
    assert "model, optimizer, scheduler, and `LoadConfig.dataloader`" in readme
    assert "scheduler step 4" in readme
    assert "latest checkpoint with `_index: 5`" in readme
    assert "configs/smoke/tiny_cpu_bfloat16_hfdata_full_resume.yaml" in readme
    assert "bfloat16 public HFDataset full checkpoint/dataloader resume" in readme
    assert "mohawk_tiny_bfloat16_eval_ppl_hfdata_cpu_run/save/latest_Perplexity" in readme
    assert "saved tensors as `torch.bfloat16`" in readme
    assert "configs/smoke/tiny_cpu_mixed_precision_resume.yaml" in readme
    assert "mohawk_tiny_mixed_precision_resume_cpu_run/save" in readme
    assert "TrainConfig.mixed_precision: true" in readme
    assert "TeacherConfig.mixed_precision: true" in readme
    assert "configs/smoke/tiny_cpu_bfloat16_resume.yaml" in readme
    assert "mohawk_tiny_bfloat16_resume_cpu_run/save" in readme
    assert "TrainConfig.model_dtype: bfloat16" in readme
    assert "TeacherConfig.model_dtype: bfloat16" in readme
    assert "configs/smoke/tiny_cpu_mixed_precision_lazy_load.yaml" in readme
    assert "mohawk_tiny_mixed_precision_lazy_cpu_run/save" in readme
    assert "configs/smoke/tiny_cpu_bfloat16_lazy_load.yaml" in readme
    assert "mohawk_tiny_bfloat16_lazy_cpu_run/save" in readme
    assert "saved\ncheckpoint tensors remain `torch.bfloat16`" in readme
    assert "Tiny CUDA lazy-load checkpoint smoke" in readme
    assert "configs/smoke/tiny_cuda_lazy_load.yaml" in readme
    assert "tiny_cuda_supervised/save" in readme
    assert "tiny_cuda_lazy_load/save" in readme
    assert "moved\nthe wrapper to `cuda:0`" in readme
    assert "Tiny CUDA DDP/FSDP lazy-load checkpoint smoke" in readme
    assert "configs/smoke/tiny_cuda_ddp_lazy_load.yaml" in readme
    assert "configs/smoke/tiny_cuda_fsdp_lazy_load.yaml" in readme
    assert "tiny_cuda_ddp_supervised/save" in readme
    assert "tiny_cuda_fsdp_supervised/save" in readme
    assert "DDP rank-0 broadcast and FSDP `FULL_SHARD`" in readme
    assert "hashes changed from the\nsource checkpoints" in readme
    assert "multi-node distributed lazy init" in readme
    assert "mixed-precision distributed\nlazy init" in readme
    assert "configs/smoke/tiny_cpu_eval_benchmark.yaml" in readme
    assert "EvalConfig -> benchmark" in readme
    assert "/tmp/mohawk_lm_eval_datasets" in readme
    assert "configs/smoke/tiny_cpu_bfloat16_eval_benchmark.yaml" in readme
    assert "bfloat16 wrapper-driven benchmark callback" in readme
    assert "bfloat16 AMP\nfallback path" in readme
    assert "Tiny CPU bfloat16 Mohawk checkpoint lm-eval" in readme
    assert "mohawk_tiny_bfloat16_fallback_cpu_run/save" in readme
    assert "infers the saved `TrainConfig.model_dtype`" in readme
    assert "Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoint lm-eval" in readme
    assert "mohawk_tiny_doubleblock_discrete_mamba2_ref_cpu_run/save" in readme
    assert "mohawk_tiny_doubleblock_vanilla_discrete_mamba2_ref_cpu_run/save" in readme
    assert "mohawk_tiny_doubleblock_hymba_discrete_mamba2_ref_cpu_run/save" in readme
    assert "mohawk_tiny_doubleblock_merger_discrete_mamba2_ref_cpu_run/save" in readme
    assert "wikitext 206463.186451" in readme
    assert "wikitext 218913.474906" in readme
    assert "CUDA lm-eval" in readme
    assert "Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint lm-eval" in readme
    assert "Slurm compute nodes cannot see\nlogin-host `/tmp`" in readme
    assert "artifacts/gpu_smoke/lm_eval_datasets" in readme
    assert "artifacts/gpu_smoke/doubleblock_ref_checkpoints/adapter" in readme
    assert "artifacts/gpu_smoke/doubleblock_ref_checkpoints/vanilla" in readme
    assert "artifacts/gpu_smoke/doubleblock_ref_checkpoints/hymba" in readme
    assert "artifacts/gpu_smoke/doubleblock_ref_checkpoints/merger" in readme
    assert "--device cuda" in readme
    assert "All four loaded on `cuda:0`" in readme
    assert "tiny single-process CUDA\nDoubleBlock-family checkpoint lm-eval" in readme
    assert "--model_dtype auto" in readme
    assert "mohawk_tiny_bfloat16_fallback_cpu_run/save" in readme
    assert "TrainConfig.model_dtype" in readme
    assert "Tiny CPU DoubleBlock DiscreteMamba2 reference checkpoints" in readme
    assert "mohawk_tiny_doubleblock_discrete_mamba2_ref_cpu_run/save" in readme
    assert "DiscreteMamba2 autoregressive step requires mamba_ssm selective_state_update" in readme
    assert "pure-PyTorch reference" in readme
    assert "autoregressive recurrence" in readme
    assert "A direct GB300 probe with `mamba-ssm==2.2.6.post3`" in readme
    assert "Tiny fast-kernel `run.py` smokes" in readme
    assert "hello Bian Jian" in readme
    assert "CUDA generation" in readme
    assert "Tiny CUDA DoubleBlock DiscreteMamba2 reference checkpoint generation" in readme
    assert "Slurm compute nodes cannot see login-host\n  `/tmp`" in readme
    assert "artifacts/gpu_smoke/doubleblock_ref_checkpoints/adapter" in readme
    assert "artifacts/gpu_smoke/doubleblock_ref_checkpoints/vanilla" in readme
    assert "artifacts/gpu_smoke/doubleblock_ref_checkpoints/hymba" in readme
    assert "artifacts/gpu_smoke/doubleblock_ref_checkpoints/merger" in readme
    assert "parameter counts `1612596`,\n  `1612324`, `1612660`, and `1612628`" in readme
    assert "printed prompt processing plus decoding times `8ms`, `6ms`, `7ms`, and\n  `7ms`" in readme
    assert "tiny single-process CUDA DoubleBlock-family\n  checkpoint generation" in readme
    assert "production Qwen/Llama hybrid checkpoints" in readme
    assert "configs/smoke/tiny_cpu_hf_teacher_supervised.yaml" in contributing
    assert "Hugging Face teacher-loading path" in contributing
    assert "first run needs\n     network access" in contributing
    assert "configs/smoke/tiny_cpu_public_hfdata_supervised.yaml" in contributing
    assert "public Hugging Face dataset training path" in contributing
    assert "NeelNanda/pile-10k" in contributing
    assert "HF_DATASETS_OFFLINE=1" in contributing
    assert "default-buffer streaming throughput" in contributing
    assert "configs/smoke/tiny_cpu_hybrid_transfer.yaml" in contributing
    assert "hybrid-weight-transfer changes" in contributing
    assert "SSM/Mamba transfer" in contributing
    assert "tiny_cpu_doubleblock_<variant>_supervised.yaml" in contributing
    assert "DoubleBlock block-family changes" in contributing
    assert "Do not treat multi-process CPU" in contributing
    assert "before DDP/FSDP setup" in contributing

    for old_status in OLD_STATUS_WORDING:
        assert old_status not in readme
        assert old_status not in contributing


def test_lm_eval_docs_use_network_first_cache_later_command():
    readme = (ROOT / "README.md").read_text()
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    matrix = (ROOT / "PUBLIC_SUPPORT_MATRIX.md").read_text()

    tiny_section = readme.split("Tiny public Hugging Face smoke", 1)[1].split("## Utility Scripts", 1)[0]
    assert "--tasks wikitext --batch_size 1 --limit 1\n```" in tiny_section
    assert "--limit 1 --local_files_only" not in tiny_section.split("```", 2)[1]
    assert "A first local-only run fails if\nthe task dataset cache is absent" in readme

    assert "--batch_size 1 --limit 1`. After both the model and task dataset are\n     cached" in contributing
    assert "A first local-only run fails when the lm-eval task dataset cache\n     is absent" in contributing
    assert "OfflineModeIsEnabled" in matrix


def test_latest_real_validation_commands_use_allowed_statuses():
    rows = _matrix_rows("Latest Real Validation Commands")
    assert rows
    for command, outcome, status in rows:
        assert command
        assert outcome
        assert status in ALLOWED_STATUSES

    commands = "\n".join(row[0] for row in rows)
    outcomes = "\n".join(row[1] for row in rows)
    assert "Fresh default runtime venv created" in outcomes
    assert "Fresh CPU manifest venv created" in outcomes
    assert "pytest-9.1.1" in outcomes
    assert "driver not visible" in outcomes
    assert "NVIDIA GB300" in outcomes
    assert "cuda available True" in outcomes
    assert "device count 4" in outcomes
    assert "mohawk-phase1-default-venv-20260626" in commands
    assert "no Torch module in system Python" in outcomes
    assert "cuda available False" in outcomes
    assert "Disk quota exceeded" in outcomes
    assert "execve()" in outcomes
    assert "103 passed, 1 skipped, 6 warnings in 234.48s" in outcomes
    assert "1 passed in 5.59s" in outcomes
    assert "tiny_cuda_teacher_ckpt" in outcomes
    assert "Tiny-CUDA-Supervised-Smoke" in outcomes
    assert "[TRAINING_WRAPPER] Device: cuda:0" in outcomes
    assert "287b32e35195cb73a9a0f35493802b98f510bff5269543d6ca721750a8a2bb46" in outcomes
    assert "WORLD_SIZE=1" in outcomes
    assert "Tiny-CUDA-Mixed-Precision-Smoke" in outcomes
    assert "mixed-precision flags True True" in outcomes
    assert "optimizer state entries 12" in outcomes
    assert "bd20786dba84dfb0975db85ea79cebeb1929e1658318fdbfa6e0d18ec91cb42f" in outcomes
    assert "Tiny-CUDA-Resume-Smoke" in outcomes
    assert "source scheduler 2 2" in outcomes
    assert "resumed scheduler 4 4" in outcomes
    assert "17e40ae4debe2dac0f42d643bc662fc4618f921b106a72c80d542a30bcad046f" in outcomes
    assert "Tiny-CUDA-Mixed-Precision-Resume-Smoke" in outcomes
    assert "6e23c54372d768965deab8fc8442d8e842938a52f7c858075d7411d25fd82f4e" in outcomes
    assert "changed_from_source True" in outcomes
    assert "rank 1 received an empty shard" in outcomes
    assert "configs/smoke/data_ddp" in outcomes
    assert "torchrun was not on PATH" in outcomes
    assert "Tiny-CUDA-DDP-Supervised-Smoke" in outcomes
    assert "mixed_precision False False" in outcomes
    assert "Broadcasted weights from rank 0 to all ranks" in outcomes
    assert "wrappers ddp ddp" in outcomes
    assert "Tiny-CUDA-DDP-Mixed-Precision-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Supervised-Smoke" in outcomes
    assert "Sharding Strategy: FULL_SHARD" in outcomes
    assert "Activation Checkpointing: False" in outcomes
    assert "Tiny-CUDA-FSDP-Mixed-Precision-Smoke" in outcomes
    assert "Mixed Precision: True" in outcomes
    assert "Tiny-CUDA-FSDP-Activation-Checkpointing-Smoke" in outcomes
    assert "Activation Checkpointing: True" in outcomes
    assert "01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in outcomes
    assert "tiny_cuda_ddp_gradient_accumulation.yaml" in commands
    assert "tiny_cuda_fsdp_gradient_accumulation.yaml" in commands
    assert "Tiny-CUDA-DDP-Gradient-Accumulation-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Gradient-Accumulation-Smoke" in outcomes
    assert "Using DDP no_sync for gradient accumulation" in outcomes
    assert "Using FSDP no_sync for gradient accumulation" in outcomes
    assert "effective_batch_size: 4" in outcomes
    assert "n_tokens: 64" in outcomes
    assert "Total Grad Steps: 2" in outcomes
    assert "scheduler states had _step_count == 2" in outcomes
    assert "num_steps == 2" in outcomes
    assert "all saved model tensors were torch.float32" in outcomes
    assert "b75cb407af8b36f88a630ae2f4a9ddcc240edbf31548899075c4f8136c729e29" in outcomes
    assert "tiny_cuda_ddp_mixed_precision_gradient_accumulation.yaml" in commands
    assert "tiny_cuda_fsdp_mixed_precision_gradient_accumulation.yaml" in commands
    assert "Tiny-CUDA-DDP-Mixed-Precision-Gradient-Accumulation-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Mixed-Precision-Gradient-Accumulation-Smoke" in outcomes
    assert "mixed-precision flags True True" in outcomes
    assert "TrainConfig.mixed_precision: true" in outcomes
    assert "TeacherConfig.mixed_precision: true" in outcomes
    assert "69a62d39bd3931ba5ccfcdaa3a5d08e285a29a9fa359a15e16cb54646c6d5203" in outcomes
    assert "a8eb5ba1843710879037f98bb659cccf4d477f885337907172c057e1da62a97e" in outcomes
    assert "tiny_cuda_ddp_bfloat16_gradient_accumulation.yaml" in commands
    assert "tiny_cuda_fsdp_bfloat16_gradient_accumulation.yaml" in commands
    assert "Tiny-CUDA-DDP-BFloat16-Gradient-Accumulation-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-BFloat16-Gradient-Accumulation-Smoke" in outcomes
    assert "TrainConfig.model_dtype: bfloat16" in outcomes
    assert "TeacherConfig.model_dtype: bfloat16" in outcomes
    assert "Model dtype: torch.bfloat16" in outcomes
    assert "Mixed Precision: False" in outcomes
    assert "all saved model tensors were torch.bfloat16" in outcomes
    assert "84819bd613ccd234514e3b6aff202d5abdc69674963d7941328e80bf3ddd7282" in outcomes
    assert "tiny_cuda_ddp_multinode_gradient_accumulation.yaml" in commands
    assert "tiny_cuda_fsdp_multinode_gradient_accumulation.yaml" in commands
    assert "configs/smoke/data_ddp_multinode" in outcomes
    assert "nvl72d091-T05" in outcomes
    assert "nvl72d091-T07" in outcomes
    assert "node_rank=0" in outcomes
    assert "node_rank=1" in outcomes
    assert "Tiny-CUDA-DDP-Multinode-Gradient-Accumulation-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Multinode-Gradient-Accumulation-Smoke" in outcomes
    assert "WORLD_SIZE=4" in outcomes
    assert "four-rank DDP over two Slurm nodes" in outcomes
    assert "four-rank FSDP over two Slurm nodes" in outcomes
    assert "effective_batch_size: 8" in outcomes
    assert "Total Grad Steps: 1" in outcomes
    assert "scheduler states had _step_count == 1" in outcomes
    assert "num_steps == 1" in outcomes
    assert "tiny_cuda_ddp_multinode_mixed_precision_gradient_accumulation.yaml" in commands
    assert "tiny_cuda_fsdp_multinode_mixed_precision_gradient_accumulation.yaml" in commands
    assert "Tiny-CUDA-DDP-Multinode-Mixed-Precision-Gradient-Accumulation-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Multinode-Mixed-Precision-Gradient-Accumulation-Smoke" in outcomes
    assert "nvl72d112-T11" in outcomes
    assert "nvl72d112-T13" in outcomes
    assert "nvl72d037-T15" in outcomes
    assert "nvl72d037-T18" in outcomes
    assert "multi-node DDP/FSDP mixed-precision evidence update" in commands
    assert "6 passed in 1.14s" in outcomes
    assert "19 passed, 8 skipped in 3.29s" in outcomes
    assert "mixed precision True True" in outcomes
    assert "TrainConfig.mixed_precision: true" in outcomes
    assert "TeacherConfig.mixed_precision: true" in outcomes
    assert "Mixed Precision: True" in outcomes
    assert "tiny_cuda_ddp_multinode_bfloat16_gradient_accumulation.yaml" in commands
    assert "tiny_cuda_fsdp_multinode_bfloat16_gradient_accumulation.yaml" in commands
    assert "Tiny-CUDA-DDP-Multinode-BFloat16-Gradient-Accumulation-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Multinode-BFloat16-Gradient-Accumulation-Smoke" in outcomes
    assert "nvl72d012-T01" in outcomes
    assert "nvl72d012-T02" in outcomes
    assert "TrainConfig.model_dtype: bfloat16" in outcomes
    assert "TeacherConfig.model_dtype: bfloat16" in outcomes
    assert "model dtype bfloat16 bfloat16" in outcomes
    assert "Model dtype: torch.bfloat16" in outcomes
    assert "all saved model tensors were torch.bfloat16" in outcomes
    assert "0bc9e79b22d4fce8a926de7260a144c0a5a0ff09a86328af5f35777ab655af15" in outcomes
    assert "multi-node DDP/FSDP bfloat16 evidence update" in commands
    assert "Static public docs/matrix assertions passed with 6 passed" in outcomes
    assert "Default lightweight suite passed with 19 passed, 8 skipped" in outcomes
    assert "tiny_cuda_ddp_multinode_resume.yaml" in commands
    assert "tiny_cuda_fsdp_multinode_resume.yaml" in commands
    assert "Tiny-CUDA-DDP-Multinode-Resume-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Multinode-Resume-Smoke" in outcomes
    assert "nvl72d094-T12" in outcomes
    assert "nvl72d094-T14" in outcomes
    assert "nvl72d016-T15" in outcomes
    assert "nvl72d016-T18" in outcomes
    assert "n_tokens 128" in outcomes
    assert "source model load from /home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_ddp_multinode_gradient_accumulation/save" in outcomes
    assert "source model load from /home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_fsdp_multinode_gradient_accumulation/save" in outcomes
    assert "scheduler target Total Grad Steps: 2" in outcomes
    assert "DDP no-sync during resumed accumulation" in outcomes
    assert "FSDP no-sync during resumed accumulation" in outcomes
    assert "scheduler states advanced 1/1 -> 2/2" in outcomes
    assert "optimizer state entries advanced 0 -> 12" in outcomes
    assert "all saved model tensors were torch.float32" in outcomes
    assert "5faf34670d9ba7799cb14ed17fb29ec3679c01a003c1d9100783181ae2d6b2d9" in outcomes
    assert "multi-node DDP/FSDP checkpoint resume evidence update" in commands
    assert "tiny_cuda_ddp_multinode_eval_hstates.yaml" in commands
    assert "tiny_cuda_fsdp_multinode_eval_hstates.yaml" in commands
    assert "Tiny-CUDA-DDP-Multinode-Eval-HStates-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Multinode-Eval-HStates-Smoke" in outcomes
    assert "nvl72d026-T08" in outcomes
    assert "nvl72d026-T10" in outcomes
    assert "EvalConfig[0].Evaluation eval_hstates" in outcomes
    assert "eval_at_start True" in outcomes
    assert "eval_at_end True" in outcomes
    assert "save_latest True" in outcomes
    assert "n_batches 1" in outcomes
    assert "eval_at_start: true" in outcomes
    assert "eval_at_end: true" in outcomes
    assert "save_latest: true" in outcomes
    assert "n_batches: 1" in outcomes
    assert "four-rank DDP eval_hstates callbacks over two Slurm nodes" in outcomes
    assert "four-rank FSDP eval_hstates callbacks over two Slurm nodes" in outcomes
    assert "eval_score: 1.593651" in outcomes
    assert "hstates_distance: 1.593651" in outcomes
    assert "latest_eval_hstates.txt" in outcomes
    assert "1.5936508178710938" in outcomes
    assert "final/latest DDP/FSDP hashes matched" in outcomes
    assert "multi-node DDP/FSDP eval_hstates evidence update" in commands
    assert "tiny_cuda_ddp_multinode_eval_ppl.yaml" in commands
    assert "tiny_cuda_fsdp_multinode_eval_ppl.yaml" in commands
    assert "Tiny-CUDA-DDP-Multinode-Eval-PPL-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Multinode-Eval-PPL-Smoke" in outcomes
    assert "nvl72d012-T03" in outcomes
    assert "nvl72d012-T04" in outcomes
    assert "nvl72d103-T01" in outcomes
    assert "nvl72d103-T02" in outcomes
    assert "EvalConfig[0].Evaluation eval_ppl" in outcomes
    assert "four-rank DDP eval_ppl callbacks over two Slurm nodes" in outcomes
    assert "four-rank FSDP eval_ppl callbacks over two Slurm nodes" in outcomes
    assert "eval_score: 44002.273438" in outcomes
    assert "perplexity: 44002.273438" in outcomes
    assert "accuracy: 0.000000" in outcomes
    assert "latest_Perplexity.txt" in outcomes
    assert "44002.2734375" in outcomes
    assert "multi-node DDP/FSDP eval_ppl evidence update" in commands
    assert "tiny_cuda_ddp_resume.yaml" in commands
    assert "two-rank DDP checkpoint resume path" in outcomes
    assert "source scheduler 1/1, target scheduler 2/2" in outcomes
    assert "source optimizer state entries 0" in outcomes
    assert "target optimizer state entries 12" in outcomes
    assert "target hash 14d447df4d4d3c6bfff71734e78cb36c71909b38ffb41be77c8b698739e5401b" in outcomes
    assert "tiny_cuda_fsdp_resume.yaml" in commands
    assert "two-rank FSDP checkpoint resume path" in outcomes
    assert "sharding FULL_SHARD FULL_SHARD" in outcomes
    assert "causal-conv1d" in outcomes
    assert "transformers 5.12.1" in outcomes
    assert "MambaCache" in outcomes
    assert "transformers 4.55.4" in outcomes
    assert "lm_eval 0.4.11" in outcomes
    assert "mamba_ssm_spec None" in outcomes
    assert "nvl72d086-T12" in outcomes
    assert "nvcc None" in outcomes
    assert "gcc /usr/bin/gcc" in outcomes
    assert "NVIDIA GB300" in outcomes
    assert "ssm-cuda-20260626" in commands
    assert "one-shot optional install failed" in outcomes
    assert "ModuleNotFoundError: No module named 'torch'" in outcomes
    assert "pip install --no-build-isolation 'mamba-ssm>=2.2.5'" in commands
    assert "mamba_ssm-2.3.2.post1" in outcomes
    assert "mamba_ssm was requested, but nvcc was not found" in outcomes
    assert "pip install --no-build-isolation 'causal-conv1d @" in commands
    assert "causal_conv1d was requested, but nvcc was not found" in outcomes
    assert "NameError: name 'bare_metal_version' is not defined" in outcomes
    assert "storage venv used 4.8G" in outcomes
    assert "storage pip cache used 3.0G" in outcomes
    assert "/home/abick/storage/mohawk/venvs/ssm-cuda-20260626/bin/python -c" in commands
    assert "python storage ssm-cuda venv" in outcomes
    assert "cuda available True" in outcomes
    assert "device count 4" in outcomes
    assert "after the optional CUDA/SSM storage-backed install evidence update" in commands
    assert "tiny_cuda_default_runtime_supervised.yaml" in commands
    assert "/home/abick/storage/mohawk/venvs/ssm-cuda-20260626/bin/python run.py --config configs/smoke/tiny_cuda_default_runtime_supervised.yaml" in commands
    assert "Tiny-CUDA-Default-Runtime-Supervised-Smoke" in outcomes
    assert "BaseDataWrapper and BaseDataGenerator" in commands
    assert "BaseDataWrapper.close()" in outcomes
    assert "NotImplementedError" in outcomes
    assert "test_base_data_wrapper_close_tolerates_leaf_generator_without_iterable" in outcomes
    assert "test_leaf_generator_missing_attribute_raises_attribute_error" in outcomes
    assert "13 passed in 20.07s" in outcomes
    assert "Slurm job 690381" in outcomes
    assert "optimizer_state_entries 12" in outcomes
    assert "after the default-runtime CUDA evidence update" in commands
    assert "6 passed in 1.48s" in outcomes
    assert "13 passed in 15.17s" in outcomes
    assert "19 passed, 8 skipped in 3.52s" in outcomes
    assert "18 passed in 73.68s" in outcomes
    assert "tiny_cuda_ddp_default_runtime_supervised.yaml" in commands
    assert "tiny_cuda_fsdp_default_runtime_supervised.yaml" in commands
    assert "Tiny-CUDA-DDP-Default-Runtime-Supervised-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Default-Runtime-Supervised-Smoke" in outcomes
    assert "/home/abick/storage/mohawk/venvs/ssm-cuda-20260626/bin/torchrun" in commands
    assert "Real Slurm job 690730" in outcomes
    assert "Real Slurm job 690735" in outcomes
    assert "MASTER_ADDR=nvl72d053-T13.cm.cluster" in outcomes
    assert "MASTER_ADDR=nvl72d027-T09.cm.cluster" in outcomes
    assert "teacher weight broadcast from rank 0" in outcomes
    assert "Sharding Strategy: FULL_SHARD" in outcomes
    assert "FSDP sharding FULL_SHARD FULL_SHARD" in outcomes
    assert "matching hash 01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in outcomes
    assert "after the default-runtime DDP/FSDP evidence update" in commands
    assert "6 passed in 0.07s" in outcomes
    assert "19 passed, 8 skipped in 3.03s" in outcomes
    assert "tiny_cuda_ddp_default_runtime_mixed_precision.yaml" in commands
    assert "tiny_cuda_fsdp_default_runtime_mixed_precision.yaml" in commands
    assert "Tiny-CUDA-DDP-Default-Runtime-Mixed-Precision-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Default-Runtime-Mixed-Precision-Smoke" in outcomes
    assert "Real Slurm job 690963" in outcomes
    assert "Real Slurm job 690971" in outcomes
    assert "MASTER_ADDR=nvl72d036-T18.cm.cluster" in outcomes
    assert "MASTER_ADDR=nvl72d071-T17.cm.cluster" in outcomes
    assert "mixed-precision flags True True" in outcomes
    assert "Sharding Strategy: FULL_SHARD" in outcomes
    assert "FSDP sharding FULL_SHARD FULL_SHARD" in outcomes
    assert "tiny_cuda_{ddp,fsdp}_default_runtime_mixed_precision/save" in commands
    assert "after the default-runtime DDP/FSDP mixed-precision evidence update" in commands
    assert "6 passed in 1.55s" in outcomes
    assert "19 passed, 8 skipped in 3.37s" in outcomes
    assert "tiny_cuda_ddp_default_runtime_bfloat16.yaml" in commands
    assert "tiny_cuda_fsdp_default_runtime_bfloat16.yaml" in commands
    assert "Tiny-CUDA-DDP-Default-Runtime-BFloat16-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Default-Runtime-BFloat16-Smoke" in outcomes
    assert "Real Slurm job 691303" in outcomes
    assert "Real Slurm job 691314" in outcomes
    assert "MASTER_ADDR=nvl72d054-T09.cm.cluster" in outcomes
    assert "MASTER_ADDR=nvl72d080-T18.cm.cluster" in outcomes
    assert "dtype bfloat16 bfloat16" in outcomes
    assert "bfloat16 AMP fallback" in outcomes
    assert "Model dtype: torch.bfloat16" in outcomes
    assert "Mixed Precision: False" in outcomes
    assert "tiny_cuda_{ddp,fsdp}_default_runtime_bfloat16/save" in commands
    assert "dtypes ['torch.bfloat16']" in outcomes
    assert "size 3222952" in outcomes
    assert "matching hash 0bc9e79b22d4fce8a926de7260a144c0a5a0ff09a86328af5f35777ab655af15" in outcomes
    assert "after the default-runtime DDP/FSDP bfloat16 evidence update" in commands
    assert "6 passed" in outcomes
    assert "19 passed, 8 skipped" in outcomes
    assert "tiny_cuda_ddp_default_runtime_resume.yaml" in commands
    assert "tiny_cuda_fsdp_default_runtime_resume.yaml" in commands
    assert "Tiny-CUDA-DDP-Default-Runtime-Resume-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Default-Runtime-Resume-Smoke" in outcomes
    assert "exactly one model load path" in outcomes
    assert "tiny_cuda_ddp_default_runtime_supervised/save" in outcomes
    assert "tiny_cuda_fsdp_default_runtime_supervised/save" in outcomes
    assert "Real Slurm job 691369" in outcomes
    assert "Real Slurm job 691375" in outcomes
    assert "MASTER_ADDR=nvl72d112-T18.cm.cluster" in outcomes
    assert "MASTER_ADDR=nvl72d112-T15.cm.cluster" in outcomes
    assert "teacher weight broadcast from rank 0" in outcomes
    assert "dataloader cycling during resumed training" in outcomes
    assert "tiny_cuda_ddp_default_runtime_resume/save" in outcomes
    assert "tiny_cuda_fsdp_default_runtime_resume/save" in outcomes
    assert "scheduler state advanced 1/1 -> 2/2" in outcomes
    assert "optimizer state entries advanced 0 -> 12" in outcomes
    assert "target hash 14d447df4d4d3c6bfff71734e78cb36c71909b38ffb41be77c8b698739e5401b" in outcomes
    assert "changed_tensors 12" in outcomes
    assert "after the default-runtime DDP/FSDP checkpoint-resume evidence update" in commands
    assert "tiny_cuda_{ddp,fsdp}_default_runtime_eval_{hstates,ppl,benchmark}.yaml" in commands
    assert "tiny_cuda_ddp_default_runtime_eval_hstates.yaml" in commands
    assert "tiny_cuda_fsdp_default_runtime_eval_hstates.yaml" in commands
    assert "tiny_cuda_ddp_default_runtime_eval_ppl.yaml" in commands
    assert "tiny_cuda_fsdp_default_runtime_eval_ppl.yaml" in commands
    assert "tiny_cuda_ddp_default_runtime_eval_benchmark.yaml" in commands
    assert "tiny_cuda_fsdp_default_runtime_eval_benchmark.yaml" in commands
    assert "Tiny-CUDA-DDP-Default-Runtime-Eval-HStates-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Default-Runtime-Eval-HStates-Smoke" in outcomes
    assert "Tiny-CUDA-DDP-Default-Runtime-Eval-PPL-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Default-Runtime-Eval-PPL-Smoke" in outcomes
    assert "Tiny-CUDA-DDP-Default-Runtime-Eval-Benchmark-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Default-Runtime-Eval-Benchmark-Smoke" in outcomes
    assert "Real Slurm job 691481" in outcomes
    assert "Real Slurm job 691627" in outcomes
    assert "Real Slurm job 691806" in outcomes
    assert "Real Slurm job 691852" in outcomes
    assert "Real Slurm job 691885" in outcomes
    assert "Real Slurm job 691925" in outcomes
    assert "MASTER_ADDR=nvl72d122-T06.cm.cluster" in outcomes
    assert "MASTER_ADDR=nvl72d112-T17.cm.cluster" in outcomes
    assert "MASTER_ADDR=nvl72d062-T10.cm.cluster" in outcomes
    assert "MASTER_ADDR=nvl72d029-T13.cm.cluster" in outcomes
    assert "MASTER_ADDR=nvl72d089-T16.cm.cluster" in outcomes
    assert "MASTER_ADDR=nvl72d053-T14.cm.cluster" in outcomes
    assert "latest_eval_hstates" in outcomes
    assert "latest_Perplexity" in outcomes
    assert "eval_score: 1.662772" in outcomes
    assert "hstates_distance: 1.662772" in outcomes
    assert "eval_score: 53548.183594" in outcomes
    assert "perplexity: 53548.183594" in outcomes
    assert "wikitext 183619.266616" in outcomes
    assert "AVG 183619.266616" in outcomes
    assert "tiny_cuda_ddp_default_runtime_eval_hstates/save" in outcomes
    assert "tiny_cuda_fsdp_default_runtime_eval_hstates/save" in outcomes
    assert "tiny_cuda_ddp_default_runtime_eval_ppl/save" in outcomes
    assert "tiny_cuda_fsdp_default_runtime_eval_ppl/save" in outcomes
    assert "tiny_cuda_ddp_default_runtime_eval_benchmark/save" in outcomes
    assert "tiny_cuda_fsdp_default_runtime_eval_benchmark/save" in outcomes
    assert "six default-runtime eval saves" in commands
    assert "latest score files record 1.6627724170684814" in outcomes
    assert "53548.18359375" in outcomes
    assert "final/latest eval hashes match 01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in outcomes
    assert "benchmark saves record save_latest False" in outcomes
    assert "matching hash b75cb407af8b36f88a630ae2f4a9ddcc240edbf31548899075c4f8136c729e29" in outcomes
    assert "after the default-runtime DDP/FSDP eval-callback evidence update" in commands
    assert "6 passed in 1.63s" in outcomes
    assert "19 passed, 8 skipped in 3.78s" in outcomes
    assert "tiny_cuda_ddp_default_runtime_multinode_gradient_accumulation.yaml" in commands
    assert "tiny_cuda_fsdp_default_runtime_multinode_gradient_accumulation.yaml" in commands
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Gradient-Accumulation-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Gradient-Accumulation-Smoke" in outcomes
    assert "Real Slurm job 692123" in outcomes
    assert "Real Slurm job 692150" in outcomes
    assert "nvl72d066-T08" in outcomes
    assert "nvl72d066-T09" in outcomes
    assert "nvl72d063-T11" in outcomes
    assert "nvl72d063-T12" in outcomes
    assert "WORLD_SIZE=4" in outcomes
    assert "configs/smoke/data_ddp_multinode" in outcomes
    assert "effective_batch_size 8" in outcomes
    assert "n_tokens 64" in outcomes
    assert "scheduler states are 1/1" in outcomes
    assert "DDP/FSDP hashes match at 01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in outcomes
    assert "after the default-runtime DDP/FSDP multi-node training evidence update" in commands
    assert "6 passed in 0.09s" in outcomes
    assert "19 passed, 8 skipped in 3.88s" in outcomes
    assert "tiny_cuda_{ddp,fsdp}_default_runtime_multinode_eval_{hstates,ppl,benchmark}.yaml" in commands
    assert "tiny_cuda_ddp_default_runtime_multinode_eval_hstates.yaml" in commands
    assert "tiny_cuda_fsdp_default_runtime_multinode_eval_hstates.yaml" in commands
    assert "tiny_cuda_ddp_default_runtime_multinode_eval_ppl.yaml" in commands
    assert "tiny_cuda_fsdp_default_runtime_multinode_eval_ppl.yaml" in commands
    assert "tiny_cuda_ddp_default_runtime_multinode_eval_benchmark.yaml" in commands
    assert "tiny_cuda_fsdp_default_runtime_multinode_eval_benchmark.yaml" in commands
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Eval-HStates-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Eval-HStates-Smoke" in outcomes
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Eval-PPL-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Eval-PPL-Smoke" in outcomes
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Eval-Benchmark-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Eval-Benchmark-Smoke" in outcomes
    assert "Real Slurm job 692419" in outcomes
    assert "Real Slurm job 692430" in outcomes
    assert "Real Slurm job 692462" in outcomes
    assert "Real Slurm job 692471" in outcomes
    assert "Real Slurm job 692475" in outcomes
    assert "Real Slurm job 692491" in outcomes
    assert "nvl72d024-T10" in outcomes
    assert "nvl72d024-T11" in outcomes
    assert "nvl72d024-T12" in outcomes
    assert "nvl72d024-T13" in outcomes
    assert "nvl72d066-T11" in outcomes
    assert "nvl72d066-T17" in outcomes
    assert "nvl72d111-T12" in outcomes
    assert "nvl72d111-T16" in outcomes
    assert "nvl72d112-T10" in outcomes
    assert "nvl72d112-T14" in outcomes
    assert "eval_score: 1.593651" in outcomes
    assert "hstates_distance: 1.593651" in outcomes
    assert "eval_score: 44002.273438" in outcomes
    assert "perplexity: 44002.273438" in outcomes
    assert "wikitext 172057.421054" in outcomes
    assert "AVG 172057.421054" in outcomes
    assert "post-save rendezvous shutdown warning" in outcomes
    assert "recvVector failed" in outcomes
    assert "six default-runtime multi-node eval saves" in commands
    assert "latest score files record 1.5936508178710938" in outcomes
    assert "44002.2734375" in outcomes
    assert "matching hash 01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in outcomes
    assert "after the default-runtime DDP/FSDP multi-node eval-callback evidence update" in commands
    assert "6 passed in 1.87s" in outcomes
    assert "19 passed, 8 skipped in 4.72s" in outcomes
    assert "tiny_cuda_ddp_default_runtime_multinode_resume.yaml" in commands
    assert "tiny_cuda_fsdp_default_runtime_multinode_resume.yaml" in commands
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Resume-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Resume-Smoke" in outcomes
    assert "Real Slurm job 692584" in outcomes
    assert "Real Slurm job 692601" in outcomes
    assert "nvl72d053-T10" in outcomes
    assert "nvl72d053-T13" in outcomes
    assert "nvl72d053-T17" in outcomes
    assert "nvl72d053-T18" in outcomes
    assert "source model load from /home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_ddp_default_runtime_multinode_gradient_accumulation/save" in outcomes
    assert "source model load from /home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_fsdp_default_runtime_multinode_gradient_accumulation/save" in outcomes
    assert "scheduler state advanced 1/1 -> 2/2" in outcomes
    assert "optimizer state entries advanced 0 -> 12" in outcomes
    assert "all 12 tensors changed" in outcomes
    assert "DDP/FSDP target hashes match at 5faf34670d9ba7799cb14ed17fb29ec3679c01a003c1d9100783181ae2d6b2d9" in outcomes
    assert "after the default-runtime DDP/FSDP multi-node checkpoint-resume evidence update" in commands
    assert "6 passed in 1.67s" in outcomes
    assert "19 passed, 8 skipped in 3.39s" in outcomes
    assert "tiny_cuda_{ddp,fsdp}_default_runtime_multinode_{mixed_precision,bfloat16}_gradient_accumulation.yaml" in commands
    assert "tiny_cuda_ddp_default_runtime_multinode_mixed_precision_gradient_accumulation.yaml" in commands
    assert "tiny_cuda_fsdp_default_runtime_multinode_mixed_precision_gradient_accumulation.yaml" in commands
    assert "tiny_cuda_ddp_default_runtime_multinode_bfloat16_gradient_accumulation.yaml" in commands
    assert "tiny_cuda_fsdp_default_runtime_multinode_bfloat16_gradient_accumulation.yaml" in commands
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-Mixed-Precision-Gradient-Accumulation-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-Mixed-Precision-Gradient-Accumulation-Smoke" in outcomes
    assert "Tiny-CUDA-DDP-Default-Runtime-Multinode-BFloat16-Gradient-Accumulation-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Default-Runtime-Multinode-BFloat16-Gradient-Accumulation-Smoke" in outcomes
    assert "Real Slurm job 692716" in outcomes
    assert "Real Slurm job 692810" in outcomes
    assert "Real Slurm job 692872" in outcomes
    assert "Real Slurm job 692887" in outcomes
    assert "nvl72d125-T07" in outcomes
    assert "nvl72d125-T08" in outcomes
    assert "nvl72d105-T17" in outcomes
    assert "nvl72d105-T18" in outcomes
    assert "nvl72d054-T01" in outcomes
    assert "nvl72d054-T02" in outcomes
    assert "nvl72d108-T15" in outcomes
    assert "nvl72d108-T18" in outcomes
    assert "mixed-precision saves have 12 torch.float32 tensors" in outcomes
    assert "bfloat16 saves have 12 torch.bfloat16 tensors" in outcomes
    assert "after the default-runtime DDP/FSDP multi-node mixed-precision/bfloat16 evidence update" in commands
    assert "19 passed, 8 skipped in 4.35s" in outcomes
    assert "du -shL ." in commands
    assert "Repo-local footprint without symlink dereference is 17M" in outcomes
    assert "Repo-local footprint without symlink dereference remains 17M" in outcomes
    assert "symlink-dereferenced footprint is 1.4G" in outcomes
    assert "symlink-dereferenced footprint remains 1.4G" in outcomes
    assert "symlink-dereferenced footprint is now 1.5G" in outcomes
    assert "symlink-dereferenced footprint is now 1.6G" in outcomes
    assert "/home/abick/mohawk/artifacts/gpu_smoke is a symlink" in outcomes
    assert "storage-backed GPU smoke artifacts use 1.4G" in outcomes
    assert "storage-backed GPU smoke artifacts use 1.5G" in outcomes
    assert "storage-backed venvs use 6.5G" in outcomes
    assert "no large generated artifact directory is stored directly in the repo" in outcomes
    assert "ZeroDivisionError" in outcomes
    assert "AssertionError" in outcomes
    assert "Unknown z_init: None" in outcomes
    assert "initializer={'z':'default','out':'default','convolution':'identity'}" in outcomes
    assert "DiscreteMamba2 fast path requires mamba_ssm" in outcomes
    assert "103 passed, 1 skipped" in outcomes
    assert "wikitext 173557.990187" in outcomes
    assert "hello EverestMist" in outcomes
    assert "Number of parameters: 1610832" in outcomes
    assert "Model dtype: torch.float32" in outcomes
    assert "mohawk_tiny_doubleblock_discrete_mamba2_ref_cpu_run/save" in commands
    assert "DiscreteMamba2 autoregressive step requires mamba_ssm selective_state_update" in outcomes
    assert "hello anguish played" in outcomes
    assert "mohawk_tiny_doubleblock_vanilla_discrete_mamba2_ref_cpu_run/save" in commands
    assert "mohawk_tiny_doubleblock_hymba_discrete_mamba2_ref_cpu_run/save" in commands
    assert "mohawk_tiny_doubleblock_merger_discrete_mamba2_ref_cpu_run/save" in commands
    assert "hello Bian Jian" in outcomes
    assert "tiny_cuda_doubleblock_discrete_mamba2_ref_supervised.yaml" in commands
    assert "Tiny-CUDA-DoubleBlockAdapter-DiscreteMamba2-Ref-Smoke" in outcomes
    assert "Slurm job 689221" in outcomes
    assert "LayeredMambaLM -> LlamaModel -> DoubleBlockAdapter" in outcomes
    assert "Trainable parameters: 1,612,596 / 1,612,596" in outcomes
    assert "Total Grad Steps: 2" in outcomes
    assert "tiny_cuda_doubleblock_discrete_mamba2_ref/save" in outcomes
    assert "optimizer state entries 19" in outcomes
    assert "26410b59df6bd130e8b87ca6ce4f35a4040fb7dad89edc6e4392afc47c85eb0c" in outcomes
    assert "max diff 0.00019963830709457397" in outcomes
    assert "tiny_cuda_doubleblock_{vanilla,hymba,merger}_discrete_mamba2_ref_supervised.yaml" in commands
    assert "tiny_cuda_doubleblock_*_discrete_mamba2_ref_supervised.yaml" in commands
    assert "Tiny-CUDA-DoubleBlockVanilla-DiscreteMamba2-Ref-Smoke" in outcomes
    assert "Tiny-CUDA-DoubleBlockHymba-DiscreteMamba2-Ref-Smoke" in outcomes
    assert "Tiny-CUDA-DoubleBlockMerger-DiscreteMamba2-Ref-Smoke" in outcomes
    assert "689237" in outcomes
    assert "689254" in outcomes
    assert "689262" in outcomes
    assert "1,612,324 / 1,612,324" in outcomes
    assert "1,612,660 / 1,612,660" in outcomes
    assert "1,612,628 / 1,612,628" in outcomes
    assert "17/17" in outcomes
    assert "23/23" in outcomes
    assert "21/21" in outcomes
    assert "tiny_cuda_doubleblock_vanilla_discrete_mamba2_ref/save" in outcomes
    assert "tiny_cuda_doubleblock_hymba_discrete_mamba2_ref/save" in outcomes
    assert "tiny_cuda_doubleblock_merger_discrete_mamba2_ref/save" in outcomes
    assert "688309cc0e531b8335568d7bcc5739087fcbf9ad82729bea0efa5711b1465053" in outcomes
    assert "668075227edf9868305ccbbcfcd287f81a51c3325acc7bbc3eb25efe57aca8fd" in outcomes
    assert "282710e0946b6830ef1d3026e4a456d572e797c2742911053293df786dd85459" in outcomes
    assert "Hymba 10/23 max diff 0.00019951441208831966" in outcomes
    assert "after the CUDA DoubleBlock-family DiscreteMamba2 reference training evidence update" in commands
    assert "mv /home/abick/mohawk/artifacts/gpu_smoke /home/abick/storage/mohawk/artifacts/gpu_smoke" in commands
    assert "/home/abick/storage/mohawk/artifacts/gpu_smoke" in outcomes
    assert "repo-side artifacts directory then reported 4.0K" in outcomes
    assert "all eight configs/smoke/tiny_cuda_{ddp,fsdp}_doubleblock*_discrete_mamba2_ref_supervised.yaml" in commands
    assert "Tiny-CUDA-DDP/FSDP-DoubleBlock{Adapter,Vanilla,Hymba,Merger}-DiscreteMamba2-Ref-Smoke" in outcomes
    assert "tiny_cuda_{ddp,fsdp}_doubleblock_{vanilla,hymba,merger}_discrete_mamba2_ref_supervised.yaml" in commands
    assert "tiny_cuda_ddp_doubleblock_discrete_mamba2_ref_supervised.yaml" in commands
    assert "tiny_cuda_fsdp_doubleblock_discrete_mamba2_ref_supervised.yaml" in commands
    assert "Tiny-CUDA-DDP-DoubleBlockAdapter-DiscreteMamba2-Ref-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-DoubleBlockAdapter-DiscreteMamba2-Ref-Smoke" in outcomes
    assert "689318" in outcomes
    assert "689324" in outcomes
    assert "689363" in outcomes
    assert "689371" in outcomes
    assert "689378" in outcomes
    assert "689396" in outcomes
    assert "689404" in outcomes
    assert "689411" in outcomes
    assert "WORLD_SIZE=2" in outcomes
    assert "wrapper ddp" in outcomes
    assert "wrapper fsdp" in outcomes
    assert "NO_SHARD" in outcomes
    assert "FULL_SHARD" in outcomes
    assert "Broadcasted weights from rank 0" in outcomes or "broadcast weights from rank 0" in outcomes
    assert "tiny_cuda_ddp_doubleblock_discrete_mamba2_ref/save" in outcomes
    assert "tiny_cuda_fsdp_doubleblock_discrete_mamba2_ref/save" in outcomes
    assert "86be1ef437abce55cc58f7bffd6f3fdeecab0639a4014967314935dc30041bf4" in outcomes
    assert "ebd61d3b2f1d789631a25612b27c51ca0ed8a5ef0ab516e32aa54a85fde2df8a" in outcomes
    assert "9ec928a4d5863363f05235ba63eb7a065f3152f4e5a0867f34feb85217614d98" in outcomes
    assert "12a16f0b147becab00aa2e6e3a69895368f215bb3c871b0e903c21e4faf2f6e7" in outcomes
    assert "all 19 common tensors changed from the adapter teacher checkpoint" in outcomes
    assert "Hymba 10/23 max diff 0.00019961617363151163" in outcomes
    assert "after the CUDA DDP/FSDP DoubleBlock-family DiscreteMamba2 reference training evidence update" in commands
    assert "1612596" in outcomes
    assert "1612324" in outcomes
    assert "1612660" in outcomes
    assert "1612628" in outcomes
    assert "generation/generate.py --model /home/abick/mohawk/artifacts/gpu_smoke/doubleblock_ref_checkpoints/adapter" in commands
    assert "doubleblock_ref_checkpoints/vanilla" in commands
    assert "doubleblock_ref_checkpoints/hymba" in commands
    assert "doubleblock_ref_checkpoints/merger" in commands
    assert "prompt processing + decoding time: 8ms" in outcomes
    assert "prompt processing plus decoding times" in outcomes
    assert "6ms" in outcomes
    assert "7ms" in outcomes
    assert "generation/generate.py --model /home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_supervised/save" in commands
    assert "Slurm GPU allocation" in outcomes
    assert "prompt processing + decoding time: 4ms" in outcomes
    assert "Model dtype: torch.bfloat16" in outcomes
    assert "mohawk_tiny_bfloat16_fallback_cpu_run/save" in outcomes
    assert "Tiny-CPU-BFloat16-Fallback-Smoke" in outcomes
    assert "TrainConfig.model_dtype bfloat16" in outcomes
    assert "model.safetensors size 3222952" in outcomes
    assert "tiny_cuda_bfloat16_supervised.yaml" in commands
    assert "tiny_cuda_ddp_bfloat16_supervised.yaml" in commands
    assert "tiny_cuda_fsdp_bfloat16_supervised.yaml" in commands
    assert "Tiny-CUDA-BFloat16-Supervised-Smoke" in outcomes
    assert "Tiny-CUDA-DDP-BFloat16-Supervised-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-BFloat16-Supervised-Smoke" in outcomes
    assert "Model dtype: torch.bfloat16" in outcomes
    assert "all saved tensors were torch.bfloat16" in outcomes
    assert "52fe58e628de69a168b351d41c8d01a52017300497a0675770ae75653ccf1ef1" in outcomes
    assert "0bc9e79b22d4fce8a926de7260a144c0a5a0ff09a86328af5f35777ab655af15" in outcomes
    assert "wikitext 206463.186451" in outcomes
    assert "checkpoint dtype inference" in outcomes
    assert "TrainConfig.model_dtype: float32" in outcomes
    assert "evals/eval_ppl.py --backend mohawk --model /home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_supervised/save" in commands
    assert "46136.37890625" in outcomes
    assert "model_dtype: bfloat16" in outcomes
    assert "tiny_cpu_eval_benchmark.yaml" in outcomes
    assert "mohawk_tiny_eval_benchmark_teacher_ckpt" in outcomes
    assert "EvalConfig -> benchmark" in outcomes
    assert "read-only home datasets cache" in outcomes
    assert "datasets.config.HF_DATASETS_CACHE" in outcomes
    assert "wikitext 206537.186039" in outcomes
    assert "mohawk_tiny_eval_benchmark_cpu_run/save" in outcomes
    assert "8a5c277726a85ba8b5646cebb6294fc867d8824e98b208dfca303c2747349370" in outcomes
    assert "tiny_cpu_bfloat16_eval_benchmark.yaml" in outcomes
    assert "timed out before final metrics or checkpoint save" in outcomes
    assert "wikitext 209029.094266" in outcomes
    assert "Tiny-CPU-BFloat16-Eval-Benchmark-Smoke" in outcomes
    assert "EvalConfig[0].Evaluation benchmark" in outcomes
    assert "tiny_cuda_ddp_eval_benchmark.yaml" in commands
    assert "tiny_cuda_fsdp_eval_benchmark.yaml" in outcomes
    assert "tiny_cuda_ddp_multinode_eval_benchmark.yaml" in outcomes
    assert "tiny_cuda_fsdp_multinode_eval_benchmark.yaml" in outcomes
    assert "Tiny-CUDA-DDP-Eval-Benchmark-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Eval-Benchmark-Smoke" in outcomes
    assert "Tiny-CUDA-DDP-Multinode-Eval-Benchmark-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Multinode-Eval-Benchmark-Smoke" in outcomes
    assert "limit: 1 was smaller than WORLD_SIZE=2" in outcomes
    assert "task.build_requests() did not find any docs!" in outcomes
    assert "limit: 2 for two-rank and limit: 4 for four-rank" in outcomes
    assert "wikitext_detokenizer" in outcomes
    assert "aggregated pandas results object" in outcomes
    assert "python3 -m compileall -q evals/benchmark.py training_wrapper/DDPTrainingWrapper.py" in commands
    assert "Slurm job 688002" in outcomes
    assert "TypeError: argument of type 'NoneType' is not iterable" in outcomes
    assert "aggregates only on rank 0" in outcomes
    assert "samples_to_file" in outcomes
    assert "python3 -m compileall -q evals/benchmark.py" in commands
    assert "Immediate post-rank-0-aggregation-fix" in commands
    assert "nvl72d074-T15" in outcomes
    assert "Slurm job 688237" in outcomes
    assert "eval_score: 183619.266616" in outcomes
    assert "wikitext 183619.266616" in outcomes
    assert "AVG 183619.266616" in outcomes
    assert "tiny_cuda_ddp_eval_benchmark/save" in outcomes
    assert "EvalConfig[0].Evaluation benchmark" in outcomes
    assert "scheduler state had _step_count == 2" in outcomes
    assert "Post-DDP-pass" in commands
    assert "Unable to allocate resources: Job violates accounting/QOS policy" in outcomes
    assert "multiple active and pending user jobs" in outcomes
    assert "Slurm job 688259" in outcomes
    assert "Tiny-CUDA-FSDP-Eval-Benchmark-Smoke" in outcomes
    assert "wrappers fsdp fsdp" in outcomes
    assert "FULL_SHARD FULL_SHARD" in outcomes
    assert "real FSDP no_sync during accumulation" in outcomes
    assert "tiny_cuda_fsdp_eval_benchmark/save" in outcomes
    assert "DDP/FSDP hashes matched at b75cb407af8b36f88a630ae2f4a9ddcc240edbf31548899075c4f8136c729e29" in outcomes
    assert "CUDA DDP/FSDP benchmark blocked-evidence update" in commands
    assert "CUDA DDP benchmark rank-0 aggregation evidence update" in commands
    assert "CUDA DDP benchmark pass evidence update" in commands
    assert "CUDA DDP/FSDP benchmark pass evidence update" in commands
    assert "CUDA DDP/FSDP multi-node benchmark allocation attempt" in commands
    assert "CUDA DDP/FSDP multi-node benchmark allocation evidence update" in commands
    assert "CUDA DDP/FSDP multi-node benchmark pass/fix evidence update" in commands
    assert "Tiny-CUDA-DDP-Multinode-Eval-Benchmark-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Multinode-Eval-Benchmark-Smoke" in outcomes
    assert "limit 4" in outcomes
    assert "Immediate execution impossible, insufficient priority" in outcomes
    assert "two-GPU-per-node DDP benchmark attempt" in outcomes
    assert "Slurm job 688389" in outcomes
    assert "Inplace update to inference tensor outside InferenceMode" in outcomes
    assert "torch.no_grad()" in outcomes
    assert "Slurm job 688406" in outcomes
    assert "Slurm job 688414" in outcomes
    assert "eval_score: 172057.421054" in outcomes
    assert "wikitext 172057.421054" in outcomes
    assert "tiny_cuda_ddp_multinode_eval_benchmark/save" in outcomes
    assert "tiny_cuda_fsdp_multinode_eval_benchmark/save" in outcomes
    assert "scheduler states had _step_count == 1" in outcomes
    assert "DDP/FSDP hashes matched at 01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in outcomes
    assert "evals/benchmark.py --backend mohawk --dir /tmp/mohawk_tiny_doubleblock_discrete_mamba2_ref_cpu_run/save" in commands
    assert "mohawk_tiny_doubleblock_vanilla_discrete_mamba2_ref_cpu_run/save" in commands
    assert "mohawk_tiny_doubleblock_hymba_discrete_mamba2_ref_cpu_run/save" in commands
    assert "mohawk_tiny_doubleblock_merger_discrete_mamba2_ref_cpu_run/save" in commands
    assert "DoubleBlockAdapter" in outcomes
    assert "DoubleBlockVanilla" in outcomes
    assert "DoubleBlockHymba" in outcomes
    assert "DoubleBlockMerger" in outcomes
    assert "19/19 keys" in outcomes
    assert "17/17, 23/23, and 21/21 keys" in outcomes
    assert "wikitext 218913.474906" in outcomes
    assert "evals/benchmark.py --backend mohawk --dir /home/abick/mohawk/artifacts/gpu_smoke/doubleblock_ref_checkpoints/adapter" in commands
    assert "artifacts/gpu_smoke/doubleblock_ref_checkpoints/vanilla" in commands
    assert "artifacts/gpu_smoke/doubleblock_ref_checkpoints/hymba" in commands
    assert "artifacts/gpu_smoke/doubleblock_ref_checkpoints/merger" in commands
    assert "HF_DATASETS_CACHE=/home/abick/mohawk/artifacts/gpu_smoke/lm_eval_datasets" in commands
    assert "HF_DATASETS_OFFLINE=1" in commands
    assert "--device cuda" in commands
    assert "Slurm job 689133" in outcomes
    assert "Slurm jobs 689142, 689143, and 689152" in outcomes
    assert "cuda:0" in outcomes
    assert "tiny_cpu_bfloat16_eval_ppl.yaml" in outcomes
    assert "Tiny-CPU-BFloat16-Eval-PPL-Smoke" in outcomes
    assert "EvalConfig[0].Evaluation eval_ppl" in outcomes
    assert "mohawk_tiny_bfloat16_eval_ppl_cpu_run/save" in outcomes
    assert "latest_Perplexity" in outcomes
    assert "49020.8125" in outcomes
    assert "Dataloader state is not implemented" in outcomes
    assert "e5185fe0ec38fae0b521167cd3fa54d81d2b15b9025f33b3b8af71a89a38a2b3" in outcomes
    assert "tiny_cpu_bfloat16_eval_ppl_save_best.yaml" in outcomes
    assert "Tiny-CPU-BFloat16-Eval-PPL-SaveBest-Smoke" in outcomes
    assert "save_best True" in outcomes
    assert "save_latest False" in outcomes
    assert "Saving best model with Perplexity score of 49020.8125" in outcomes
    assert "mohawk_tiny_bfloat16_eval_ppl_save_best_cpu_run/save" in outcomes
    assert "best_Perplexity.txt" in outcomes
    assert "46163.3125" in outcomes
    assert '"model_dtype": "torch.float32"' in outcomes
    assert "49020.8125" in outcomes
    assert '"model_dtype": "torch.bfloat16"' in outcomes
    assert "default --backend auto" in outcomes
    assert "evals/eval_ppl.py --backend mohawk --model /tmp/mohawk_tiny_doubleblock_discrete_mamba2_ref_cpu_run/save" in commands
    assert "mohawk_tiny_doubleblock_vanilla_discrete_mamba2_ref_cpu_run/save" in commands
    assert "mohawk_tiny_doubleblock_hymba_discrete_mamba2_ref_cpu_run/save" in commands
    assert "mohawk_tiny_doubleblock_merger_discrete_mamba2_ref_cpu_run/save" in commands
    assert "DoubleBlockAdapter" in outcomes
    assert "DoubleBlockVanilla" in outcomes
    assert "DoubleBlockHymba" in outcomes
    assert "DoubleBlockMerger" in outcomes
    assert "66080.9453125" in outcomes
    assert "42200.625" in outcomes
    assert "64897.3828125" in outcomes
    assert "70790.328125" in outcomes
    assert "QOSMinGRES" in outcomes
    assert "FileNotFoundError: /tmp/mohawk_tiny_doubleblock_discrete_mamba2_ref_cpu_run/save/config.json" in outcomes
    assert "artifacts/gpu_smoke/doubleblock_ref_checkpoints/{adapter,vanilla,hymba,merger}" in commands
    assert "total size 75M" in outcomes
    assert "evals/eval_ppl.py --backend mohawk --model /home/abick/mohawk/artifacts/gpu_smoke/doubleblock_ref_checkpoints/adapter" in commands
    assert "doubleblock_ref_checkpoints/vanilla" in commands
    assert "doubleblock_ref_checkpoints/hymba" in commands
    assert "doubleblock_ref_checkpoints/merger" in commands
    assert "--device cuda" in commands
    assert "42200.62890625" in outcomes
    assert "64897.38671875" in outcomes
    assert "70790.3203125" in outcomes
    assert "10 passed" in outcomes
    assert "12 passed" in outcomes
    assert "20 passed" in outcomes
    assert "18 passed" in outcomes
    assert "Hugging Face-style output flag aliases" in outcomes
    assert "one-token padding regression" in outcomes
    assert "preserving configured scheduler num_steps on resume" in outcomes
    assert "scheduler _step_count == 2" in outcomes
    assert "Teacher model should have at least 2 hidden states" in outcomes
    assert "mohawk_tiny_hstates_cpu_run/save" in outcomes
    assert "tiny_cpu_bfloat16_hstates.yaml" in outcomes
    assert "Tiny-CPU-BFloat16-HStates-Smoke" in outcomes
    assert "DistillConfig.type hstates" in outcomes
    assert "mohawk_tiny_bfloat16_hstates_cpu_run/save" in outcomes
    assert "42ba2930c56370fcdbbc3ac5a9aa03c02bc11b2c47f671a1272cde227df2a70c" in outcomes
    assert "tiny_cpu_bfloat16_sequential_hstates.yaml" in outcomes
    assert "Tiny-CPU-BFloat16-Sequential-HStates-Smoke" in outcomes
    assert "DistillConfig.type sequential_hstates" in outcomes
    assert "mohawk_tiny_bfloat16_sequential_hstates_cpu_run/save" in outcomes
    assert "bb3c68f0d13a1da992fd61f86d7d5348fe0990ac9b9c3d8537a248c152972957" in outcomes
    assert "tiny_cpu_bfloat16_matrices.yaml" in outcomes
    assert "Tiny-CPU-BFloat16-Matrices-Smoke" in outcomes
    assert "DistillConfig.type matrices" in outcomes
    assert "mohawk_tiny_bfloat16_matrices_cpu_run/save" in outcomes
    assert "1e206bf68e4ea570c0b324ad314bc3fbf9d51e331fa179342808c17cbede0270" in outcomes
    assert "tiny_cpu_bfloat16_dpo.yaml" in outcomes
    assert "Tiny-CPU-BFloat16-DPO-Smoke" in outcomes
    assert "DistillConfig.type dpo" in outcomes
    assert "collate_type=preference" in outcomes
    assert "mohawk_tiny_bfloat16_dpo_cpu_run/save" in outcomes
    assert "6cc7796660ade9e861d3115d20a351ed8e03d6fc7d7256d1cee490746539d475" in outcomes
    assert "tiny_cpu_bfloat16_supervised_instruct.yaml" in outcomes
    assert "Tiny-CPU-BFloat16-Supervised-Instruct-Smoke" in outcomes
    assert "DistillConfig.type supervised_instruct" in outcomes
    assert "collate_type=instruction" in outcomes
    assert "return_dict=True" in outcomes
    assert "mohawk_tiny_bfloat16_supervised_instruct_cpu_run/save" in outcomes
    assert "8c0b64ea9bc99724503935f143d7a66d93e990a3f62c6948948c7f41b2c78294" in outcomes
    assert "9c5e08cfc440165badaf7844aeb9fcfb10c8100980be9e580d51909aa8d01d6e" in outcomes
    assert "tiny_cpu_supervised_instruct_teacher_logits.yaml" in outcomes
    assert "Tiny-CPU-Supervised-Instruct-Teacher-Logits-Smoke" in outcomes
    assert "supervision True" in outcomes
    assert "generate False" in outcomes
    assert "mohawk_tiny_supervised_instruct_teacher_logits_cpu_run/save" in outcomes
    assert "3b6aa68bc5a25a6bfe23ec768b38f643044d110cd369468f1e453c49d1a63f8c" in outcomes
    assert "tiny_cpu_supervised_instruct_generate.yaml" in outcomes
    assert "Tiny-CPU-Supervised-Instruct-Generate-Smoke" in outcomes
    assert "generate True" in outcomes
    assert "generation_max_new_tokens 2" in outcomes
    assert "mohawk_tiny_supervised_instruct_generate_cpu_run/save" in outcomes
    assert "a298f6bfb00563b3e932c3de47022f97e241c1fcb47bdd9d1e4d52df94dc5198" in outcomes
    assert "tiny_cpu_eval_hstates.yaml" in outcomes
    assert "mohawk_tiny_eval_hstates_teacher_ckpt" in outcomes
    assert "No CUDA GPUs are available" in outcomes
    assert "wrapper-driven eval_hstates callback twice" in outcomes
    assert "eval_score: 1.291738" in outcomes
    assert "hstates_distance: 1.291738" in outcomes
    assert "mohawk_tiny_eval_hstates_cpu_run/save" in outcomes
    assert "b29506d2ca727b64122a5ad446bd78fb71bf5245b17a6e6cb979c1653fd86f26" in outcomes
    assert "tiny_cpu_bfloat16_eval_hstates.yaml" in outcomes
    assert "Tiny-CPU-BFloat16-Eval-HStates-Smoke" in outcomes
    assert "EvalConfig[0].Evaluation eval_hstates" in outcomes
    assert "mohawk_tiny_bfloat16_eval_hstates_cpu_run/save" in outcomes
    assert "latest_eval_hstates" in outcomes
    assert "1.2890625" in outcomes
    assert "c8ab9d7d3fa494e07c5bfd35e7b663f2876772173d1e1d3c049008cb64450aaa" in outcomes
    assert "tiny_cuda_eval_hstates.yaml" in commands
    assert "tiny_cuda_ddp_eval_hstates.yaml" in commands
    assert "tiny_cuda_fsdp_eval_hstates.yaml" in commands
    assert "Tiny-CUDA-Eval-HStates-Smoke" in outcomes
    assert "Tiny-CUDA-DDP-Eval-HStates-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Eval-HStates-Smoke" in outcomes
    assert "eval_score: 1.636902" in outcomes
    assert "eval_score: 1.662772" in outcomes
    assert "latest_eval_hstates" in outcomes
    assert "Inplace update to inference tensor outside InferenceMode" in outcomes
    assert "eval_hstates torch.no_grad() fix" in commands
    assert "wrappers fsdp fsdp" in outcomes
    assert "sharding FULL_SHARD FULL_SHARD" in outcomes
    assert "287b32e35195cb73a9a0f35493802b98f510bff5269543d6ca721750a8a2bb46" in outcomes
    assert "01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in outcomes
    assert "run.py --config configs/smoke/tiny_cuda_bfloat16_eval_hstates.yaml" in commands
    assert "torchrun --standalone --nproc_per_node=2 run.py --config configs/smoke/tiny_cuda_ddp_bfloat16_eval_hstates.yaml" in commands
    assert "torchrun --standalone --nproc_per_node=2 run.py --config configs/smoke/tiny_cuda_fsdp_bfloat16_eval_hstates.yaml" in commands
    assert "Tiny-CUDA-BFloat16-Eval-HStates-Smoke" in outcomes
    assert "Tiny-CUDA-DDP-BFloat16-Eval-HStates-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-BFloat16-Eval-HStates-Smoke" in outcomes
    assert "EvalConfig[0].Evaluation eval_hstates" in outcomes
    assert "single CUDA final/latest scheduler states had _step_count == 2" in outcomes
    assert "DDP/FSDP had _step_count == 1" in outcomes
    assert "all final/latest tensors were torch.bfloat16" in outcomes
    assert "1.640625" in outcomes
    assert "1.6640625" in outcomes
    assert "52fe58e628de69a168b351d41c8d01a52017300497a0675770ae75653ccf1ef1" in outcomes
    assert "0bc9e79b22d4fce8a926de7260a144c0a5a0ff09a86328af5f35777ab655af15" in outcomes
    assert "tiny_cpu_qwen2_supervised.yaml --output /home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_qwen2_teacher_ckpt" in commands
    assert "tiny_cpu_phi_supervised.yaml --output /home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_phi_teacher_ckpt" in commands
    assert "tiny_cpu_falcon_supervised.yaml --output /home/abick/mohawk/artifacts/gpu_smoke/tiny_cuda_falcon_teacher_ckpt" in commands
    assert "run.py --config configs/smoke/tiny_cuda_qwen2_bfloat16_eval_callbacks.yaml" in commands
    assert "run.py --config configs/smoke/tiny_cuda_phi_bfloat16_eval_callbacks.yaml" in commands
    assert "run.py --config configs/smoke/tiny_cuda_falcon_bfloat16_eval_callbacks.yaml" in commands
    assert "Tiny-CUDA-Qwen2-BFloat16-Eval-Callbacks-Smoke" in outcomes
    assert "Tiny-CUDA-Phi-BFloat16-Eval-Callbacks-Smoke" in outcomes
    assert "Tiny-CUDA-Falcon-BFloat16-Eval-Callbacks-Smoke" in outcomes
    assert "Qwen2Model -> Qwen2Block -> Qwen2Attention" in outcomes
    assert "LlamaModel -> MambaPhi -> PhiAttention" in outcomes
    assert "LlamaModel -> FalconBlock -> FalconMambaMixer" in outcomes
    assert "loaded 15/15 teacher keys" in outcomes
    assert "loaded 18/18 teacher keys" in outcomes
    assert "loaded 13/13 teacher keys" in outcomes
    assert "bfloat16 student/teacher dtypes" in outcomes
    assert "eval_hstates plus eval_ppl" in outcomes
    assert "59874.14453125" in outcomes
    assert "40134.85546875" in outcomes
    assert "62943.96875" in outcomes
    assert "8414e24632d042ced981bc7f4457205e643195d41f9ff7d6b4a0f37da58d3a19" in outcomes
    assert "e84464330dba07b4ef35e0f538d923fd4c188cae2cb1458b6cb18f93b60c66f0" in outcomes
    assert "a0297d015d195ccda5cb633c16624df99e256053a09f14d31a140073bc73d790" in outcomes
    assert "configs/smoke/tiny_cuda_ddp_qwen2_bfloat16_eval_callbacks.yaml" in commands
    assert "configs/smoke/tiny_cuda_fsdp_qwen2_bfloat16_eval_callbacks.yaml" in commands
    assert "configs/smoke/tiny_cuda_ddp_phi_bfloat16_eval_callbacks.yaml" in commands
    assert "configs/smoke/tiny_cuda_fsdp_phi_bfloat16_eval_callbacks.yaml" in commands
    assert "configs/smoke/tiny_cuda_ddp_falcon_bfloat16_eval_callbacks.yaml" in commands
    assert "configs/smoke/tiny_cuda_fsdp_falcon_bfloat16_eval_callbacks.yaml" in commands
    assert "stale configs/smoke/data eval entries" in outcomes
    assert "all six configs loaded with exactly two eval callbacks" in outcomes
    assert "Qwen2 bfloat16 eval callback path" in outcomes
    assert "Phi bfloat16 eval callback path" in outcomes
    assert "Falcon bfloat16 eval callback path" in outcomes
    assert "eval_hstates score 1.4765625" in outcomes
    assert "eval_ppl score 69068.7421875" in outcomes
    assert "eval_hstates score 1.6484375" in outcomes
    assert "eval_ppl score 50082.56640625" in outcomes
    assert "eval_hstates score 0.99609375" in outcomes
    assert "eval_ppl score 53790.79296875" in outcomes
    assert "all final/latest model tensors were torch.bfloat16" in outcomes
    assert "scheduler states had _step_count == 1" in outcomes
    assert "c95790cea5ec1fe21235972050d52293b981a21fb4f50bfd1acb3e9f308511ba" in outcomes
    assert "82d94929797662c9abc5b7ecb20962f82dfef0c41d252be6f5a4be8e31045e3b" in outcomes
    assert "ef48a53d4ba3941435096b197361619301b7560ef96322543ab0ece75b3c3dde" in outcomes
    assert "tiny_cpu_eval_ppl.yaml" in outcomes
    assert "mohawk_tiny_eval_ppl_teacher_ckpt" in outcomes
    assert "EvalConfig -> eval_ppl" in outcomes
    assert "NotImplementedError" in outcomes
    assert "state_dict()" in outcomes
    assert "wrapper-driven eval_ppl callback twice" in outcomes
    assert "Dataloader state is not implemented" in outcomes
    assert "latest_Perplexity" in outcomes
    assert "mohawk_tiny_eval_ppl_cpu_run/save" in outcomes
    assert "46137.34375" in outcomes
    assert "ada2086c37b7c885e663df2589d4aa4a318178e2163906772a2efd4eaa8c36b0" in outcomes
    assert "run.py --config configs/smoke/tiny_cuda_eval_ppl.yaml" in commands
    assert "Tiny-CUDA-Eval-PPL-Smoke" in outcomes
    assert "student and teacher devices cuda:0" in outcomes
    assert "final/latest scheduler states had _step_count == 2" in outcomes
    assert "final/latest tensors were all torch.float32" in outcomes
    assert "46136.37890625" in outcomes
    assert "287b32e35195cb73a9a0f35493802b98f510bff5269543d6ca721750a8a2bb46" in outcomes
    assert "torchrun --standalone --nproc_per_node=2 run.py --config configs/smoke/tiny_cuda_ddp_eval_ppl.yaml" in commands
    assert "rank 1 without an eval batch" in outcomes
    assert "Slurm cancelled the step for time limit" in outcomes
    assert "Tiny-CUDA-DDP-Eval-PPL-Smoke" in outcomes
    assert "student and teacher wrappers ddp" in outcomes
    assert "eval_score: 53548.183594" in outcomes
    assert "saved config.yaml had Tiny-CUDA-DDP-Eval-PPL-Smoke" in outcomes
    assert "eval data configs/smoke/data_ddp" in outcomes
    assert "53548.18359375" in outcomes
    assert "torchrun --standalone --nproc_per_node=2 run.py --config configs/smoke/tiny_cuda_fsdp_eval_ppl.yaml" in commands
    assert "RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed" in outcomes
    assert "all-rank evaluator checkpointing" in commands
    assert "eval_ppl torch.no_grad() fix" in commands
    assert "saved config.yaml had Tiny-CUDA-FSDP-Eval-PPL-Smoke" in outcomes
    assert "wrappers fsdp fsdp" in outcomes
    assert "sharding FULL_SHARD FULL_SHARD" in outcomes
    assert "run.py --config configs/smoke/tiny_cuda_bfloat16_eval_ppl.yaml" in commands
    assert "torchrun --standalone --nproc_per_node=2 run.py --config configs/smoke/tiny_cuda_ddp_bfloat16_eval_ppl.yaml" in commands
    assert "torchrun --standalone --nproc_per_node=2 run.py --config configs/smoke/tiny_cuda_fsdp_bfloat16_eval_ppl.yaml" in commands
    assert "Tiny-CUDA-BFloat16-Eval-PPL-Smoke" in outcomes
    assert "Tiny-CUDA-DDP-BFloat16-Eval-PPL-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-BFloat16-Eval-PPL-Smoke" in outcomes
    assert "TrainConfig.model_dtype bfloat16" in outcomes
    assert "TeacherConfig.model_dtype bfloat16" in outcomes
    assert "single CUDA final/latest scheduler states had _step_count == 2" in outcomes
    assert "DDP/FSDP had _step_count == 1" in outcomes
    assert "all final/latest tensors were torch.bfloat16" in outcomes
    assert "49020.80859375" in outcomes
    assert "55746.6015625" in outcomes
    assert "52fe58e628de69a168b351d41c8d01a52017300497a0675770ae75653ccf1ef1" in outcomes
    assert "0bc9e79b22d4fce8a926de7260a144c0a5a0ff09a86328af5f35777ab655af15" in outcomes
    assert "tiny_cpu_eval_ppl_frequency.yaml" in outcomes
    assert "frequency: 1" in outcomes
    assert "mohawk_tiny_eval_ppl_frequency_cpu_run/save" in outcomes
    assert "tiny_cpu_eval_ppl_save_best.yaml" in outcomes
    assert "did not create /tmp/mohawk_tiny_eval_ppl_save_best_cpu_run/save/best_Perplexity" in outcomes
    assert "missing current best as best" in outcomes
    assert "Saving best model with Perplexity score of 46137.34375" in outcomes
    assert "best_Perplexity.txt" in outcomes
    assert "no dataloader_state_dict.pth was written" in outcomes
    assert "tiny_cpu_eval_multi.yaml" in outcomes
    assert "two EvalConfig entries" in outcomes
    assert "wikitext 206549.081664" in outcomes
    assert "mohawk_tiny_eval_multi_cpu_run/save" in outcomes
    assert "tiny_cpu_eval_ppl_hfdata_save_latest.yaml" in outcomes
    assert "public HFDataset caches offline" in outcomes
    assert "29/29 cached sshleifer/tiny-gpt2 teacher keys" in outcomes
    assert "eval_score: 46154.773438" in outcomes
    assert "mohawk_tiny_eval_ppl_hfdata_cpu_run/save" in outcomes
    assert "dataloader_state_dict.pth" in outcomes
    assert "'_index': 2" in outcomes
    assert "46154.7734375" in outcomes
    assert "tiny_cpu_bfloat16_eval_ppl_hfdata_save_latest.yaml" in outcomes
    assert "Tiny-CPU-BFloat16-Eval-PPL-HFData-SaveLatest-Smoke" in outcomes
    assert "mohawk_tiny_bfloat16_eval_ppl_hfdata_cpu_run/save" in outcomes
    assert "TrainConfig.model_dtype bfloat16" in outcomes
    assert "TeacherConfig.model_dtype bfloat16" in outcomes
    assert "d914f7dc45244954594966c9e3bff9884f82d3947c703513c16014f2fc6d40c5" in outcomes
    assert "tiny_cpu_bfloat16_eval_ppl_hfdata_save_best.yaml" in outcomes
    assert "Tiny-CPU-BFloat16-Eval-PPL-HFData-SaveBest-Smoke" in outcomes
    assert "mohawk_tiny_bfloat16_eval_ppl_hfdata_save_best_cpu_run/save" in outcomes
    assert "save_best True" in outcomes
    assert "save_latest False" in outcomes
    assert "best_Perplexity.txt" in outcomes
    assert "tiny_cpu_eval_ppl_hfdata_save_best.yaml" in outcomes
    assert "mohawk_tiny_eval_ppl_hfdata_save_best_cpu_run/save" in outcomes
    assert "best_Perplexity.txt" in outcomes
    assert "tiny_cpu_hfdata_dataloader_resume.yaml" in outcomes
    assert "Starting NeelNanda/pile-10k from index 2" in outcomes
    assert "eval_score: 46148.125000" in outcomes
    assert "mohawk_tiny_hfdata_dataloader_resume_cpu_run/save" in outcomes
    assert "1f44aff34235041ac68bc914800030c0f2e23a8e50eaf263e9664cd001cee4d5" in outcomes
    assert "'_index': 4" in outcomes
    assert "46148.125" in outcomes
    assert "tiny_cpu_hfdata_full_resume.yaml" in outcomes
    assert "model, optimizer, scheduler, and public HFDataset state" in outcomes
    assert "12/12 student checkpoint keys" in outcomes
    assert "scheduler step 2 to 4" in outcomes
    assert "eval_score: 46159.878906" in outcomes
    assert "mohawk_tiny_hfdata_full_resume_cpu_run/save" in outcomes
    assert "_step_count == 4" in outcomes
    assert "num_steps == 4" in outcomes
    assert "'_index': 5" in outcomes
    assert "00376f568ca2fdb6abd25f26a8f66ddff1370592787f5bfb518632116cb97eee" in outcomes
    assert "46159.87890625" in outcomes
    assert "tiny_cpu_bfloat16_hfdata_full_resume.yaml" in outcomes
    assert "Tiny-CPU-BFloat16-HFData-Full-Resume-Smoke" in outcomes
    assert "mohawk_tiny_bfloat16_hfdata_full_resume_cpu_run/save" in outcomes
    assert "one bfloat16 LoadConfig.model path" in outcomes
    assert "source scheduler had _step_count == 2" in outcomes
    assert "final/latest scheduler states had _step_count == 4" in outcomes
    assert "final/latest tensors were all torch.bfloat16" in outcomes
    assert "393eb34c4f142454c1134f918595da7a997d01732109da3df810e8b49fcc3e7d" in outcomes
    assert "mohawk_tiny_sequential_hstates_cpu_run/save" in outcomes
    assert "mohawk_tiny_matrices_cpu_run/save" in outcomes
    assert "mohawk_tiny_dpo_cpu_run/save" in outcomes
    assert "mohawk_tiny_supervised_instruct_cpu_run/save" in outcomes
    assert "lazy initialization on meta" in outcomes
    assert "mohawk_tiny_lazy_cpu_run/save" in outcomes
    assert "lazy-load model hash" in outcomes
    assert "tiny_cpu_mixed_precision_resume.yaml" in outcomes
    assert "mohawk_tiny_mixed_precision_resume_cpu_run/save" in outcomes
    assert "TrainConfig.mixed_precision True" in outcomes
    assert "TeacherConfig.mixed_precision True" in outcomes
    assert "n_tokens 32" in outcomes
    assert "db56f67881c62d6bcb076b67adb34e38fea076855be99c7c0d928c07272e1e8a" in outcomes
    assert "tiny_cpu_mixed_precision_lazy_load.yaml" in outcomes
    assert "mohawk_tiny_mixed_precision_lazy_cpu_run/save" in outcomes
    assert "TrainConfig.init_fn lazy" in outcomes
    assert "e9dd521c6fef505708167e2a5f048ca8feccefbdbbf482709c52b71e887f2016" in outcomes
    assert "tiny_cpu_bfloat16_resume.yaml" in outcomes
    assert "mohawk_tiny_bfloat16_resume_cpu_run/save" in outcomes
    assert "TrainConfig.model_dtype bfloat16" in outcomes
    assert "TeacherConfig.model_dtype bfloat16" in outcomes
    assert "torch.bfloat16" in outcomes
    assert "10057436f48a3989cee46b70fb6798a05e8fdb7bd65d08b2b8f6f207075c7ce2" in outcomes
    assert "tiny_cpu_bfloat16_lazy_load.yaml" in outcomes
    assert "mohawk_tiny_bfloat16_lazy_cpu_run/save" in outcomes
    assert "a279c41e3c27638b1b658be50e25343e754d54d0c536b223808dc3b3115c38eb" in outcomes
    assert "configs/smoke/tiny_cuda_lazy_load.yaml" in commands
    assert "Tiny-CUDA-Lazy-Load-Smoke" in outcomes
    assert "Slurm job 688316" in outcomes
    assert "loaded 12/12 student keys" in outcomes
    assert "cuda:0" in outcomes
    assert "mohawk/artifacts/gpu_smoke/tiny_cuda_lazy_load/save" in outcomes
    assert "source hash 287b32e35195cb73a9a0f35493802b98f510bff5269543d6ca721750a8a2bb46" in outcomes
    assert "lazy-load hash 52729570d2fecd2c714de7933b52ea805d1c1ab49d0c8f6cb02e27ca046360e1" in outcomes
    assert "hash_changed True" in outcomes
    assert "configs/smoke/tiny_cuda_ddp_lazy_load.yaml" in commands
    assert "configs/smoke/tiny_cuda_fsdp_lazy_load.yaml" in commands
    assert "Tiny-CUDA-DDP-Lazy-Load-Smoke" in outcomes
    assert "Tiny-CUDA-FSDP-Lazy-Load-Smoke" in outcomes
    assert "Slurm job 688353" in outcomes
    assert "Slurm job 688357" in outcomes
    assert "WORLD_SIZE=2" in outcomes
    assert "TrainConfig.init_fn lazy" in outcomes
    assert "n_tokens 32" in outcomes
    assert "effective_batch_size 2" in outcomes
    assert "configs/smoke/data_ddp" in outcomes
    assert "DDP broadcast from rank 0" in outcomes
    assert "FULL_SHARD" in outcomes
    assert "Total Grad Steps: 2" in outcomes
    assert "scheduler states had _step_count == 2" in outcomes
    assert "num_steps == 2" in outcomes
    assert "all 12 tensors changed from source" in outcomes
    assert "source model file hash 01292eb95b96e580140c4d7cea0215dac631811b27506d01a9d2f3daef330e55" in outcomes
    assert "DDP/FSDP lazy-load model file hash 14d447df4d4d3c6bfff71734e78cb36c71909b38ffb41be77c8b698739e5401b" in outcomes
    assert "CUDA lazy-load evidence update" in commands
    assert "CUDA DDP/FSDP lazy-load evidence update" in commands
    assert "6 passed" in outcomes
    assert "8 passed" in outcomes
    assert "HFDataset consumed-count resume state" in outcomes
    assert "streaming shuffle-buffer override" in outcomes
    assert "allenai/c4" in outcomes
    assert "batch_shape (1, 495)" in outcomes
    assert "shuffle_buffer_size': 2" in outcomes
    assert "tiny_cpu_c4_streaming_supervised.yaml" in outcomes
    assert "HFDataset -> Tokenize -> PaddingDataLoader -> CycleDataLoader -> TorchDataLoader" in outcomes
    assert "mohawk_tiny_c4_streaming_cpu_run/save" in outcomes
    assert "DistillConfig.name Tiny-CPU-C4-Streaming-Supervised-Smoke" in outcomes
    assert "n_batches 2" in outcomes
    assert "dataloader_state_dict.pth was absent" in outcomes
    assert "c5f9a133cbcd343d88c39edcf9b35bf83f46315f76ea3cf44c9d25257dc849dc" in outcomes
    assert "tiny_cpu_fineweb_edu_streaming_supervised.yaml" in commands
    assert "HuggingFaceFW/fineweb-edu" in outcomes
    assert "sample-100BT" in outcomes
    assert "/home/abick/storage/mohawk/fineweb_edu_streaming_smoke" in outcomes
    assert "Signal(11)" in outcomes
    assert "exit code 139" in outcomes
    assert "Fatal Python error: PyGILState_Release" in outcomes
    assert "12 torch.float32 tensors" in outcomes
    assert "b5038277318dc21293aa7aff418ea141841e1e83d59566d316439d4b502f626d" in outcomes
    assert "minimal non-Mohawk script" in outcomes
    assert "exit code 134" in outcomes
    assert "timed out after 180 seconds" in outcomes
    assert "Explicit one-row FineWeb-Edu streaming cleanup probe" in commands
    assert "exited with status 0" in outcomes
    assert "control probe without explicit cleanup" in outcomes
    assert "HFDataset.close" in commands
    assert "PyTorch _dataset_fetcher.dataset_iter closure" in commands
    assert "11 passed" in outcomes
    assert "dataloader-only probe consumed a (1, 8) batch and exited cleanly" in outcomes
    assert "real faulthandler run.py --config configs/smoke/tiny_cpu_fineweb_edu_streaming_supervised.yaml retry still exited 139" in outcomes
    assert "71b740e0d4bee1b557cb83ce2e0db4a8c1352adfed7148b369edced4956f0b3e" in outcomes
    assert "429 Too Many Requests" in outcomes
    assert "post-env-restore rerun did not reach training" in outcomes
    assert "216.228.125.131" in outcomes
    assert "not streamable data shards" in outcomes
    assert "tiny_cpu_fineweb_edu_streaming_supervised_default_runtime.yaml" in commands
    assert "Tiny-CPU-FineWeb-Edu-Streaming-Default-Runtime-Supervised-Smoke" in outcomes
    assert "/home/abick/storage/mohawk/fineweb_edu_streaming_smoke_default_runtime" in outcomes
    assert "sshleifer/tiny-gpt2 False" in outcomes
    assert "Real storage-backed default-runtime FineWeb-Edu streaming run.py smoke exited 0" in outcomes
    assert "without the shared-env shutdown crash" in outcomes
    assert "Tokenize.local_files_only False" in outcomes
    assert "9e447f7af4d9d10fbd0bab04b7078d39bc6239f1681a73cdebc2c8c7762676d2" in outcomes
    assert "default-runtime FineWeb storage 19M" in outcomes
    assert "Footprint checks after the FineWeb-Edu default-runtime evidence update" in commands
    assert "storage venvs were 6.5G" in outcomes
    assert "repo actual size 17M" in outcomes
    assert "blank env override preservation" in commands
    assert "HF_TOKEN" in outcomes
    assert "test_distill_entrypoint_empty_env_override_preserves_existing_token" in outcomes
    assert "18 passed in 92.78s" in outcomes
    assert "streaming_spec None" in outcomes
    assert "from streaming.base.util import clean_stale_shared_memory" in outcomes
    assert "C4DataLoader requires an optional dependency" in outcomes
    assert "configs/Llama/8B/llama/matrices.yaml" in outcomes
    assert "configs/Llama/8B/bases/_matrices.yaml" in outcomes
    assert "block_name LlamaBlock" in outcomes
    assert "/tmp/mohawk_cache/distillation/llama" in outcomes
    assert "configs/Phi/1.5B/phi/supervised.yaml supervised 6000000000.0 128" in outcomes
    assert "configs/Phi/1.5B/phi/hstates.yaml hstates 10000000000.0 1" in outcomes
    assert "configs/Phi/1.5B/phi/matrices.yaml matrices 1000000000.0 128" in outcomes
    assert "tdc True" in outcomes
    assert "teacher microsoft/phi-1_5" in outcomes
    assert "tiny_cpu_random_data_supervised.yaml" in outcomes
    assert "RandomDataLoader -> CycleDataLoader" in outcomes
    assert "mohawk_tiny_random_data_cpu_run/save" in outcomes
    assert "batch1_shape (1, 8)" in outcomes
    assert "batch2_shape (1, 8)" in outcomes
    assert "batch3_shape (1, 8)" in outcomes
    assert "batch_dtype torch.int64" in outcomes
    assert "[32487, 27591, 7093, 953, 10379, 5139, 38222, 10161]" in outcomes
    assert "[28812, 40410, 8215, 14673, 11984, 16660, 49908, 34954]" in outcomes
    assert "[35078, 11511, 47418, 15036, 44506, 263, 20248, 44628]" in outcomes
    assert "different_batches True" in outcomes
    assert "cycled_after_two True" in outcomes
    assert "Tiny-CPU-RandomDataLoader-Smoke" in outcomes
    assert "max_seq_len 8" in outcomes
    assert "random_train_n_tokens 16" in outcomes
    assert "random_batch_size 1" in outcomes
    assert "optimizer_groups 1" in outcomes
    assert "5f5f7f11632e49ebdeddde02acffca65dd5c876cfb8cf13e8224f2d6d666d9d9" in outcomes
    assert "tiny_cpu_shuffle_loader_supervised.yaml" in outcomes
    assert "ShuffleLoader -> CycleDataLoader" in outcomes
    assert "mohawk_tiny_shuffle_loader_cpu_run/save" in outcomes
    assert "tiny_shuffle_source_a.yaml" in outcomes
    assert "tiny_shuffle_source_b.yaml" in outcomes
    assert "batch_keys ['input_ids']" in outcomes
    assert "shapes [(1, 8), (1, 8), (1, 8), (1, 8)]" in outcomes
    assert "dtype torch.int64" in outcomes
    assert "[1477, 18137, 17130, 285, 1219, 19301, 3047, 6291]" in outcomes
    assert "[1477, 18137, 12159, 285, 1219, 19301, 3047, 6291]" in outcomes
    assert "unique_first_cycle 2" in outcomes
    assert "third_equals_second True" in outcomes
    assert "Tiny-CPU-ShuffleLoader-Smoke" in outcomes
    assert "shuffle_batch_size 1" in outcomes
    assert "torch_config_batch_size 1" in outcomes
    assert "ff21b755d76cb2a337aab8e08443b0c98fa04aed66bdd30fd0456a6a30eb6e30" in outcomes
    assert "tiny_cpu_conversation_collate.yaml" in outcomes
    assert "torch.Size([2, 15])" in outcomes
    assert "JSONIterableDataset -> Tokenize(collate_type=conversation) -> PaddingDataLoader -> CycleDataLoader -> TorchDataLoader" in outcomes
    assert "configs/smoke/tiny_chat_template.jinja" in outcomes
    assert "mohawk_tiny_conversation_collate_cpu_run/save" in outcomes
    assert "batch_keys ['input_ids']" in outcomes
    assert "batch_shape (1, 8)" in outcomes
    assert "batch_dtype torch.int64" in outcomes
    assert "[7220, 25, 13816, 23748, 13, 198, 562, 10167]" in outcomes
    assert "Tiny-CPU-Conversation-Collate-Smoke" in outcomes
    assert "collate_type conversation" in outcomes
    assert "data_dir configs/smoke/data_conversation" in outcomes
    assert "30a57cc4eff783c7161192d04b48898326f3e3738ab83440b482182355f22624" in outcomes
    assert "tiny_cpu_classic_collate.yaml" in outcomes
    assert "JSONIterableDataset -> Tokenize(collate_type=classic) -> PaddingDataLoader -> CycleDataLoader -> TorchDataLoader" in outcomes
    assert "mohawk_tiny_classic_collate_cpu_run/save" in outcomes
    assert "batch_keys ['input_ids']" in outcomes
    assert "batch_shape (1, 8)" in outcomes
    assert "batch_dtype torch.int64" in outcomes
    assert "[7220, 25, 13816, 23748, 13, 198, 50256, 50256]" in outcomes
    assert "Tiny-CPU-Classic-Collate-Smoke" in outcomes
    assert "collate_type classic" in outcomes
    assert "data_dir configs/smoke/data_classic" in outcomes
    assert "22c14fb304146274b424342dbace0e6a3691cf5924cbaf1ccd5c422a17614123" in outcomes
    assert "tiny_cpu_kv_raw_packing.yaml" in outcomes
    assert "RoundRobinLoader -> KVRetrieval -> Tokenize(collate_type=raw) -> PackingDataLoader -> CycleDataLoader -> TorchDataLoader" in outcomes
    assert "mohawk_tiny_kv_raw_packing_cpu_run/save" in outcomes
    assert "batch_keys ['attention_mask', 'input_ids', 'position_ids']" in outcomes
    assert "input_shape (1, 8)" in outcomes
    assert "position_shape (1, 8)" in outcomes
    assert "attention_shape (1, 8)" in outcomes
    assert "input_dtype torch.int64" in outcomes
    assert "[13579, 273, 1096, 262, 1708, 22155, 25, 198]" in outcomes
    assert "position_ids [0, 1, 2, 3, 4, 5, 6, 7]" in outcomes
    assert "attention_sum 8" in outcomes
    assert "Tiny-CPU-KV-Raw-Packing-Smoke" in outcomes
    assert "nested loaders ['KVRetrieval', 'Tokenize']" in outcomes
    assert "nested_collate_type raw" in outcomes
    assert "packing_max_seq_len 8" in outcomes
    assert "5f00fc0d077c74b11caa844cfdbae21102f95cf743973b3bc7cf1000115d157d" in outcomes
    assert "tiny_cpu_niah_dataset_supervised.yaml" in outcomes
    assert "NeedleInHaystackDataset -> CycleDataLoader -> TorchDataLoader" in outcomes
    assert "mohawk_tiny_niah_dataset_cpu_run/save" in outcomes
    assert "batch_keys ['input_ids', 'response_ids']" in outcomes
    assert "input_shape (1, 96)" in outcomes
    assert "response_shape (1, 3)" in outcomes
    assert "input_dtype torch.int64" in outcomes
    assert "response_dtype torch.int64" in outcomes
    assert "[32, 2041, 5536, 1271, 318, 7104, 1626, 262, 1708, 2420, 13, 6889]" in outcomes
    assert "response tokens [2327, 21495, 1959]" in outcomes
    assert "max_seq_len 96" in outcomes
    assert "pad_count 23" in outcomes
    assert "Tiny-CPU-NIAH-Dataset-Smoke" in outcomes
    assert "local_files_only True" in outcomes
    assert "type_haystack none" in outcomes
    assert "num_needle_k 1" in outcomes
    assert "tokens_to_generate 2" in outcomes
    assert "train_n_tokens 192" in outcomes
    assert "b229762d8c4f7555f8a640fd9fdd5cc186e18ab3366f76eaf077f89f7690ec1a" in outcomes
    assert "tiny_cpu_copying_task_supervised.yaml" in outcomes
    assert "CopyingTaskDataset -> CycleDataLoader -> TorchDataLoader" in outcomes
    assert "mohawk_tiny_copying_task_cpu_run/save" in outcomes
    assert "batch_shape (1, 24)" in outcomes
    assert "batch_dtype torch.int64" in outcomes
    assert "[6310, 2762, 25, 11436, 262, 1708, 3146, 13, 198, 21947, 25, 718, 11, 21, 198, 31077, 25, 718, 11, 21, 50256, 50256, 50256, 50256]" in outcomes
    assert "pad_count 4" in outcomes
    assert "Tiny-CPU-CopyingTask-Dataset-Smoke" in outcomes
    assert "max_seq_len 24" in outcomes
    assert "local_files_only True" in outcomes
    assert "train_n_tokens 48" in outcomes
    assert "optimizer_groups 1" in outcomes
    assert "411990a52399afe4302955bf8972e42471f5ee6e0169ed039e38a573a19100c9" in outcomes
    assert "tiny_cpu_sequential_loader_supervised.yaml" in outcomes
    assert "SequentialLoader -> PaddingDataLoader -> CycleDataLoader -> TorchDataLoader" in outcomes
    assert "mohawk_tiny_sequential_loader_cpu_run/save" in outcomes
    assert "batch1_shape (1, 8)" in outcomes
    assert "batch2_shape (1, 8)" in outcomes
    assert "batch_dtype torch.int64" in outcomes
    assert "[26591, 285, 1219, 19301, 8379, 50256, 50256, 50256]" in outcomes
    assert "[31361, 285, 1219, 19301, 8379, 50256, 50256, 50256]" in outcomes
    assert "different_batches True" in outcomes
    assert "SequentialLoader setting dataloader 0 and then 1" in outcomes
    assert "Tiny-CPU-Sequential-Loader-Smoke" in outcomes
    assert "max_samples 1" in outcomes
    assert "configs/smoke/data_sequence_a" in outcomes
    assert "configs/smoke/data_sequence_b" in outcomes
    assert "local_files_only [True, True]" in outcomes
    assert "padding_max_seq_len 8" in outcomes
    assert "d1e60333ba54ab9d2763b4c7e06700b24f438601aa3bdc2cd151bc6e97e75fa7" in outcomes
    assert "tiny_cpu_aggregation_loader_supervised.yaml" in outcomes
    assert "RoundRobinLoader -> PackingDataLoader -> AggregationDataLoader -> CycleDataLoader -> TorchDataLoader" in outcomes
    assert "mohawk_tiny_aggregation_loader_cpu_run/save" in outcomes
    assert "Aggregating 2 samples" in outcomes
    assert "Aggregated 2 samples" in outcomes
    assert "batch1_input_shape (1, 8)" in outcomes
    assert "batch2_input_shape (1, 8)" in outcomes
    assert "batch1_position_shape (1, 8)" in outcomes
    assert "batch2_position_shape (1, 8)" in outcomes
    assert "batch1_attention_shape (1, 8)" in outcomes
    assert "batch2_attention_shape (1, 8)" in outcomes
    assert "batch_dtype torch.int64" in outcomes
    assert "[9460, 43068, 17130, 285, 1219, 19301, 3047, 6291]" in outcomes
    assert "[9460, 43068, 12159, 285, 1219, 19301, 3047, 6291]" in outcomes
    assert "batch1_attention_sum 8" in outcomes
    assert "batch2_attention_sum 8" in outcomes
    assert "Tiny-CPU-Aggregation-Loader-Smoke" in outcomes
    assert "aggregation_size 2" in outcomes
    assert "packing_max_seq_len 8" in outcomes
    assert "configs/smoke/data_aggregation_a" in outcomes
    assert "configs/smoke/data_aggregation_b" in outcomes
    assert "55cb16302e332da1d06c9f1873af1d0c6bea4d48d13a0a082d4a6d389fb4d03d" in outcomes
    assert "mohawk_tiny_qwen2_teacher_ckpt" in outcomes
    assert "Qwen2Model -> Qwen2Block -> Qwen2Attention" in outcomes
    assert "mohawk_tiny_qwen2_cpu_run/save" in outcomes
    assert "6445128" in outcomes
    assert "028e2305f692299d7fafe7debc67a1594e60778f0e864c31297236d54f74dbcd" in outcomes
    assert "mohawk_tiny_phi_teacher_ckpt" in outcomes
    assert "Cannot copy out of meta tensor" in outcomes
    assert "mohawk_tiny_phi_cpu_run/save" in outcomes
    assert "6648836" in outcomes
    assert "d38637c90ef02b2bb710bebcd8d87945a8a27843b34170c009e4b1b9ebd77f67" in outcomes
    assert "mohawk_tiny_falcon_teacher_ckpt" in outcomes
    assert "Buffer is a Parameter" in outcomes
    assert "mohawk_tiny_falcon_cpu_run/save" in outcomes
    assert "6447768" in outcomes
    assert "553001640f86568e6379b3b6756f50601638b4f4db91d5b384a1789d6493e2a4" in outcomes
    assert "tiny_cpu_doubleblock_adapter_supervised.yaml" in outcomes
    assert "DoubleBlockVanilla" in outcomes
    assert "DoubleBlockHymba" in outcomes
    assert "DoubleBlockMerger" in outcomes
    assert "mohawk_tiny_doubleblock_adapter_cpu_run/save" in outcomes
    assert "6447392" in outcomes
    assert "6446096" in outcomes
    assert "6448040" in outcomes
    assert "6447728" in outcomes
    assert "5d8be23f394221dfdb2a79e6921b0f9f32628de652764027e3ad158530cfc584" in outcomes
    assert "4db33e20aafae91309bbb05877145d85da2c94494c9028c7c667b0fc9a780e19" in outcomes
    assert "47f48b5a504c8d16c283b4b2316b5485f006aae4c776fe74500ec61206b2b069" in outcomes
    assert "3d4ec4399d45be8ad96a52b327e86b5e12623f6911bb5a20a155625ce0953e0e" in outcomes
    assert "'_index': -1" in outcomes
    assert "'_index': 1" in outcomes
    assert "OfflineModeIsEnabled" in outcomes
    assert "read-only filesystem" in outcomes
    assert "No module named 'cartesia_pytorch'" in outcomes
    assert "tiny_cpu_hybrid_transfer.yaml" in outcomes
    assert "hf-internal-testing/tiny-random-LlamaForCausalLM" in outcomes
    assert "7 matching student preload keys" in outcomes
    assert "21/21 real tiny Llama teacher keys" in outcomes
    assert "DoubleBlockAdapter" in outcomes
    assert "mixer2.self_attn" in outcomes
    assert "4116640" in outcomes
    assert "'q': 0.0" in outcomes
    assert "f27c6a8cc74fa11e918d8169ece052ccb0489337a3b60d08f6b0cf4ed805d4b6" in outcomes
    assert "universal CUDA requirement" in outcomes
    assert "requires CUDA because it benchmarks CUDA graph generation throughput" in outcomes
    assert "No throughput benchmark ran because CUDA is unavailable" in outcomes
    assert "srun -p batch -A nemotron_arch_dev" in commands
    assert "progressed past the early CUDA guard" in outcomes
    assert "no model, tokenizer, or throughput benchmark ran" in outcomes
    assert "tools/benchmark_throughput.py --model codestral" in commands
    assert "tools/benchmark_throughput.py --model falcon" in commands
    assert "Reached the real Codestral-Mamba throughput branch" in outcomes
    assert "Reached the real Falcon-Mamba throughput branch" in outcomes
    assert "mistarl_to_mamba -> get_mamba_classes" in outcomes
    assert "falcon_to_mamba -> get_mamba_classes" in outcomes
    assert "tools/visualize_attention.py --model_name sshleifer/tiny-gpt2" in commands
    assert "tools/visualize_attention.py --model_name hf-internal-testing/tiny-random-LlamaForCausalLM" in commands
    assert "local cache for hf-internal-testing/tiny-random-LlamaForCausalLM contained only config.json" in outcomes
    assert "Network-approved" in commands
    assert "Downloaded/loaded the public tiny random Llama model and tokenizer" in outcomes
    assert "loaded the same public tiny random Llama model/tokenizer offline" in outcomes
    assert "PNG image data, 256 x 622" in outcomes
    assert "593bb14787d14db3ca6f631649aaf8cf3858cd9a455a8115e7253210a1328806" in outcomes
    assert "visualize_attention_cuda" in commands
    assert "--device cuda" in commands
    assert "Device: cuda" in outcomes
    assert "db3c010fae59d7418ba0cd40ed91eb5c5a483e58d00d3682e991c46e5edd18ef" in outcomes
    assert "storage-backed output directory used 76K" in outcomes
    assert "tools/visualize_attention.py --model_name meta-llama/Llama-3.2-3B-Instruct" in commands
    assert "visualize_attention_representative" in commands
    assert "LocalEntryNotFoundError" in outcomes
    assert "outgoing Hugging Face traffic was disabled" in outcomes
    assert "run.py --config configs/Llama/8B/llama/hstates.yaml" in commands
    assert "Real representative Llama 8B hstates run.py probe launched" in outcomes
    assert "HuggingFaceTB/finemath" in outcomes
    assert "PackingDataLoader.max_seq_len: 2048" in outcomes
    assert "ConnectionError: Couldn't reach 'HuggingFaceFW/fineweb-edu'" in outcomes
    assert "before model init, teacher load, optimizer construction, or training" in outcomes
    assert "run.py --config configs/Llama/8B/llama/hstates.yaml,configs/Llama/8B/llama/hstates.yaml" in commands
    assert "Real representative comma-separated run.py probe launched" in outcomes
    assert "parsed and loaded both Llama 8B hstates config entries" in outcomes
    assert "second-config execution" in outcomes
    assert "tiny_cpu_hf_teacher_supervised.yaml" in outcomes
    assert "tiny_cpu_public_hfdata_supervised.yaml" in outcomes
    assert "read-only home cache" in outcomes
    assert "HF_DATASETS_CACHE" in outcomes
    assert "Hugging Face DNS" in outcomes
    assert "LocalEntryNotFoundError" in outcomes
    assert "isolated /tmp/mohawk_tiny_hf_teacher_cpu_run/huggingface/hub" in outcomes
    assert "29 loaded keys" in outcomes
    assert "29/29 real HF teacher keys" in outcomes
    assert "mohawk_tiny_hf_teacher_cpu_run/save" in outcomes
    assert "6a45a7b408356f127bf72d89ea33e2a7ea045627fa0b92475f86345c601e9a9e" in outcomes
    assert "mohawk_tiny_public_hfdata_cpu_run/save" in outcomes
    assert "110821dc9516f4eabc3c66ea9fb20697e40803f889786bc257f4deed9e1b732f" in outcomes
    assert "tiny_cpu_gradient_accumulation.yaml" in outcomes
    assert "effective_batch_size == 2" in outcomes
    assert "accumulation_steps == 2" in outcomes
    assert "n_batches == 4" in outcomes
    assert "db0807a38e96f60b84e2aa68ad3eb9d3a05f7b14091df541d9fe473e083b7b2c" in outcomes
    assert "tiny_cpu_wsd_scheduler.yaml" in outcomes
    assert "OptimizerConfig.scheduler.name: wsd" in outcomes
    assert "Warmup Steps: 1" in outcomes
    assert "Decay Steps: 1" in outcomes
    assert "Total Grad Steps: 4" in outcomes
    assert "Tiny-CPU-WSD-Scheduler-Smoke" in outcomes
    assert "scheduler _step_count == 4" in outcomes
    assert "last_epoch == 3" in outcomes
    assert "num_steps == 4" in outcomes
    assert "_last_lr == [0.0001]" in outcomes
    assert "cbaff7bf8a10732de5205340eaa3943ffe06371c400984ef855ed50541a1751b" in outcomes
    assert "tiny_cpu_adam_optimizer.yaml" in outcomes
    assert "OptimizerConfig.optimizer: Adam" in outcomes
    assert "Optimizer configuration: Adam (" in outcomes
    assert "decoupled_weight_decay: False" in outcomes
    assert "Tiny-CPU-Adam-Optimizer-Smoke" in outcomes
    assert "optimizer Adam" in outcomes
    assert "param_groups 1" in outcomes
    assert "group_betas (0.9, 0.95)" in outcomes
    assert "state_entries 12" in outcomes
    assert "['exp_avg', 'exp_avg_sq', 'step']" in outcomes
    assert "e153eb131e5bbbf7f1e1ad9913dcfc6082fe5a8c5c95822768d10a58ba653687" in outcomes
    assert "tiny_cpu_optimize_weights_whitelist.yaml" in outcomes
    assert "tiny_cpu_optimize_weights_blacklist.yaml" in outcomes
    assert "Trainable parameters: 804,112 / 1,610,832" in outcomes
    assert "Trainable parameters: 806,720 / 1,610,832" in outcomes
    assert "optimizer_params 1" in outcomes
    assert "optimizer_params 11" in outcomes
    assert "changed only lm_head.weight" in outcomes
    assert "left only lm_head.weight unchanged" in outcomes
    assert "6183032e556db32bb7e3ad59d801530f04116a4551c6efe0ede45681f889dbd2" in outcomes
    assert "6351426c0d02e1e1471edc590934932daa404c8ee15dd0603f0bf74409cfa536" in outcomes
    assert "tiny_cpu_load_model_whitelist.yaml" in outcomes
    assert "tiny_cpu_load_model_blacklist.yaml" in outcomes
    assert "LoadConfig.model.white_list: [lm_head]" in outcomes
    assert "LoadConfig.model.black_list: [lm_head]" in outcomes
    assert "Loaded keys: 1" in outcomes
    assert "Missing keys: 11" in outcomes
    assert "Loaded keys: 11" in outcomes
    assert "Missing keys: 1" in outcomes
    assert "source_equal_count 1" in outcomes
    assert "source_equal ['lm_head.weight']" in outcomes
    assert "source_different_count 11" in outcomes
    assert "source_equal_count 11" in outcomes
    assert "source_different ['lm_head.weight']" in outcomes
    assert "expected_match True" in outcomes
    assert "8d01e9141c1752f14212292b412825b9a8ff17dff7bbc076f5eb726f342780f3" in outcomes
    assert "cfcfef52e5be35c99f25758f4bc17e72464245521ba45d2f946eb950a0fec015" in outcomes
    assert "tiny_cpu_load_model_rename.yaml" in outcomes
    assert "--rename-key lm_head.weight=renamed_lm_head.weight" in outcomes
    assert "lm_head.weight absent" in outcomes
    assert "renamed_lm_head.weight present" in outcomes
    assert "LoadConfig.model.rename: {renamed_lm_head: lm_head}" in outcomes
    assert "source_has_lm_head False" in outcomes
    assert "source_has_renamed True" in outcomes
    assert "renamed_equal True" in outcomes
    assert "equal_original_names_count 0" in outcomes
    assert "6776e8e1f62c8bc6372e9afffbb21bd78bd38cc3f78e18c4e77f7fac216daaa6" in outcomes
    assert "tiny_cpu_load_model_sequence.yaml" in outcomes
    assert "base_has_lm_head True" in outcomes
    assert "alt_has_lm_head False" in outcomes
    assert "alt_has_renamed True" in outcomes
    assert "head_equal_original False" in outcomes
    assert "head_max_abs_diff 0.49912628531455994" in outcomes
    assert "Loaded 12 keys in total" in outcomes
    assert "first_black_list ['lm_head']" in outcomes
    assert "second_rename {'renamed_lm_head': 'lm_head'}" in outcomes
    assert "optimizer_lr 0.0" in outcomes
    assert "backbone_equal_count 11" in outcomes
    assert "backbone_mismatches []" in outcomes
    assert "head_equal_alt True" in outcomes
    assert "head_equal_base False" in outcomes
    assert "head_alt_base_equal False" in outcomes
    assert "221881260f780b853e481ae250d48221d1ab120b8b367f8219588a8f7936ba12" in outcomes
    assert "tiny_cpu_load_model_strict_missing.yaml" in outcomes
    assert "tiny_cpu_load_model_strict_unexpected.yaml" in outcomes
    assert "Expected-failure strict missing-key guard" in outcomes
    assert "Expected-failure strict unexpected-key guard" in outcomes
    assert "allow_missing_keys: false" in outcomes
    assert "allow_unexpected_keys: false" in outcomes
    assert "AssertionError: Missing keys" in outcomes
    assert "AssertionError: Unexpected keys: ['renamed_lm_head.weight']" in outcomes
    assert "no teacher setup, training loop, optimizer step, or model checkpoint save ran" in outcomes
    assert "only the early config.yaml artifact" in outcomes
    assert "did not write model.safetensors" in outcomes
    assert "tiny_cpu_mixed_precision.yaml" in outcomes
    assert "TrainConfig.mixed_precision: true" in outcomes
    assert "TeacherConfig.mixed_precision: true" in outcomes
    assert "train_mixed_precision True" in outcomes
    assert "teacher_mixed_precision True" in outcomes
    assert "3375940e29d65712f3426bf2cac70ff85c2365d1b8a590c6554d1edd9206f163" in outcomes
    assert "tiny_cpu_mixed_precision_gradient_accumulation.yaml" in outcomes
    assert "effective_batch_size 2" in outcomes
    assert "batch_size 1" in outcomes
    assert "n_tokens 32" in outcomes
    assert "mohawk_tiny_mixed_precision_gradient_accumulation_cpu_run/save" in outcomes
    assert "524e21de7874c2a5a29f99ac5dbfe0ec85583b516af001aa205cdd14708b233a" in outcomes
    assert "tiny_cpu_bfloat16_mixed_precision_fallback.yaml" in outcomes
    assert "float != c10::BFloat16" in outcomes
    assert "Cannot use mixed precision with bfloat16" in outcomes
    assert "all saved model tensors were torch.bfloat16" in outcomes
    assert "3222952" in outcomes
    assert "a2943247863988b07f553af2359eaaab936674d8c76148ffa7801a27390b4754" in outcomes
    assert "tiny_cpu_bfloat16_gradient_accumulation_fallback.yaml" in outcomes
    assert "mohawk_tiny_bfloat16_gradient_accumulation_cpu_run/save" in outcomes
    assert "train True bfloat16" in outcomes
    assert "teacher True bfloat16" in outcomes
    assert "0b1885aa63aa315e61d7c244fa188375bdc5d0405a54466f1b1ade0f8322a1d1" in outcomes
    assert "tiny_cpu_qwen2_bfloat16_gradient_accumulation_fallback.yaml" in outcomes
    assert "tiny_cpu_phi_bfloat16_gradient_accumulation_fallback.yaml" in outcomes
    assert "tiny_cpu_falcon_bfloat16_gradient_accumulation_fallback.yaml" in outcomes
    assert "mohawk_tiny_qwen2_bfloat16_gradient_accumulation_cpu_run/save" in outcomes
    assert "mohawk_tiny_phi_bfloat16_gradient_accumulation_cpu_run/save" in outcomes
    assert "mohawk_tiny_falcon_bfloat16_gradient_accumulation_cpu_run/save" in outcomes
    assert "de79eb48388fc3e1fc2f0d51090287bf56432378819edb401638977063e98c3b" in outcomes
    assert "b33c9800b8805dac077546f9352be1850d1a16358a5000a4aafcaed15b7cb254" in outcomes
    assert "caa827ffba2775c917dae416fe5cf660d925c03f43bb3fbd4af977f7a9b4a25e" in outcomes
    assert "tiny_cpu_qwen2_bfloat16_mixed_precision_fallback.yaml" in outcomes
    assert "mat1 and mat2 must have the same dtype" in outcomes
    assert "Float and BFloat16" in outcomes
    assert "15/15 keys" in outcomes
    assert "mohawk_tiny_qwen2_bfloat16_fallback_cpu_run/save" in outcomes
    assert "3223384" in outcomes
    assert "34fba47d457715fc90c45cc81cc3f4a187ba9e55661f0352abef53057395b9e0" in outcomes
    assert "tiny_cpu_phi_bfloat16_mixed_precision_fallback.yaml" in outcomes
    assert "BFloat16 and Float" in outcomes
    assert "18/18 keys" in outcomes
    assert "mohawk_tiny_phi_bfloat16_fallback_cpu_run/save" in outcomes
    assert "3325370" in outcomes
    assert "bf5232b363bc71eb665857afafc2c644c469291e7ccfa25611778b5999ddba8d" in outcomes
    assert "tiny_cpu_falcon_bfloat16_mixed_precision_fallback.yaml" in outcomes
    assert "13/13 keys" in outcomes
    assert "fp32 residual path returned float32 hidden states" in outcomes
    assert "mohawk_tiny_falcon_bfloat16_fallback_cpu_run/save" in outcomes
    assert "3224552" in outcomes
    assert "6493528660f88e271dda8b25f649b49581b5b1852c70ac7498aa17137e3a32be" in outcomes
    assert "tiny_cpu_compile_model.yaml" in outcomes
    assert "torch.compile Inductor" in outcomes
    assert "AttributeError: 'MixerModel' object has no attribute 'embedding'" in outcomes
    assert "_orig_mod." in outcomes
    assert "orig_mod_prefix False" in outcomes
    assert "load_missing []" in outcomes
    assert "load_unexpected []" in outcomes
    assert "44640dc7493a12ffd96e65774835a60a466383b921a0ff81831c69e9d059c439" in outcomes
    assert "tiny_cpu_bfloat16_compile_model.yaml" in outcomes
    assert "mohawk_tiny_bfloat16_compile_model_cpu_run/save" in outcomes
    assert "train_compile True" in outcomes
    assert "teacher_compile False" in outcomes
    assert "train True bfloat16" in outcomes
    assert "teacher True bfloat16" in outcomes
    assert "all saved model tensors were torch.bfloat16" in outcomes
    assert "3e606413e6aa7465dd7dc6ea0677f3ecb89f449185e7768ef5b7fac96c454695" in outcomes
    assert "tiny_cpu_teacher_compile_model.yaml" in outcomes
    assert "mohawk_tiny_teacher_compile_model_cpu_run/save" in outcomes
    assert "TrainConfig.compile_model: false" in outcomes
    assert "TeacherConfig.compile_model: true" in outcomes
    assert "set inference parameters non-trainable" in outcomes
    assert "train_compile False" in outcomes
    assert "teacher_compile True" in outcomes
    assert "all saved model tensors were torch.float32" in outcomes
    assert "8c4c51efd1bdd1966139a863087e7b5d726eb42e6d942f090dcc8f50ddac9227" in outcomes
    assert "tiny_cpu_student_teacher_compile_model.yaml" in outcomes
    assert "mohawk_tiny_student_teacher_compile_model_cpu_run/save" in outcomes
    assert "compiled both student and teacher" in outcomes
    assert "4a8fb367c41906c5fee49cd786c47844157fe6ccaf9a5fb7005e5f9010a64b7d" in outcomes
    assert "tiny_cpu_bfloat16_teacher_compile_model.yaml" in outcomes
    assert "mohawk_tiny_bfloat16_teacher_compile_model_cpu_run/save" in outcomes
    assert "compiled the bfloat16 teacher" in outcomes
    assert "d92f42c1d8bf9b50810ec3369ab911f89688db9491f1d8a00e6b2c7976ab5c5a" in outcomes
    assert "tiny_cpu_bfloat16_student_teacher_compile_model.yaml" in outcomes
    assert "mohawk_tiny_bfloat16_student_teacher_compile_model_cpu_run/save" in outcomes
    assert "1637d7e9f7f2822aafc77b631f21b8b073ec08169a4c28a4d182509d33bbf5f1" in outcomes
    assert "chosen_ids" in outcomes
    assert "response_ids" in outcomes
    assert "Dimension specified as 0" in outcomes
    assert "tokenizer.chat_template is not set" in outcomes
    assert "output_attentions=True" in outcomes
    assert "scheduler _step_count == 4" in outcomes
    assert "num_steps == 4" in outcomes
    assert "e153eb131e5bbbf7f1e1ad9913dcfc6082fe5a8c5c95822768d10a58ba653687" in outcomes
    assert "d09f21632232ce28b91f5f0e5150859c8279071816c5462e04d9af337cab4584" in outcomes
    assert "PyTorch rendezvous TCP store" in outcomes
    assert "WORLD_SIZE=2" in outcomes
    assert "Mohawk training requires CUDA" in outcomes
    assert "No process group, DDP wrapper, FSDP wrapper, optimizer step, or checkpoint save ran" in outcomes
    assert "two real supervised training runs sequentially" in outcomes
    assert "tiny_cpu_supervised.yaml,configs/smoke/tiny_cpu_eval_ppl.yaml" in outcomes
    assert "real comma-separated train+eval callback chain" in outcomes
    assert "EvalConfig -> eval_ppl" in outcomes
    assert "latest_Perplexity.txt" in outcomes
    assert "46137.34375" in outcomes
    assert "ada2086c37b7c885e663df2589d4aa4a318178e2163906772a2efd4eaa8c36b0" in outcomes
    assert "tiny_cuda_comma_supervised.yaml,configs/smoke/tiny_cuda_comma_eval_ppl.yaml" in commands
    assert "Tiny-CUDA-Comma-Supervised-Smoke" in outcomes
    assert "Tiny-CUDA-Comma-Eval-PPL-Smoke" in outcomes
    assert "Real Slurm job 689748" in outcomes
    assert "EvalConfig -> eval_ppl" in outcomes
    assert "46136.37890625" in outcomes
    assert "4c5a6e0dcdd8b17cba44e0bd7b364518dca115d046f8753ae058cea651b88f29" in outcomes
