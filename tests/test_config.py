from pathlib import Path

import yaml

from utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigSyntaxLoader(yaml.FullLoader):
    pass


ConfigSyntaxLoader.add_constructor(
    "!load_yaml",
    lambda loader, node: {"__load_yaml__": loader.construct_scalar(node)},
)


def test_all_yaml_files_parse():
    failures = []
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        try:
            yaml.load(path.read_text(), Loader=ConfigSyntaxLoader)
        except Exception as exc:  # pragma: no cover - assertion reports details
            failures.append(f"{path.relative_to(ROOT)}: {type(exc).__name__}: {exc}")

    assert failures == []


def test_documented_starting_point_configs_load():
    for path in [
        "configs/Qwen2/1.5B/hybrid/adapter.yaml",
        "configs/Llama/1B/hybrid/mohawk_8.yaml",
        "configs/Llama/8B/bases/_supervised.yaml",
    ]:
        cfg = load_config(str(ROOT / path), CONSTANTS={"slurm_job_id": "0"})
        assert cfg.DistillConfig.type


def test_fresh_mamba23_smoke_config_resolves_fast_path_and_storage_paths():
    cfg = load_config(
        str(
            ROOT
            / "configs/smoke/tiny_cuda_doubleblock_discrete_mamba2_fast_mamba23_supervised.yaml"
        ),
        CONSTANTS={"slurm_job_id": "0"},
    )
    block = cfg.ComponentsConfig.MixerModel.Blocks[0]

    assert cfg.DistillConfig.name == (
        "Tiny-CUDA-DoubleBlockAdapter-DiscreteMamba2-Fast-Mamba23-Smoke"
    )
    assert block.name == "DoubleBlockAdapter"
    assert block.ssm_layer.name == "DiscreteMamba2"
    assert block.ssm_layer.input.use_ref_impl is False
    assert cfg.ManagementConfig.paths.base_dir == (
        "/home/abick/storage/mohawk/artifacts/gpu_smoke/"
        "tiny_cuda_doubleblock_discrete_mamba2_fast_mamba23"
    )
    assert cfg.ManagementConfig.env_vars.TRITON_CACHE_DIR == (
        "/home/abick/storage/mohawk/tmp/triton-mamba23-fast-smoke"
    )


def test_distributed_fineweb_direct_shard_configs_resolve_production_path():
    shard_url = (
        "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/"
        "main/sample/100BT/000_00000.parquet"
    )
    scenarios = [
        ("ddp", "NO_SHARD"),
        ("fsdp", "FULL_SHARD"),
    ]

    for wrapper, sharding_strategy in scenarios:
        cfg = load_config(
            str(
                ROOT
                / (
                    "configs/smoke/tiny_cuda_"
                    f"{wrapper}_fineweb_edu_direct_shard_supervised.yaml"
                )
            ),
            CONSTANTS={"slurm_job_id": "0"},
        )

        assert cfg.DistillConfig.name == (
            f"Tiny-CUDA-{wrapper.upper()}-FineWeb-Edu-Direct-Shard-"
            "Supervised-Smoke"
        )
        assert cfg.TrainConfig.wrapper_type == wrapper
        assert cfg.TeacherConfig.wrapper_type == wrapper
        assert cfg.TrainConfig.sharding_strategy == sharding_strategy
        assert cfg.TeacherConfig.sharding_strategy == sharding_strategy
        assert cfg.TrainConfig.n_tokens == 8192
        assert cfg.TrainConfig.effective_batch_size == 2
        assert cfg.TrainDataConfig.loaders == [
            "HFDataset",
            "Tokenize",
            "PackingDataLoader",
            "TorchDataLoader",
        ]
        assert cfg.TrainDataConfig.HFDataset.path == "parquet"
        assert cfg.TrainDataConfig.HFDataset.name is None
        assert cfg.TrainDataConfig.HFDataset.streaming is True
        assert cfg.TrainDataConfig.HFDataset.data_files.to_dict() == {
            "train": [shard_url]
        }
        assert cfg.TrainDataConfig.PackingDataLoader.max_seq_len == 2048
        assert cfg.TrainDataConfig.TorchDataLoader.batch_size == 1
        assert cfg.ManagementConfig.paths.base_dir == (
            "/home/abick/storage/mohawk/fineweb_edu_direct_shard_cuda_"
            f"{wrapper}"
        )


def test_production_qwen2_hybrid_configs_resolve_real_architecture_and_tokenizer():
    transfer_path = (
        ROOT
        / "configs/smoke/production_qwen2_1_5b_hybrid_arch40_transfer.yaml"
    )
    train_path = (
        ROOT
        / "configs/smoke/production_qwen2_1_5b_hybrid_arch40_supervised.yaml"
    )

    transfer_cfg = load_config(str(transfer_path), CONSTANTS={"slurm_job_id": "0"})
    train_cfg = load_config(str(train_path), CONSTANTS={"slurm_job_id": "0"})
    blocks = transfer_cfg.ComponentsConfig.MixerModel.Blocks

    assert transfer_cfg.ComponentsConfig.input.vocab_size == 151936
    assert transfer_cfg.ComponentsConfig.MixerModel.input.d_model == 1536
    assert sum(block.n_layers for block in blocks) == 28
    assert sum(
        block.n_layers * block.attn_layer.input.num_attention_heads
        for block in blocks
    ) == 40
    assert all(block.name == "DoubleBlockAdapter" for block in blocks)
    assert all(block.ssm_layer.name == "DiscreteMamba2" for block in blocks)
    assert all(block.ssm_layer.input.use_ref_impl is False for block in blocks)
    assert transfer_cfg.TrainConfig.model_dtype == "bfloat16"
    assert transfer_cfg.TrainConfig.attn_implementation == "eager"
    assert transfer_cfg.TeacherConfig.dir == "Qwen/Qwen2.5-1.5B-Instruct"
    assert transfer_cfg.TrainConfig.tokenizer == "Qwen/Qwen2.5-1.5B-Instruct"
    assert transfer_cfg.ManagementConfig.env_vars.HF_HUB_CACHE == (
        "/home/abick/storage/mohawk/qwen2_1_5b_hf_cache_20260630"
    )

    assert train_cfg.LoadConfig.model[0].path == (
        "/home/abick/storage/mohawk/artifacts/"
        "production_qwen2_1_5b_arch40_transfer/save"
    )
    assert train_cfg.TrainConfig.n_tokens == 256
    assert train_cfg.OptimizerConfig.betas == [0.9, 0.95]
    assert train_cfg.TrainDataConfig.loaders == [
        "JSONIterableDataset",
        "Tokenize",
        "PaddingDataLoader",
        "CycleDataLoader",
        "TorchDataLoader",
    ]
    assert train_cfg.TrainDataConfig.Tokenize.local_files_only is True
    assert train_cfg.TrainDataConfig.PaddingDataLoader.max_seq_len == 128


def test_production_llama3_2_3b_arch20_probe_resolves_full_hybrid_shape():
    cfg = load_config(
        str(
            ROOT
            / "configs/smoke/production_llama3_2_3b_hybrid_arch20_probe.yaml"
        ),
        CONSTANTS={"slurm_job_id": "0"},
    )
    blocks = cfg.ComponentsConfig.MixerModel.Blocks
    mixer_input = cfg.ComponentsConfig.MixerModel.input

    assert cfg.DistillConfig.name == (
        "Production-Llama-3.2-3B-ARCH20-Fast-Hybrid-Probe"
    )
    assert cfg.ComponentsConfig.input.vocab_size == 128256
    assert mixer_input.d_model == 3072
    assert mixer_input.n_layer == 28
    assert mixer_input.num_attention_heads == 24
    assert mixer_input.num_key_value_heads == 8
    assert mixer_input.head_dim == 128
    assert mixer_input.max_position_embeddings == 131072
    assert sum(block.n_layers for block in blocks) == 28
    assert sum(
        block.n_layers * block.attn_layer.input.num_attention_heads
        for block in blocks
    ) == 20
    assert all(block.name == "DoubleBlockAdapter" for block in blocks)
    assert all(block.ssm_layer.name == "DiscreteMamba2" for block in blocks)
    assert all(block.ssm_layer.input.use_ref_impl is False for block in blocks)
    assert cfg.TrainConfig.model_dtype == "bfloat16"
    assert cfg.TrainConfig.attn_implementation == "eager"
    assert cfg.TeacherConfig.dir == "meta-llama/Llama-3.2-3B-Instruct"
    assert cfg.LoadConfig.model == []
    assert cfg.ManagementConfig.paths.base_dir == (
        "/home/abick/storage/mohawk/artifacts/"
        "production_llama3_2_3b_arch20_probe"
    )


def test_qwen2_mixer_configs_provide_rotary_fields():
    required_fields = {
        "norm_epsilon": 1e-6,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000,
        "head_dim": 128,
    }
    paths = sorted((ROOT / "configs/Qwen2/1.5B/hybrid").glob("architecture_*.yaml"))
    paths.append(ROOT / "configs/Qwen2/1.5B/pure/components.yaml")

    for path in paths:
        cfg = yaml.load(path.read_text(), Loader=ConfigSyntaxLoader)
        mixer_input = cfg["ComponentsConfig"]["MixerModel"]["input"]
        for field, expected in required_fields.items():
            assert mixer_input[field] == expected, path.relative_to(ROOT)


def test_load_inheritance_interpolation_load_yaml_and_list_reset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("dataset.yaml").write_text("Dataset:\n  name: local\n")
    Path("base.yaml").write_text(
        "\n".join(
            [
                "A:",
                "  x: base",
                "Items: [one, two]",
                "AppendOnly: [base]",
            ]
        )
    )
    Path("child.yaml").write_text(
        "\n".join(
            [
                "LOAD:",
                "- base.yaml",
                "A:",
                "  y: ${A.x}",
                "ConstValue: ${answer}",
                "Loaded: !load_yaml dataset.yaml",
                "Items: [null, child]",
                "AppendOnly: [child]",
            ]
        )
    )

    cfg = load_config("child.yaml", CONSTANTS={"answer": "forty-two"})

    assert cfg.A.x == "base"
    assert cfg.A.y == "base"
    assert cfg.ConstValue == "forty-two"
    assert cfg.Loaded.Dataset.name == "local"
    assert cfg.Items == ["child"]
    assert cfg.AppendOnly == ["base", "child"]


def test_nested_inheritance_preserves_an_explicit_empty_list(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("base.yaml").write_text("Items: [base]\n")
    Path("middle.yaml").write_text("LOAD: [base.yaml]\nItems: [null]\n")
    Path("child.yaml").write_text("LOAD: [middle.yaml]\nName: child\n")

    cfg = load_config("child.yaml")

    assert cfg.Items == []
    assert cfg.Name == "child"
