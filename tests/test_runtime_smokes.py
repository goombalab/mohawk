from contextlib import contextmanager
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.slow]

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from utils.config import Config


def _wrapper_config(tmp_path):
    return Config.from_dict(
        {
            "OptimizerConfig": {
                "optimizer": "AdamW",
                "lr": 1e-3,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "optimize_weights": {
                    "black_list": None,
                    "white_list": None,
                },
                "scheduler": {
                    "name": "constant",
                    "warmup_steps": 0,
                    "decay_steps": 0,
                    "warmup_start_lr": 0.0,
                    "min_lr": 0.0,
                },
            },
            "TrainConfig": {
                "n_batches": 2,
                "accumulation_steps": 1,
            },
            "ManagementConfig": {
                "paths": {
                    "tmp_dir": str(tmp_path / "tmp"),
                    "save_dir": str(tmp_path / "save"),
                }
            },
        }
    )


class TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)

    def forward(self, inputs=None, **kwargs):
        if inputs is None:
            inputs = kwargs["inputs"]
        return self.proj(inputs)


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_centralized_wrapper_cpu_forward_backward_step_and_save(tmp_path, mixed_precision):
    from training_wrapper.CentralizedTrainingWrapper import CentralizedTrainingWrapper
    from utils.weights_utils import load_weights

    cfg = _wrapper_config(tmp_path)
    wrapper = CentralizedTrainingWrapper(
        model=TinyModule(),
        config=cfg,
        mode="train",
        model_dtype=torch.float32,
        mixed_precision=mixed_precision,
    )

    assert wrapper.mixed_precision is mixed_precision
    assert hasattr(wrapper, "scaler") is mixed_precision

    weight_before = wrapper.model.proj.weight.detach().cpu().clone()
    scheduler_step_before = wrapper.scheduler.state_dict()["_step_count"]
    loss = wrapper(inputs=torch.ones(1, 2)).float().sum()
    wrapper.backward(loss)
    wrapper.step()
    assert not torch.equal(wrapper.model.proj.weight.detach().cpu(), weight_before)
    assert wrapper.scheduler.state_dict()["_step_count"] == scheduler_step_before + 1
    saved_dir = wrapper.save_weights()

    save_path = Path(saved_dir)
    assert (save_path / "model.safetensors").exists()
    assert (save_path / "optimizer.pth").exists()
    assert (save_path / "scheduler.pth").exists()
    assert (save_path / "config.json").exists()
    optimizer_state = torch.load(save_path / "optimizer.pth", map_location="cpu")
    scheduler_state = torch.load(save_path / "scheduler.pth", map_location="cpu")
    assert optimizer_state["param_groups"]
    assert "_step_count" in scheduler_state

    reloaded = TinyModule()
    reloaded, loaded_keys, missing_keys = load_weights(
        reloaded,
        model_dtype=torch.float32,
        load_cfgs=[
            Config.from_dict(
                {
                    "path": str(save_path),
                    "allow_missing_keys": False,
                    "allow_unexpected_keys": False,
                }
            )
        ],
    )

    assert loaded_keys
    assert missing_keys == []
    for name, value in wrapper.model.state_dict().items():
        assert torch.equal(value.cpu(), reloaded.state_dict()[name])


@pytest.mark.parametrize("init_name", ["eager", "lazy"])
def test_init_functions_load_checkpoint_into_cpu_wrapper(tmp_path, monkeypatch, init_name):
    import utils.init_model as init_model
    from training_wrapper.CentralizedTrainingWrapper import CentralizedTrainingWrapper

    checkpoint_model = TinyModule()
    with torch.no_grad():
        checkpoint_model.proj.weight.fill_(0.25)
        checkpoint_model.proj.bias.fill_(0.5)
    torch.save(checkpoint_model.state_dict(), tmp_path / "pytorch_model.bin")

    build_devices = []

    def build_tiny_model(*args, **kwargs):
        model = TinyModule()
        build_devices.append(model.proj.weight.device.type)
        return model

    monkeypatch.setattr(init_model, "build_model", build_tiny_model)

    cfg = _wrapper_config(tmp_path / init_name)
    details_cfg = Config.from_dict(
        {
            "model_dtype": "float32",
            "compile_model": False,
            "mixed_precision": False,
        }
    )
    load_cfg = Config.from_dict(
        {
            "model": [
                {
                    "path": str(tmp_path),
                    "allow_missing_keys": False,
                    "allow_unexpected_keys": False,
                }
            ]
        }
    )

    wrapper = init_model.get_init_fn(init_name)(
        cfg=cfg,
        details_cfg=details_cfg,
        load_cfg=load_cfg,
        mode="train",
        components_cfg=Config.from_dict({}),
    )

    assert isinstance(wrapper, CentralizedTrainingWrapper)
    assert wrapper.mode == "train"
    assert wrapper.model.training is True
    assert hasattr(wrapper, "optimizer")
    assert hasattr(wrapper, "scheduler")
    assert all(param.requires_grad for param in wrapper.model.parameters())
    assert build_devices == (["meta"] if init_name == "lazy" else ["cpu"])
    assert wrapper.model.proj.weight.device.type == "cpu"
    assert torch.equal(wrapper.model.proj.weight, checkpoint_model.proj.weight)
    assert torch.equal(wrapper.model.proj.bias, checkpoint_model.proj.bias)
    weight_before_step = wrapper.model.proj.weight.detach().cpu().clone()
    scheduler_step_before = wrapper.scheduler.state_dict()["_step_count"]
    loss = wrapper(inputs=torch.ones(1, 2)).float().sum()
    wrapper.backward(loss)
    wrapper.step()
    assert not torch.equal(wrapper.model.proj.weight.detach().cpu(), weight_before_step)
    assert wrapper.scheduler.state_dict()["_step_count"] == scheduler_step_before + 1


def test_move_meta_to_cpu_handles_nonpersistent_parameter_buffers():
    import utils.init_model as init_model

    class ParameterBufferModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "scratch",
                torch.nn.Parameter(torch.ones(2), requires_grad=False),
                persistent=False,
            )

    with init_model.parameter_device(torch.device("meta")):
        module = ParameterBufferModule()

    assert module.scratch.device.type == "meta"
    assert isinstance(module._buffers["scratch"], torch.nn.Parameter)

    init_model.move_meta_to_cpu(module)

    assert module.scratch.device.type == "cpu"
    assert not isinstance(module._buffers["scratch"], torch.nn.Parameter)
    assert "scratch" not in module.state_dict()


def _assert_optimizer_state_matches(actual, expected):
    assert actual["param_groups"] == expected["param_groups"]
    assert actual["state"].keys() == expected["state"].keys()
    for param_key, expected_state in expected["state"].items():
        actual_state = actual["state"][param_key]
        assert actual_state.keys() == expected_state.keys()
        for state_key, expected_value in expected_state.items():
            actual_value = actual_state[state_key]
            if isinstance(expected_value, torch.Tensor):
                assert torch.equal(actual_value.cpu(), expected_value.cpu())
            else:
                assert actual_value == expected_value


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_centralized_wrapper_resume_restores_model_optimizer_and_scheduler(tmp_path, monkeypatch, mixed_precision):
    import utils.init_model as init_model
    from training_wrapper.CentralizedTrainingWrapper import CentralizedTrainingWrapper

    cfg = _wrapper_config(tmp_path / "initial")
    wrapper = CentralizedTrainingWrapper(
        model=TinyModule(),
        config=cfg,
        mode="train",
        model_dtype=torch.float32,
        mixed_precision=mixed_precision,
    )
    loss = wrapper(inputs=torch.ones(1, 2)).float().sum()
    wrapper.backward(loss)
    wrapper.step()
    saved_dir = wrapper.save_weights()
    save_path = Path(saved_dir)

    saved_model_state = {key: value.cpu().clone() for key, value in wrapper.model.state_dict().items()}
    saved_optimizer_state = torch.load(save_path / "optimizer.pth", map_location="cpu")
    saved_scheduler_state = torch.load(save_path / "scheduler.pth", map_location="cpu")

    resume_cfg = _wrapper_config(tmp_path / "resume")
    resume_cfg.LoadConfig = Config.from_dict(
        {
            "model": [
                {
                    "path": str(save_path),
                    "allow_missing_keys": False,
                    "allow_unexpected_keys": False,
                }
            ],
            "optimizer": {"path": str(save_path)},
            "scheduler": {"path": str(save_path)},
        }
    )
    details_cfg = Config.from_dict(
        {
            "model_dtype": "float32",
            "compile_model": False,
            "mixed_precision": mixed_precision,
        }
    )
    monkeypatch.setattr(init_model, "build_model", lambda *args, **kwargs: TinyModule())

    resumed = init_model.eager_init(
        cfg=resume_cfg,
        details_cfg=details_cfg,
        load_cfg=resume_cfg.LoadConfig,
        mode="train",
        components_cfg=Config.from_dict({}),
    )

    assert resumed.mixed_precision is mixed_precision
    assert hasattr(resumed, "scaler") is mixed_precision
    for name, expected_value in saved_model_state.items():
        assert torch.equal(resumed.model.state_dict()[name].cpu(), expected_value)
    _assert_optimizer_state_matches(resumed.optimizer.state_dict(), saved_optimizer_state)
    assert resumed.scheduler.state_dict() == saved_scheduler_state

    step_count_before = resumed.scheduler.state_dict()["_step_count"]
    weight_before_resumed_step = resumed.model.proj.weight.detach().cpu().clone()
    loss = resumed(inputs=torch.ones(1, 2)).float().sum()
    resumed.backward(loss)
    resumed.step()
    assert not torch.equal(resumed.model.proj.weight.detach().cpu(), weight_before_resumed_step)
    assert resumed.scheduler.state_dict()["_step_count"] == step_count_before + 1


def test_fsdp_activation_checkpointing_switch_invokes_checkpoint_api(monkeypatch, tmp_path):
    import importlib

    fsdp_wrapper = importlib.import_module("training_wrapper.FSDPTrainingWrapper")

    class TinyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)

        def forward(self, inputs):
            return self.proj(inputs)

    class TinyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = SimpleNamespace(layers=torch.nn.ModuleList([TinyBlock()]))

        def forward(self, inputs):
            return self.backbone.layers[0](inputs)

    class FakeFSDP(torch.nn.Module):
        def __init__(self, module, **kwargs):
            super().__init__()
            self.module = module
            self.kwargs = kwargs

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    monkeypatch.setattr(fsdp_wrapper.torch_dist, "is_initialized", lambda: True)
    monkeypatch.setattr(fsdp_wrapper.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(fsdp_wrapper, "FSDP", FakeFSDP)

    calls = []

    def fake_checkpoint_wrapper(module, **kwargs):
        return module, kwargs

    def fake_apply_activation_checkpointing(module, checkpoint_wrapper_fn, check_fn):
        block = module.module.backbone.layers[0]
        wrapped_block, wrapper_kwargs = checkpoint_wrapper_fn(block)
        calls.append(
            {
                "module": module,
                "check_block": check_fn(block),
                "check_root": check_fn(module.module),
                "wrapped_block": wrapped_block,
                "wrapper_kwargs": wrapper_kwargs,
            }
        )

    monkeypatch.setattr(fsdp_wrapper, "checkpoint_wrapper", fake_checkpoint_wrapper)
    monkeypatch.setattr(fsdp_wrapper, "apply_activation_checkpointing", fake_apply_activation_checkpointing)

    call_deltas = {}
    for enabled in [True, False]:
        wrapper = object.__new__(fsdp_wrapper.FSDPTrainingWrapper)
        wrapper.config = _wrapper_config(tmp_path / str(enabled))
        wrapper.config.TrainConfig.activation_checkpointing = enabled
        wrapper._model = TinyTransformer()
        wrapper.mixed_precision = False
        wrapper.model_dtype = torch.float32
        wrapper.weights_from_rank0 = False
        wrapper.cpu_offload = False
        wrapper.use_orig_params = False

        wrapped = wrapper._setup_fsdp()

        assert isinstance(wrapped, FakeFSDP)
        assert wrapped.kwargs["device_id"] == 0
        assert "auto_wrap_policy" in wrapped.kwargs
        call_deltas[enabled] = len(calls)

    assert len(calls) == 1
    assert call_deltas == {True: 1, False: 1}
    assert calls[0]["check_block"] is True
    assert calls[0]["check_root"] is False
    assert calls[0]["wrapped_block"].__class__ is TinyBlock
    assert calls[0]["wrapper_kwargs"] == {
        "offload_to_cpu": False,
        "checkpoint_impl": fsdp_wrapper.CheckpointImpl.NO_REENTRANT,
    }


def test_fsdp_activation_checkpointing_skips_checkpoint_api_without_block_layout(monkeypatch, tmp_path):
    import importlib

    fsdp_wrapper = importlib.import_module("training_wrapper.FSDPTrainingWrapper")

    class TinyUnstructured(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)

        def forward(self, inputs):
            return self.proj(inputs)

    class FakeFSDP(torch.nn.Module):
        def __init__(self, module, **kwargs):
            super().__init__()
            self.module = module
            self.kwargs = kwargs

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    def fail_if_checkpointing_is_applied(*args, **kwargs):
        raise AssertionError("checkpoint API should not run without an inferred block class")

    warnings = []
    monkeypatch.setattr(fsdp_wrapper.torch_dist, "is_initialized", lambda: True)
    monkeypatch.setattr(fsdp_wrapper.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(fsdp_wrapper, "FSDP", FakeFSDP)
    monkeypatch.setattr(fsdp_wrapper, "apply_activation_checkpointing", fail_if_checkpointing_is_applied)
    monkeypatch.setattr(fsdp_wrapper.logger, "warning", lambda message: warnings.append(message))

    wrapper = object.__new__(fsdp_wrapper.FSDPTrainingWrapper)
    wrapper.config = _wrapper_config(tmp_path)
    wrapper.config.TrainConfig.activation_checkpointing = True
    wrapper._model = TinyUnstructured()
    wrapper.mixed_precision = False
    wrapper.model_dtype = torch.float32
    wrapper.weights_from_rank0 = False
    wrapper.cpu_offload = False
    wrapper.use_orig_params = False

    wrapped = wrapper._setup_fsdp()

    assert isinstance(wrapped, FakeFSDP)
    assert "auto_wrap_policy" not in wrapped.kwargs
    assert wrapped.kwargs["use_orig_params"] is False
    assert any("Could not infer transformer block class" in message for message in warnings)
    assert any("activation_checkpointing requested" in message for message in warnings)


def test_perplexity_evaluator_runs_on_cpu_with_fake_dataloader(monkeypatch):
    from evals import eval_ppl

    class FakeDataloader:
        tokenizer = SimpleNamespace(eos_token_id=4, pad_token_id=0)

        def __iter__(self):
            yield {"input_ids": torch.tensor([[1, 2, 3, 4]])}

    class PredictNextToken(torch.nn.Module):
        def forward(self, input_ids, position_ids=None):
            logits = torch.full(
                (input_ids.shape[0], input_ids.shape[1], 5),
                -10.0,
                device=input_ids.device,
            )
            for idx in range(input_ids.shape[1] - 1):
                logits[:, idx, input_ids[:, idx + 1]] = 10.0
            return SimpleNamespace(logits=logits)

    class Wrapper:
        def __init__(self):
            self.model = PredictNextToken()

        def __call__(self, *args, **kwargs):
            return self.model(*args, **kwargs)

    monkeypatch.setattr(
        eval_ppl,
        "setup_dataloader",
        lambda data_cfg, split: FakeDataloader(),
    )

    for initially_training in [True, False]:
        wrapper = Wrapper()
        wrapper.model.train(initially_training)
        evaluator = eval_ppl.evaluator(
            cfg=Config.from_dict({}),
            DataConfig=Config.from_dict({}),
            n_batches=1,
        )
        result = evaluator(wrapper)

        assert result["perplexity"] < 1.01
        assert result["accuracy"] == 1.0
        assert wrapper.model.training is initially_training


@pytest.mark.parametrize("initially_training", [True, False])
def test_perplexity_evaluator_runs_with_cached_tiny_hf_model(monkeypatch, initially_training):
    transformers = pytest.importorskip("transformers")
    from evals import eval_ppl

    model_id = "sshleifer/tiny-gpt2"
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
    except OSError as exc:
        pytest.skip(f"{model_id} is not available in the local Hugging Face cache: {exc}")

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    class CachedTinyHfDataloader:
        def __init__(self):
            self.tokenizer = tokenizer
            self.batch = tokenizer(
                "hello world",
                return_tensors="pt",
                padding=False,
                add_special_tokens=True,
            )

        def __iter__(self):
            yield {"input_ids": self.batch.input_ids}

    class Wrapper:
        def __init__(self):
            self.model = model

        def __call__(self, *args, **kwargs):
            return self.model(*args, **kwargs)

    monkeypatch.setattr(
        eval_ppl,
        "setup_dataloader",
        lambda data_cfg, split: CachedTinyHfDataloader(),
    )

    wrapper = Wrapper()
    wrapper.model.train(initially_training)
    evaluator = eval_ppl.evaluator(
        cfg=Config.from_dict({}),
        DataConfig=Config.from_dict({}),
        n_batches=1,
    )
    result = evaluator(wrapper)

    assert result["perplexity"] > 0
    assert 0.0 <= result["accuracy"] <= 1.0
    assert wrapper.model.training is initially_training


def test_set_config_accepts_comma_separated_configs_and_saves_each(tmp_path):
    from distill.distill import set_config

    config_paths = []
    for name in ["first", "second"]:
        save_dir = tmp_path / f"save-{name}"
        config_path = tmp_path / f"{name}.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "ManagementConfig:",
                    "  paths:",
                    f"    save_dir: {save_dir}",
                    "DistillConfig:",
                    f"  name: {name}",
                ]
            )
        )
        config_paths.append(config_path)

    configs = set_config(f" {config_paths[0]} , , {config_paths[1]} , ")

    assert [cfg.DistillConfig.name for cfg in configs] == ["first", "second"]
    assert (tmp_path / "save-first" / "config.yaml").exists()
    assert (tmp_path / "save-second" / "config.yaml").exists()

    with pytest.raises(ValueError, match="At least one config path"):
        set_config(" , , ")


def test_distill_entrypoint_executes_comma_separated_configs_in_order(tmp_path, monkeypatch):
    import sys
    import utils.distributed as distributed
    from distill import distill as distill_module

    @contextmanager
    def no_distributed(*args, **kwargs):
        yield

    @contextmanager
    def no_logging(config, wandb_id=None):
        yield

    executed = []
    barrier_env_snapshots = []
    config_paths = []
    for name, env_value in [("first", "one"), ("second", "two")]:
        save_dir = tmp_path / f"save-{name}"
        config_path = tmp_path / f"{name}.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "ManagementConfig:",
                    "  paths:",
                    f"    save_dir: {save_dir}",
                    "  env_vars:",
                    f"    MOHAWK_SMOKE_ENV: {env_value}",
                    f"    MOHAWK_EXISTING_ENV: override-{env_value}",
                    "  wandb:",
                    "    project: smoke",
                    "    entity: smoke",
                    "DistillConfig:",
                    f"  name: {name}",
                ]
            )
        )
        config_paths.append(config_path)

    def fake_distillation_code(config):
        executed.append(
            (
                config.DistillConfig.name,
                os.environ.get("MOHAWK_SMOKE_ENV"),
                os.environ.get("MOHAWK_EXISTING_ENV"),
            )
        )
        assert os.environ["MOHAWK_EXISTING_ENV"] == f"override-{os.environ['MOHAWK_SMOKE_ENV']}"

    def snapshot_barrier():
        barrier_env_snapshots.append(
            (
                os.environ.get("MOHAWK_SMOKE_ENV"),
                os.environ.get("MOHAWK_EXISTING_ENV"),
            )
        )

    monkeypatch.setattr(distributed, "init_distributed", no_distributed)
    monkeypatch.setattr(distill_module, "barrier", snapshot_barrier)
    monkeypatch.setattr(distill_module, "init_logging", no_logging)
    monkeypatch.setattr(distill_module, "distillation_code", fake_distillation_code)
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", ",".join(str(path) for path in config_paths)])
    monkeypatch.delenv("MOHAWK_SMOKE_ENV", raising=False)
    monkeypatch.setenv("MOHAWK_EXISTING_ENV", "original")

    distill_module.distill()

    assert executed == [
        ("first", "one", "override-one"),
        ("second", "two", "override-two"),
    ]
    assert barrier_env_snapshots == [
        (None, "original"),
        (None, "original"),
        (None, "original"),
    ]
    assert "MOHAWK_SMOKE_ENV" not in os.environ
    assert os.environ["MOHAWK_EXISTING_ENV"] == "original"


def test_distill_entrypoint_restores_env_when_config_execution_fails(tmp_path, monkeypatch):
    import sys
    import utils.distributed as distributed
    from distill import distill as distill_module

    @contextmanager
    def no_distributed(*args, **kwargs):
        yield

    @contextmanager
    def no_logging(config, wandb_id=None):
        yield

    save_dir = tmp_path / "save"
    config_path = tmp_path / "failing.yaml"
    config_path.write_text(
        "\n".join(
            [
                "ManagementConfig:",
                "  paths:",
                f"    save_dir: {save_dir}",
                "  env_vars:",
                "    MOHAWK_SMOKE_ENV: failing",
                "    MOHAWK_EXISTING_ENV: override-failing",
                "  wandb:",
                "    project: smoke",
                "    entity: smoke",
                "DistillConfig:",
                "  name: failing",
            ]
        )
    )

    def fail_distillation_code(config):
        assert config.DistillConfig.name == "failing"
        assert os.environ["MOHAWK_SMOKE_ENV"] == "failing"
        assert os.environ["MOHAWK_EXISTING_ENV"] == "override-failing"
        raise RuntimeError("config failed")

    monkeypatch.setattr(distributed, "init_distributed", no_distributed)
    monkeypatch.setattr(distill_module, "barrier", lambda: None)
    monkeypatch.setattr(distill_module, "init_logging", no_logging)
    monkeypatch.setattr(distill_module, "distillation_code", fail_distillation_code)
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(config_path)])
    monkeypatch.delenv("MOHAWK_SMOKE_ENV", raising=False)
    monkeypatch.setenv("MOHAWK_EXISTING_ENV", "original")

    with pytest.raises(RuntimeError, match="config failed"):
        distill_module.distill()

    assert "MOHAWK_SMOKE_ENV" not in os.environ
    assert os.environ["MOHAWK_EXISTING_ENV"] == "original"


def test_distill_entrypoint_does_not_clear_unrelated_runtime_env(tmp_path, monkeypatch):
    import sys
    import utils.distributed as distributed
    from distill import distill as distill_module

    @contextmanager
    def no_distributed(*args, **kwargs):
        yield

    @contextmanager
    def no_logging(config, wandb_id=None):
        yield

    save_dir = tmp_path / "save"
    config_path = tmp_path / "runtime_env.yaml"
    config_path.write_text(
        "\n".join(
            [
                "ManagementConfig:",
                "  paths:",
                f"    save_dir: {save_dir}",
                "  env_vars:",
                "    MOHAWK_SMOKE_ENV: configured",
                "  wandb:",
                "    project: smoke",
                "    entity: smoke",
                "DistillConfig:",
                "  name: runtime-env",
            ]
        )
    )

    def fake_distillation_code(config):
        assert os.environ["MOHAWK_SMOKE_ENV"] == "configured"
        os.environ["MOHAWK_RUNTIME_ADDED_ENV"] = "preserved"

    monkeypatch.setattr(distributed, "init_distributed", no_distributed)
    monkeypatch.setattr(distill_module, "barrier", lambda: None)
    monkeypatch.setattr(distill_module, "init_logging", no_logging)
    monkeypatch.setattr(distill_module, "distillation_code", fake_distillation_code)
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(config_path)])
    monkeypatch.delenv("MOHAWK_SMOKE_ENV", raising=False)
    monkeypatch.delenv("MOHAWK_RUNTIME_ADDED_ENV", raising=False)

    distill_module.distill()

    assert "MOHAWK_SMOKE_ENV" not in os.environ
    assert os.environ["MOHAWK_RUNTIME_ADDED_ENV"] == "preserved"


def test_distill_entrypoint_empty_env_override_preserves_existing_token(tmp_path, monkeypatch):
    import sys
    import utils.distributed as distributed
    from distill import distill as distill_module

    @contextmanager
    def no_distributed(*args, **kwargs):
        yield

    @contextmanager
    def no_logging(config, wandb_id=None):
        yield

    save_dir = tmp_path / "save"
    config_path = tmp_path / "empty_token.yaml"
    config_path.write_text(
        "\n".join(
            [
                "ManagementConfig:",
                "  paths:",
                f"    save_dir: {save_dir}",
                "  env_vars:",
                '    HF_TOKEN: ""',
                "  wandb:",
                "    project: smoke",
                "    entity: smoke",
                "DistillConfig:",
                "  name: empty-token",
            ]
        )
    )

    def fake_distillation_code(config):
        assert config.DistillConfig.name == "empty-token"
        assert os.environ["HF_TOKEN"] == "real-token"

    monkeypatch.setattr(distributed, "init_distributed", no_distributed)
    monkeypatch.setattr(distill_module, "barrier", lambda: None)
    monkeypatch.setattr(distill_module, "init_logging", no_logging)
    monkeypatch.setattr(distill_module, "distillation_code", fake_distillation_code)
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(config_path)])
    monkeypatch.setenv("HF_TOKEN", "real-token")

    distill_module.distill()

    assert os.environ["HF_TOKEN"] == "real-token"


def test_distill_entrypoint_restores_env_when_logging_setup_fails(tmp_path, monkeypatch):
    import sys
    import utils.distributed as distributed
    from distill import distill as distill_module

    @contextmanager
    def no_distributed(*args, **kwargs):
        yield

    @contextmanager
    def fail_logging(config, wandb_id=None):
        assert config.DistillConfig.name == "logging-fails"
        assert os.environ["MOHAWK_SMOKE_ENV"] == "logging"
        assert os.environ["MOHAWK_EXISTING_ENV"] == "override-logging"
        raise RuntimeError("logging failed")
        yield

    save_dir = tmp_path / "save"
    config_path = tmp_path / "logging_fails.yaml"
    config_path.write_text(
        "\n".join(
            [
                "ManagementConfig:",
                "  paths:",
                f"    save_dir: {save_dir}",
                "  env_vars:",
                "    MOHAWK_SMOKE_ENV: logging",
                "    MOHAWK_EXISTING_ENV: override-logging",
                "  wandb:",
                "    project: smoke",
                "    entity: smoke",
                "DistillConfig:",
                "  name: logging-fails",
            ]
        )
    )

    def fail_if_distillation_runs(config):
        raise AssertionError("distillation_code should not run when logging setup fails")

    monkeypatch.setattr(distributed, "init_distributed", no_distributed)
    monkeypatch.setattr(distill_module, "barrier", lambda: None)
    monkeypatch.setattr(distill_module, "init_logging", fail_logging)
    monkeypatch.setattr(distill_module, "distillation_code", fail_if_distillation_runs)
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(config_path)])
    monkeypatch.delenv("MOHAWK_SMOKE_ENV", raising=False)
    monkeypatch.setenv("MOHAWK_EXISTING_ENV", "original")

    with pytest.raises(RuntimeError, match="logging failed"):
        distill_module.distill()

    assert "MOHAWK_SMOKE_ENV" not in os.environ
    assert os.environ["MOHAWK_EXISTING_ENV"] == "original"
