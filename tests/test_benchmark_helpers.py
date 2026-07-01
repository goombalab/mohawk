import inspect
import importlib.metadata
import os
import pytest
import sys
from contextlib import contextmanager
from types import SimpleNamespace

pytestmark = [pytest.mark.slow]

pytest.importorskip("pandas")
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("lm_eval")

from evals import benchmark


def test_installed_lm_eval_version_matches_public_benchmark_claim():
    assert importlib.metadata.version("lm_eval") == "0.4.11"


def test_resolve_hf_model_source_uses_local_inputs_without_snapshot(tmp_path):
    local_model = tmp_path / "model"
    local_model.mkdir()

    assert benchmark._resolve_hf_model_source(str(local_model), local_files_only=True) == str(local_model)
    assert benchmark._resolve_hf_model_source("sshleifer/tiny-gpt2", local_files_only=False) == "sshleifer/tiny-gpt2"


def test_resolve_hf_model_source_uses_cached_snapshot_for_local_files_only(monkeypatch):
    calls = {}

    def fake_snapshot_download(model_id, **kwargs):
        calls["snapshot"] = (model_id, kwargs)
        return "/cached/sshleifer/tiny-gpt2"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    assert benchmark._resolve_hf_model_source("sshleifer/tiny-gpt2", local_files_only=True) == "/cached/sshleifer/tiny-gpt2"
    assert calls["snapshot"] == ("sshleifer/tiny-gpt2", {"local_files_only": True})


def test_resolve_hf_model_source_errors_when_local_snapshot_is_missing(monkeypatch):
    def fake_snapshot_download(model_id, **kwargs):
        raise OSError("not cached")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    with pytest.raises(benchmark.LocalHfCacheError, match="Populate the cache first"):
        benchmark._resolve_hf_model_source("sshleifer/tiny-gpt2", local_files_only=True)


def test_benchmark_parser_exposes_public_smoke_and_backend_flags():
    parser = benchmark.build_arg_parser()
    args = parser.parse_args(
        [
            "--dir",
            "sshleifer/tiny-gpt2",
            "--tasks",
            "wikitext, arc_easy",
            "--batch_size",
            "1",
            "--limit",
            "1",
            "--local_files_only",
            "--backend",
            "hf",
            "--device",
            "cpu",
            "--model-registration-module",
            "fla.models.raven",
        ]
    )

    assert args.dir == "sshleifer/tiny-gpt2"
    assert benchmark._tasks_from_arg(args.tasks) == ["wikitext", "arc_easy"]
    assert args.batch_size == 1
    assert args.limit == 1
    assert args.local_files_only is True
    assert args.backend == "hf"
    assert args.device == "cpu"
    assert args.model_registration_module == ["fla.models.raven"]
    assert "lm-eval task data" in parser.format_help()


def test_public_tiny_hf_command_parses_and_auto_routes_to_hf(monkeypatch):
    calls = []

    def fake_can_load_hf_config(model_dir, local_files_only):
        calls.append((model_dir, local_files_only))
        return True

    monkeypatch.setattr(benchmark, "_can_load_hf_config", fake_can_load_hf_config)

    args = benchmark.build_arg_parser().parse_args(
        " ".join(
            [
                "--dir",
                "sshleifer/tiny-gpt2",
                "--tasks",
                "wikitext",
                "--batch_size",
                "1",
                "--limit",
                "1",
                "--local_files_only",
            ]
        ).split()
    )

    assert args.backend == "auto"
    assert benchmark._tasks_from_arg(args.tasks) == ["wikitext"]
    assert benchmark._select_backend_for_cli(args) == "hf"
    assert calls == [("sshleifer/tiny-gpt2", True)]


def test_benchmark_parser_requires_model_dir_and_tasks():
    parser = benchmark.build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--dir", "sshleifer/tiny-gpt2"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--tasks", "wikitext"])


def test_tasks_from_arg_trims_whitespace_and_rejects_empty_specs():
    assert benchmark._tasks_from_arg(" wikitext , arc_easy ") == ["wikitext", "arc_easy"]
    assert benchmark._tasks_from_arg("wikitext,, arc_easy,") == ["wikitext", "arc_easy"]

    with pytest.raises(ValueError, match="At least one lm-eval task"):
        benchmark._tasks_from_arg(" , , ")


def test_run_hf_benchmark_rejects_empty_tasks_before_model_resolution(monkeypatch):
    def fail_if_model_resolution_is_reached(*args, **kwargs):
        raise AssertionError("model source resolution should not run for an empty task list")

    monkeypatch.setattr(benchmark, "_resolve_hf_model_source", fail_if_model_resolution_is_reached)

    with pytest.raises(ValueError, match="At least one lm-eval task"):
        benchmark.run_hf_benchmark(
            SimpleNamespace(
                dir="sshleifer/tiny-gpt2",
                tasks=" , , ",
                batch_size=1,
                limit=1,
                local_files_only=True,
                backend="hf",
                device="cpu",
                num_fewshot=0,
            )
        )


def test_cli_backend_selection_rejects_empty_tasks_before_backend_probe(monkeypatch):
    def fail_if_backend_selection_is_reached(*args, **kwargs):
        raise AssertionError("backend selection should not run for an empty task list")

    monkeypatch.setattr(benchmark, "_select_backend", fail_if_backend_selection_is_reached)

    with pytest.raises(ValueError, match="At least one lm-eval task"):
        benchmark._select_backend_for_cli(
            SimpleNamespace(
                backend="auto",
                dir="sshleifer/tiny-gpt2",
                tasks=" , , ",
                local_files_only=True,
            )
        )


def test_cli_backend_exit_helper_returns_selected_backend(monkeypatch):
    monkeypatch.setattr(benchmark, "_select_backend_for_cli", lambda args: "hf")

    assert benchmark._select_backend_or_exit(SimpleNamespace()) == "hf"


def test_cli_backend_exit_helper_converts_empty_task_error_to_system_exit(monkeypatch):
    error = ValueError("At least one lm-eval task must be provided.")

    def reject_empty_tasks(args):
        raise error

    monkeypatch.setattr(benchmark, "_select_backend_for_cli", reject_empty_tasks)

    with pytest.raises(SystemExit) as raised:
        benchmark._select_backend_or_exit(SimpleNamespace())

    assert str(raised.value) == "At least one lm-eval task must be provided."
    assert raised.value.__cause__ is error


def test_cli_backend_exit_helper_converts_local_cache_error_to_system_exit(monkeypatch):
    error = benchmark.LocalHfCacheError("missing local cache")

    def reject_missing_cache(args):
        raise error

    monkeypatch.setattr(benchmark, "_select_backend_for_cli", reject_missing_cache)

    with pytest.raises(SystemExit) as raised:
        benchmark._select_backend_or_exit(SimpleNamespace())

    assert str(raised.value) == "missing local cache"
    assert raised.value.__cause__ is error


def test_installed_lm_eval_hflm_accepts_initialized_model_and_tokenizer_contract():
    from lm_eval.models.huggingface import HFLM as LMEvalHFLM

    signature = inspect.signature(LMEvalHFLM.__init__)
    for parameter in ["pretrained", "tokenizer", "backend", "batch_size", "device", "dtype"]:
        assert parameter in signature.parameters


def test_local_hf_cache_mode_sets_and_restores_offline_flags(monkeypatch):
    fake_datasets = SimpleNamespace(config=SimpleNamespace(HF_DATASETS_OFFLINE=False))
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setenv("HF_HUB_OFFLINE", "previous")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)

    with benchmark._local_hf_cache_mode(True):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ["HF_DATASETS_OFFLINE"] == "1"
        assert fake_datasets.config.HF_DATASETS_OFFLINE is True

    assert os.environ["HF_HUB_OFFLINE"] == "previous"
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert "HF_DATASETS_OFFLINE" not in os.environ
    assert fake_datasets.config.HF_DATASETS_OFFLINE is False


def test_local_hf_cache_mode_noops_when_disabled_and_restores_existing_values(monkeypatch):
    fake_datasets = SimpleNamespace(config=SimpleNamespace(HF_DATASETS_OFFLINE=True))
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    for name, value in {
        "HF_HUB_OFFLINE": "hub-previous",
        "TRANSFORMERS_OFFLINE": "transformers-previous",
        "HF_DATASETS_OFFLINE": "datasets-previous",
    }.items():
        monkeypatch.setenv(name, value)

    with benchmark._local_hf_cache_mode(False):
        assert os.environ["HF_HUB_OFFLINE"] == "hub-previous"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "transformers-previous"
        assert os.environ["HF_DATASETS_OFFLINE"] == "datasets-previous"
        assert fake_datasets.config.HF_DATASETS_OFFLINE is True

    with benchmark._local_hf_cache_mode(True):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ["HF_DATASETS_OFFLINE"] == "1"
        assert fake_datasets.config.HF_DATASETS_OFFLINE is True

    assert os.environ["HF_HUB_OFFLINE"] == "hub-previous"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "transformers-previous"
    assert os.environ["HF_DATASETS_OFFLINE"] == "datasets-previous"
    assert fake_datasets.config.HF_DATASETS_OFFLINE is True


def test_local_hf_cache_mode_disabled_preserves_import_time_flags_and_sessions(monkeypatch):
    import datasets
    import huggingface_hub.constants as hub_constants
    import huggingface_hub.utils._http as hub_http
    import transformers.utils.hub as transformers_hub

    resets = []
    monkeypatch.setattr(datasets.config, "HF_DATASETS_OFFLINE", False)
    monkeypatch.setattr(datasets.config, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(hub_http, "reset_sessions", lambda: resets.append("reset"))
    monkeypatch.setattr(transformers_hub, "_is_offline_mode", False)

    with benchmark._local_hf_cache_mode(False):
        assert datasets.config.HF_DATASETS_OFFLINE is False
        assert datasets.config.HF_HUB_OFFLINE is False
        assert hub_constants.HF_HUB_OFFLINE is False
        assert transformers_hub._is_offline_mode is False

    assert datasets.config.HF_DATASETS_OFFLINE is False
    assert datasets.config.HF_HUB_OFFLINE is False
    assert hub_constants.HF_HUB_OFFLINE is False
    assert transformers_hub._is_offline_mode is False
    assert resets == []


def test_local_hf_cache_mode_restores_existing_values_after_exception(monkeypatch):
    fake_datasets = SimpleNamespace(config=SimpleNamespace(HF_DATASETS_OFFLINE=False))
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setenv("HF_HUB_OFFLINE", "hub-previous")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "datasets-previous")

    with pytest.raises(RuntimeError, match="lm-eval failed"):
        with benchmark._local_hf_cache_mode(True):
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
            assert os.environ["HF_DATASETS_OFFLINE"] == "1"
            assert fake_datasets.config.HF_DATASETS_OFFLINE is True
            raise RuntimeError("lm-eval failed")

    assert os.environ["HF_HUB_OFFLINE"] == "hub-previous"
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert os.environ["HF_DATASETS_OFFLINE"] == "datasets-previous"
    assert fake_datasets.config.HF_DATASETS_OFFLINE is False


def test_local_hf_cache_mode_sets_import_time_offline_flags_and_resets_sessions(monkeypatch):
    import datasets
    import huggingface_hub.constants as hub_constants
    import huggingface_hub.utils._http as hub_http
    import transformers.utils.hub as transformers_hub

    resets = []
    monkeypatch.setattr(datasets.config, "HF_DATASETS_OFFLINE", False)
    monkeypatch.setattr(datasets.config, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(hub_http, "reset_sessions", lambda: resets.append("reset"))
    monkeypatch.setattr(transformers_hub, "_is_offline_mode", False)

    with benchmark._local_hf_cache_mode(True):
        assert datasets.config.HF_DATASETS_OFFLINE is True
        assert datasets.config.HF_HUB_OFFLINE is True
        assert hub_constants.HF_HUB_OFFLINE is True
        assert transformers_hub._is_offline_mode is True
        assert resets == ["reset"]

    assert datasets.config.HF_DATASETS_OFFLINE is False
    assert datasets.config.HF_HUB_OFFLINE is False
    assert hub_constants.HF_HUB_OFFLINE is False
    assert transformers_hub._is_offline_mode is False
    assert resets == ["reset", "reset"]


def test_local_hf_cache_mode_restores_import_time_flags_after_exception(monkeypatch):
    import datasets
    import huggingface_hub.constants as hub_constants
    import huggingface_hub.utils._http as hub_http
    import transformers.utils.hub as transformers_hub

    resets = []
    monkeypatch.setattr(datasets.config, "HF_DATASETS_OFFLINE", False)
    monkeypatch.setattr(datasets.config, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(hub_http, "reset_sessions", lambda: resets.append("reset"))
    monkeypatch.setattr(transformers_hub, "_is_offline_mode", False)

    with pytest.raises(RuntimeError, match="lm-eval failed"):
        with benchmark._local_hf_cache_mode(True):
            assert datasets.config.HF_DATASETS_OFFLINE is True
            assert datasets.config.HF_HUB_OFFLINE is True
            assert hub_constants.HF_HUB_OFFLINE is True
            assert transformers_hub._is_offline_mode is True
            raise RuntimeError("lm-eval failed")

    assert datasets.config.HF_DATASETS_OFFLINE is False
    assert datasets.config.HF_HUB_OFFLINE is False
    assert hub_constants.HF_HUB_OFFLINE is False
    assert transformers_hub._is_offline_mode is False
    assert resets == ["reset", "reset"]


def test_aggregate_results_uses_lm_eval_priority_metrics_and_average():
    results = benchmark.aggregate_results(
        {
            "results": {
                "wikitext": {
                    "alias": "wikitext",
                    "word_perplexity,none": 10.0,
                    "bits_per_byte,none": 1.5,
                },
                "arc_easy": {
                    "acc,none": 0.25,
                    "acc_norm,none": 0.5,
                },
                "ignored_task": {
                    "acc,none": 1.0,
                },
            }
        },
        tasks=["wikitext", "arc_easy"],
    )

    assert results.loc["wikitext", "Result"] == 10.0
    assert results.loc["arc_easy", "Result"] == 0.25
    assert results.loc["AVG", "Result"] == pytest.approx(5.125)
    assert "ignored_task" not in results.index


def test_run_hf_benchmark_passes_initialized_transformers_model_to_lm_eval(monkeypatch):
    calls = {}

    class FakeModel:
        def __init__(self):
            self.eval_called = False
            self.to_device = None

        def eval(self):
            self.eval_called = True
            return self

        def to(self, device):
            self.to_device = device
            return self

    fake_model = FakeModel()

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls["model_load"] = (model_name, kwargs)
            return fake_model

    class FakeTokenizer:
        def __init__(self):
            self.pad_token = None
            self.eos_token = "<eos>"

    fake_tokenizer = FakeTokenizer()
    fake_datasets = SimpleNamespace(config=SimpleNamespace(HF_DATASETS_OFFLINE=False))

    class FakeTokenizerLoader:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls["tokenizer_load"] = (model_name, kwargs)
            return fake_tokenizer

    class FakeLMEvalHFLM:
        def __init__(self, **kwargs):
            calls["hflm_kwargs"] = kwargs

    def fake_simple_evaluate(**kwargs):
        calls["eval_kwargs"] = kwargs
        calls["offline_env"] = {
            name: os.environ.get(name)
            for name in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]
        }
        calls["offline_datasets_config"] = fake_datasets.config.HF_DATASETS_OFFLINE
        return {
            "results": {
                "wikitext": {
                    "word_perplexity,none": 3.0,
                }
            }
        }

    import lm_eval.models.huggingface as lm_eval_huggingface

    monkeypatch.setattr(benchmark, "AutoModelForCausalLM", FakeModelLoader)
    monkeypatch.setattr(benchmark, "AutoTokenizer", FakeTokenizerLoader)
    monkeypatch.setattr(benchmark, "simple_evaluate", fake_simple_evaluate)
    monkeypatch.setattr(lm_eval_huggingface, "HFLM", FakeLMEvalHFLM)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setattr(
        benchmark,
        "_resolve_hf_model_source",
        lambda model_dir, local_files_only: "/cached/sshleifer/tiny-gpt2",
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "outer-hub")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "outer-datasets")

    results = benchmark.run_hf_benchmark(
        SimpleNamespace(
            dir="sshleifer/tiny-gpt2",
            tasks="wikitext",
            batch_size=2,
            limit=1,
            local_files_only=True,
            backend="hf",
            device="cpu",
            num_fewshot=0,
        )
    )

    assert calls["model_load"] == (
        "/cached/sshleifer/tiny-gpt2",
        {"local_files_only": True, "torch_dtype": benchmark.torch.float32},
    )
    assert fake_model.eval_called is True
    assert fake_model.to_device is None
    assert calls["tokenizer_load"] == (
        "/cached/sshleifer/tiny-gpt2",
        {"local_files_only": True},
    )
    assert fake_tokenizer.pad_token == "<eos>"
    assert calls["hflm_kwargs"]["pretrained"] is fake_model
    assert calls["hflm_kwargs"]["tokenizer"] is fake_tokenizer
    assert calls["hflm_kwargs"]["backend"] == "causal"
    assert calls["hflm_kwargs"]["batch_size"] == 2
    assert calls["hflm_kwargs"]["device"] == "cpu"
    assert calls["hflm_kwargs"]["dtype"] is None
    assert calls["eval_kwargs"]["model"].__class__ is FakeLMEvalHFLM
    assert calls["eval_kwargs"]["model_args"] is None
    assert calls["eval_kwargs"]["batch_size"] == 2
    assert calls["eval_kwargs"]["device"] == "cpu"
    assert calls["eval_kwargs"]["limit"] == 1
    assert calls["offline_env"] == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    assert calls["offline_datasets_config"] is True
    assert os.environ["HF_HUB_OFFLINE"] == "outer-hub"
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert os.environ["HF_DATASETS_OFFLINE"] == "outer-datasets"
    assert fake_datasets.config.HF_DATASETS_OFFLINE is False
    assert results.loc["wikitext", "Result"] == 3.0


def test_run_hf_benchmark_preserves_online_state_without_local_files_only(monkeypatch):
    calls = {}
    fake_datasets = SimpleNamespace(config=SimpleNamespace(HF_DATASETS_OFFLINE=False))

    class FakeModel:
        def eval(self):
            calls["model_eval"] = True
            return self

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls["model_load"] = (model_name, kwargs)
            return FakeModel()

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

    class FakeTokenizerLoader:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls["tokenizer_load"] = (model_name, kwargs)
            return FakeTokenizer()

    class FakeLMEvalHFLM:
        def __init__(self, **kwargs):
            calls["hflm_kwargs"] = kwargs

    def fake_resolve_model_source(model_dir, local_files_only):
        calls["resolve"] = (model_dir, local_files_only)
        return model_dir

    def fake_simple_evaluate(**kwargs):
        calls["eval_kwargs"] = kwargs
        calls["eval_env"] = {
            name: os.environ.get(name)
            for name in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]
        }
        calls["eval_datasets_config"] = fake_datasets.config.HF_DATASETS_OFFLINE
        return {
            "results": {
                "wikitext": {
                    "word_perplexity,none": 5.0,
                }
            }
        }

    import lm_eval.models.huggingface as lm_eval_huggingface

    monkeypatch.setattr(benchmark, "AutoModelForCausalLM", FakeModelLoader)
    monkeypatch.setattr(benchmark, "AutoTokenizer", FakeTokenizerLoader)
    monkeypatch.setattr(benchmark, "simple_evaluate", fake_simple_evaluate)
    monkeypatch.setattr(lm_eval_huggingface, "HFLM", FakeLMEvalHFLM)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setattr(benchmark, "_resolve_hf_model_source", fake_resolve_model_source)
    monkeypatch.setenv("HF_HUB_OFFLINE", "outer-hub")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "outer-datasets")

    results = benchmark.run_hf_benchmark(
        SimpleNamespace(
            dir="sshleifer/tiny-gpt2",
            tasks="wikitext",
            batch_size=1,
            limit=1,
            local_files_only=False,
            backend="hf",
            device="cpu",
            num_fewshot=0,
        )
    )

    assert calls["resolve"] == ("sshleifer/tiny-gpt2", False)
    assert calls["model_load"] == (
        "sshleifer/tiny-gpt2",
        {"local_files_only": False, "torch_dtype": benchmark.torch.float32},
    )
    assert calls["tokenizer_load"] == ("sshleifer/tiny-gpt2", {"local_files_only": False})
    assert calls["model_eval"] is True
    assert calls["hflm_kwargs"]["pretrained"].__class__ is FakeModel
    assert calls["eval_kwargs"]["model"].__class__ is FakeLMEvalHFLM
    assert calls["eval_env"] == {
        "HF_HUB_OFFLINE": "outer-hub",
        "TRANSFORMERS_OFFLINE": None,
        "HF_DATASETS_OFFLINE": "outer-datasets",
    }
    assert calls["eval_datasets_config"] is False
    assert os.environ["HF_HUB_OFFLINE"] == "outer-hub"
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert os.environ["HF_DATASETS_OFFLINE"] == "outer-datasets"
    assert fake_datasets.config.HF_DATASETS_OFFLINE is False
    assert results.loc["wikitext", "Result"] == 5.0


def test_run_hf_benchmark_restores_local_only_state_when_lm_eval_raises(monkeypatch):
    fake_datasets = SimpleNamespace(config=SimpleNamespace(HF_DATASETS_OFFLINE=False))

    class FakeModel:
        def eval(self):
            return self

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return FakeModel()

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

    class FakeTokenizerLoader:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return FakeTokenizer()

    class FakeLMEvalHFLM:
        def __init__(self, **kwargs):
            pass

    def fake_simple_evaluate(**kwargs):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ["HF_DATASETS_OFFLINE"] == "1"
        assert fake_datasets.config.HF_DATASETS_OFFLINE is True
        raise RuntimeError("lm-eval failed")

    import lm_eval.models.huggingface as lm_eval_huggingface

    monkeypatch.setattr(benchmark, "AutoModelForCausalLM", FakeModelLoader)
    monkeypatch.setattr(benchmark, "AutoTokenizer", FakeTokenizerLoader)
    monkeypatch.setattr(benchmark, "simple_evaluate", fake_simple_evaluate)
    monkeypatch.setattr(lm_eval_huggingface, "HFLM", FakeLMEvalHFLM)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setattr(
        benchmark,
        "_resolve_hf_model_source",
        lambda model_dir, local_files_only: "/cached/sshleifer/tiny-gpt2",
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "outer-hub")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "outer-transformers")
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)

    with pytest.raises(RuntimeError, match="lm-eval failed"):
        benchmark.run_hf_benchmark(
            SimpleNamespace(
                dir="sshleifer/tiny-gpt2",
                tasks="wikitext",
                batch_size=2,
                limit=1,
                local_files_only=True,
                backend="hf",
                device="cpu",
                num_fewshot=0,
            )
        )

    assert os.environ["HF_HUB_OFFLINE"] == "outer-hub"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "outer-transformers"
    assert "HF_DATASETS_OFFLINE" not in os.environ
    assert fake_datasets.config.HF_DATASETS_OFFLINE is False


def test_mohawk_benchmark_helper_passes_initialized_model_to_lm_eval(monkeypatch):
    calls = {}

    class FakeWrappedModel:
        batch_size = 4

    def fake_simple_evaluate(**kwargs):
        calls["eval_kwargs"] = kwargs
        return {
            "results": {
                "wikitext": {
                    "word_perplexity,none": 7.0,
                }
            }
        }

    monkeypatch.setattr(benchmark, "simple_evaluate", fake_simple_evaluate)
    monkeypatch.setattr(benchmark, "samples_to_file", lambda evaluation: calls.setdefault("samples", evaluation))
    monkeypatch.setattr(benchmark, "local_rank", 0)
    monkeypatch.setattr(benchmark, "world_size", 1)

    wrapped_model = FakeWrappedModel()
    results = benchmark._benchmark(
        wrapped_model=wrapped_model,
        tasks=["wikitext"],
        log_samples=False,
        limit=1,
        num_fewshot=0,
        device="cuda:0",
    )

    assert calls["eval_kwargs"]["model"] is wrapped_model
    assert calls["eval_kwargs"]["model_args"] is None
    assert calls["eval_kwargs"]["tasks"] == ["wikitext"]
    assert calls["eval_kwargs"]["batch_size"] == 4
    assert calls["eval_kwargs"]["device"] == "cuda:0"
    assert calls["eval_kwargs"]["limit"] == 1
    assert calls["eval_kwargs"]["num_fewshot"] == 0
    assert calls["eval_kwargs"]["cache_requests"] is False
    assert calls["eval_kwargs"]["delete_requests_cache"] is False
    assert calls["samples"]["results"]["wikitext"]["word_perplexity,none"] == 7.0
    assert results.loc["wikitext", "Result"] == 7.0


def test_mohawk_core_uses_shared_task_parser_before_resource_gated_work(monkeypatch):
    calls = {}
    fake_wrapper = object()

    class FakeEvaluator:
        def __init__(self, **kwargs):
            calls["evaluator_kwargs"] = kwargs

        def __call__(self, wrapper_model):
            calls["wrapper_model"] = wrapper_model
            return {"eval_score": 0.0}

    def fake_lazy_init(**kwargs):
        calls["lazy_init_kwargs"] = kwargs
        return fake_wrapper

    def fake_get_tokenizer(model_dir, **kwargs):
        calls["tokenizer_dir"] = (model_dir, kwargs)
        return "tokenizer"

    monkeypatch.setattr(benchmark, "logger", SimpleNamespace(info=lambda message: calls.setdefault("logs", []).append(message)))
    monkeypatch.setattr(benchmark, "cfg", SimpleNamespace(TrainConfig=SimpleNamespace()), raising=False)
    monkeypatch.setattr(benchmark, "lazy_init", fake_lazy_init)
    monkeypatch.setattr(benchmark, "get_tokenizer", fake_get_tokenizer)
    monkeypatch.setattr(benchmark, "evaluator", FakeEvaluator)

    benchmark._run_mohawk_benchmark(
        SimpleNamespace(
            dir="/checkpoint",
            tasks=" wikitext, , arc_easy,",
            batch_size=3,
            limit=2,
            num_fewshot=1,
            local_files_only=True,
            device="cpu",
        )
    )

    assert calls["evaluator_kwargs"]["tasks"] == ["wikitext", "arc_easy"]
    assert calls["evaluator_kwargs"]["batch_size"] == 3
    assert calls["evaluator_kwargs"]["limit"] == 2
    assert calls["evaluator_kwargs"]["num_fewshot"] == 1
    assert calls["evaluator_kwargs"]["tokenizer"] == "tokenizer"
    assert calls["tokenizer_dir"] == ("/checkpoint", {"local_files_only": True})
    assert calls["lazy_init_kwargs"]["load_cfg"].model[0].path == "/checkpoint"
    assert calls["wrapper_model"] is fake_wrapper


def test_mohawk_core_rejects_empty_tasks_before_resource_gated_init(monkeypatch):
    def fail_if_lazy_init_is_reached(*args, **kwargs):
        raise AssertionError("lazy_init should not run for an empty task list")

    monkeypatch.setattr(benchmark, "logger", SimpleNamespace(info=lambda message: None))
    monkeypatch.setattr(benchmark, "lazy_init", fail_if_lazy_init_is_reached)

    with pytest.raises(ValueError, match="At least one lm-eval task"):
        benchmark._run_mohawk_benchmark(
            SimpleNamespace(
                dir="/checkpoint",
                tasks=" , , ",
                batch_size=3,
                limit=2,
                num_fewshot=1,
            )
        )


def test_mohawk_main_enters_distributed_wrapper_before_core(monkeypatch):
    import utils.distributed as distributed

    calls = []

    @contextmanager
    def fake_init_distributed(*args, **kwargs):
        calls.append(("distributed", args, kwargs))
        yield

    monkeypatch.setattr(distributed, "init_distributed", fake_init_distributed)
    monkeypatch.setattr(benchmark, "_run_mohawk_benchmark", lambda args: calls.append(("core", args)))

    args = SimpleNamespace(dir="/checkpoint", tasks="wikitext", device=None)
    benchmark.main(args)

    assert calls[0][0] == "distributed"
    assert calls[0][1] == ()
    assert calls[0][2] == {"backend": "nccl"}
    assert calls[1] == ("core", args)


def test_can_load_hf_config_resolves_local_snapshot_for_local_files_only(monkeypatch):
    calls = {}

    class FakeConfigLoader:
        @staticmethod
        def from_pretrained(model_source, **kwargs):
            calls["config_load"] = (model_source, kwargs)

    monkeypatch.setattr(benchmark, "AutoConfig", FakeConfigLoader)
    monkeypatch.setattr(
        benchmark,
        "_resolve_hf_model_source",
        lambda model_dir, local_files_only: "/cached/sshleifer/tiny-gpt2",
    )

    assert benchmark._can_load_hf_config("sshleifer/tiny-gpt2", local_files_only=True)
    assert calls["config_load"] == (
        "/cached/sshleifer/tiny-gpt2",
        {"local_files_only": True},
    )


def test_can_load_hf_config_propagates_local_cache_miss(monkeypatch):
    def missing_snapshot(model_dir, local_files_only):
        raise benchmark.LocalHfCacheError("missing local cache")

    monkeypatch.setattr(benchmark, "_resolve_hf_model_source", missing_snapshot)

    with pytest.raises(benchmark.LocalHfCacheError, match="missing local cache"):
        benchmark._can_load_hf_config("sshleifer/tiny-gpt2", local_files_only=True)


def test_mohawk_backend_does_not_probe_huggingface_cache(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("HF config probe should not run for --backend mohawk")

    monkeypatch.setattr(benchmark, "_can_load_hf_config", fail_if_called)

    args = SimpleNamespace(
        backend="mohawk",
        dir="missing-or-private-checkpoint",
        local_files_only=True,
    )
    assert benchmark._should_run_hf_backend(args) is False
    assert benchmark._select_backend(args) == "mohawk"


def test_hf_backend_selected_when_hf_config_probe_succeeds(monkeypatch):
    calls = []

    def fake_can_load_hf_config(model_dir, local_files_only):
        calls.append((model_dir, local_files_only))
        return True

    monkeypatch.setattr(benchmark, "_can_load_hf_config", fake_can_load_hf_config)

    args = SimpleNamespace(
        backend="auto",
        dir="sshleifer/tiny-gpt2",
        local_files_only=True,
    )

    assert benchmark._select_backend(args) == "hf"
    assert calls == [("sshleifer/tiny-gpt2", True)]


@pytest.mark.hf
def test_auto_backend_selects_cached_tiny_hf_model_with_local_files_only():
    try:
        benchmark._resolve_hf_model_source("sshleifer/tiny-gpt2", local_files_only=True)
    except benchmark.LocalHfCacheError as exc:
        pytest.skip(f"sshleifer/tiny-gpt2 is not cached locally: {exc}")

    args = SimpleNamespace(
        backend="auto",
        dir="sshleifer/tiny-gpt2",
        local_files_only=True,
    )

    assert benchmark._select_backend(args) == "hf"


def test_auto_backend_falls_back_to_mohawk_when_hf_config_is_not_loadable(monkeypatch):
    calls = []

    def fake_can_load_hf_config(model_dir, local_files_only):
        calls.append((model_dir, local_files_only))
        return False

    monkeypatch.setattr(benchmark, "_can_load_hf_config", fake_can_load_hf_config)

    args = SimpleNamespace(
        backend="auto",
        dir="/checkpoint/that/is/not/an/hf/config",
        local_files_only=True,
    )

    assert benchmark._select_backend(args) == "mohawk"
    assert calls == [("/checkpoint/that/is/not/an/hf/config", True)]


def test_hf_backend_propagates_local_cache_error(monkeypatch):
    def missing_cache(model_dir, local_files_only):
        raise benchmark.LocalHfCacheError("missing local cache")

    monkeypatch.setattr(benchmark, "_can_load_hf_config", missing_cache)

    args = SimpleNamespace(
        backend="hf",
        dir="sshleifer/tiny-gpt2",
        local_files_only=True,
    )
    with pytest.raises(benchmark.LocalHfCacheError, match="missing local cache"):
        benchmark._should_run_hf_backend(args)


def test_backend_selection_propagates_local_cache_error(monkeypatch):
    def missing_cache(model_dir, local_files_only):
        raise benchmark.LocalHfCacheError("missing local cache")

    monkeypatch.setattr(benchmark, "_can_load_hf_config", missing_cache)

    args = SimpleNamespace(
        backend="auto",
        dir="sshleifer/tiny-gpt2",
        local_files_only=True,
    )
    with pytest.raises(benchmark.LocalHfCacheError, match="missing local cache"):
        benchmark._select_backend(args)


def test_hf_backend_rejects_non_hf_model_after_probe_fails(monkeypatch):
    monkeypatch.setattr(benchmark, "_should_run_hf_backend", lambda args: False)

    args = SimpleNamespace(
        backend="hf",
        dir="/checkpoint/without/hf/config",
    )

    with pytest.raises(ValueError, match="Could not load"):
        benchmark._select_backend(args)
