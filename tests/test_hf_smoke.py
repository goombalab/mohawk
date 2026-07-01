import json
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

import pytest

pytestmark = [pytest.mark.hf, pytest.mark.slow]

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "sshleifer/tiny-gpt2"


def test_build_model_resolves_local_hf_snapshot_for_generation(monkeypatch, tmp_path):
    from utils import build_model

    local_model = tmp_path / "local-model"
    local_model.mkdir()
    assert build_model.resolve_hf_model_source(str(local_model), local_files_only=True) == str(local_model)
    assert build_model.resolve_hf_model_source(MODEL_ID, local_files_only=False) == MODEL_ID

    calls = {}

    def fake_snapshot_download(model_id, **kwargs):
        calls["snapshot"] = (model_id, kwargs)
        return "/cached/sshleifer/tiny-gpt2"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    assert build_model.resolve_hf_model_source(MODEL_ID, local_files_only=True) == "/cached/sshleifer/tiny-gpt2"
    assert calls["snapshot"] == (MODEL_ID, {"local_files_only": True})


def test_build_model_imports_optional_transformers_registration_modules():
    from utils import build_model

    imported = build_model.import_model_registration_modules(["json"])
    assert [module.__name__ for module in imported] == ["json"]

    with pytest.raises(RuntimeError, match="Could not import model registration module"):
        build_model.import_model_registration_modules(
            ["mohawk_test_module_that_does_not_exist"]
        )


def test_hidden_state_similarity_visualization_helpers(tmp_path):
    from tools import visualize_attention
    from PIL import Image

    hidden_states = tuple(torch.randn(1, 5, 8) for _ in range(5))
    selected = visualize_attention.select_valid_layers(hidden_states)
    assert selected == [0, 2, 4]
    assert visualize_attention.parse_layers("0, 2,4") == [0, 2, 4]
    with pytest.raises(ValueError, match="At least one hidden-state index"):
        visualize_attention.parse_layers(" , ")

    matrices = visualize_attention.hidden_state_similarity_matrices(
        hidden_states, selected
    )
    assert set(matrices) == {(0, 0), (2, 0), (4, 0)}
    for matrix in matrices.values():
        assert matrix.shape == (5, 5)
        assert torch.isfinite(matrix).all()
        assert torch.allclose(matrix, matrix.transpose(0, 1))
        assert torch.allclose(matrix.diagonal(), torch.ones(5), atol=1e-5)

    with pytest.raises(ValueError, match="outside the loaded model shape"):
        visualize_attention.select_valid_layers(hidden_states, [5])

    output_path = tmp_path / "hidden-similarity.png"
    visualize_attention.visualize_attention_heads_pil(
        matrices=matrices,
        file_name=output_path,
        pred="17",
        example_important_heads=set(),
        model_name="namespace/representative-model-with-a-long-name",
        layer_count=len(hidden_states),
        num_heads=1,
        choices=["17", "23", "4", "8"],
        matrix_kind="hidden_similarity",
    )
    with Image.open(output_path) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.width >= 512
        assert image.height > image.width


def test_build_model_get_tokenizer_recovers_from_mohawk_checkpoint_config(tmp_path):
    tiny_model_and_tokenizer()
    from utils import build_model

    checkpoint_dir = tmp_path / "mohawk-checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"name": "LayeredMambaLM", "input": {"vocab_size": 50257}})
    )
    (checkpoint_dir / "config.yaml").write_text(
        "TrainConfig:\n"
        f"  tokenizer: {MODEL_ID}\n"
        "TrainDataConfig:\n"
        "  Tokenize:\n"
        f"    tokenizer: {MODEL_ID}\n"
        "    local_files_only: true\n"
    )

    tokenizer = build_model.get_tokenizer(str(checkpoint_dir), local_files_only=False)
    assert tokenizer.name_or_path == MODEL_ID


def tiny_model_and_tokenizer():
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, local_files_only=True)
    except OSError as exc:
        pytest.skip(f"{MODEL_ID} is not available in the local Hugging Face cache: {exc}")
    return model, tokenizer


def test_cached_tiny_hf_model_forward_backward_optimizer_checkpoint_reload():
    model, tokenizer = tiny_model_and_tokenizer()
    input_ids = tokenizer("hello world", return_tensors="pt").input_ids
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    outputs = model(input_ids, labels=input_ids)
    outputs.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        tokenizer.save_pretrained(tmp)
        reloaded = transformers.AutoModelForCausalLM.from_pretrained(tmp, local_files_only=True)
        logits = reloaded(input_ids).logits

    assert logits.shape[:2] == input_ids.shape


def test_generation_script_runs_with_cached_tiny_hf_model():
    tiny_model_and_tokenizer()
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "generation/generate.py",
            "--model",
            MODEL_ID,
            "--local_files_only",
            "--prompt",
            "hello",
            "--genlen",
            "2",
            "--repeats",
            "1",
            "--top_k",
            "1",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "Generated text:" in result.stdout


def test_eval_ppl_script_runs_with_cached_tiny_hf_model():
    tiny_model_and_tokenizer()
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "evals/eval_ppl.py",
            "--model",
            MODEL_ID,
            "--local_files_only",
            "--device",
            "cpu",
            "--n_batches",
            "1",
            "--max_seq_len",
            "16",
            "--batch_size",
            "1",
            "--text",
            "hello world from mohawk",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert '"perplexity":' in result.stdout
    assert '"accuracy":' in result.stdout


def test_run_py_tiny_cpu_supervised_training_smoke():
    tiny_model_and_tokenizer()
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["WANDB_MODE"] = "disabled"

    checkpoint_result = subprocess.run(
        [
            sys.executable,
            "tools/create_tiny_smoke_checkpoint.py",
            "--config",
            "configs/smoke/tiny_cpu_supervised.yaml",
            "--output",
            "/tmp/mohawk_tiny_teacher_ckpt",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert checkpoint_result.returncode == 0, checkpoint_result.stderr
    assert "Saved tiny smoke checkpoint" in checkpoint_result.stdout

    env["MOHAWK_ALLOW_CPU_TRAINING"] = "1"
    train_result = subprocess.run(
        [
            sys.executable,
            "run.py",
            "--config",
            "configs/smoke/tiny_cpu_supervised.yaml",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=240,
    )
    assert train_result.returncode == 0, train_result.stderr
    combined_output = train_result.stdout + train_result.stderr
    assert "MOHAWK_ALLOW_CPU_TRAINING=1" in combined_output
    assert "Checkpoint saved successfully" in combined_output
    assert "Training finished" in combined_output

    save_dir = Path("/tmp/mohawk_tiny_cpu_run/save")
    assert (save_dir / "model.safetensors").is_file()
    assert (save_dir / "optimizer.pth").is_file()
    assert (save_dir / "scheduler.pth").is_file()
    assert (save_dir / "config.json").is_file()
    assert torch.load(save_dir / "scheduler.pth", map_location="cpu")["_step_count"] == 2
