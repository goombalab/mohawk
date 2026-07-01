import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


HELP_COMMANDS = [
    ["run.py"],
    ["evals/benchmark.py"],
    ["evals/eval_ppl.py"],
    ["generation/generate.py"],
    ["tools/create_tiny_smoke_checkpoint.py"],
    ["tools/hybrid_weights_transfer.py"],
    ["tools/benchmark_throughput.py"],
    ["tools/visualize_attention.py"],
]
README_FRESH_CHECKOUT_HELP_COMMANDS = {
    ("run.py",),
    ("evals/benchmark.py",),
    ("generation/generate.py",),
}


def test_readme_preserves_publication_citations():
    readme = (ROOT / "README.md").read_text()

    for citation_key in [
        "@misc{bick2026retrieval,",
        "@article{bick2025llamba,",
        "@misc{paliotta2025thinking,",
        "@misc{mohawk,",
    ]:
        assert citation_key in readme


def test_public_cli_help_does_not_import_heavy_runtime(command):
    if tuple(command) in README_FRESH_CHECKOUT_HELP_COMMANDS:
        readme = (ROOT / "README.md").read_text()
        assert f"python {' '.join(command)} --help" in readme

    result = subprocess.run(
        [sys.executable, *command, "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "Traceback" not in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    if command == ["run.py"]:
        assert "--config" in result.stdout
    if command == ["evals/benchmark.py"]:
        for flag in ["--limit", "--local_files_only", "--backend", "--device", "--model-registration-module"]:
            assert flag in result.stdout
        assert "lm-eval task data" in result.stdout
    if command == ["evals/eval_ppl.py"]:
        for flag in ["--model", "--backend", "--text", "--n_batches", "--max_seq_len", "--batch_size", "--local_files_only", "--device", "--model-registration-module"]:
            assert flag in result.stdout
    if command == ["generation/generate.py"]:
        for flag in ["--model", "--prompt", "--local_files_only", "--model_dtype", "--model-registration-module"]:
            assert flag in result.stdout
    if command == ["tools/create_tiny_smoke_checkpoint.py"]:
        for flag in ["--config", "--output", "--rename-key"]:
            assert flag in result.stdout
    if command == ["tools/hybrid_weights_transfer.py"]:
        for flag in ["--config", "--heads", "--device", "--allow-unexpected-student-load"]:
            assert flag in result.stdout
    if command == ["tools/benchmark_throughput.py"]:
        for flag in [
            "--model",
            "--hf-model-name",
            "--local_files_only",
            "--llamba-config-only",
            "--batch-sizes",
        ]:
            assert flag in result.stdout
    if command == ["tools/visualize_attention.py"]:
        for flag in ["--model_name", "--output_dir", "--local_files_only", "--device", "--heads", "--mode", "--layers", "--model-registration-module"]:
            assert flag in result.stdout


def test_run_without_config_fails_before_training_imports():
    result = subprocess.run(
        [sys.executable, "run.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "--config" in result.stderr


def test_throughput_benchmark_uses_synchronized_cuda_event_timing():
    source = (ROOT / "tools/benchmark_throughput.py").read_text()

    assert "torch.cuda.Event(enable_timing=True)" in source
    assert 'static_inputs["num_last_tokens"] = 1' in source
    assert "Prefill batch; Prompt tokens; Time (s); Throughput (tokens/s)" in source
    assert "end_event.synchronize()" in source
    assert "start_event.elapsed_time(end_event)" in source
    assert "time.time()" not in source


def pytest_generate_tests(metafunc):
    if "command" in metafunc.fixturenames:
        metafunc.parametrize("command", HELP_COMMANDS, ids=[" ".join(cmd) for cmd in HELP_COMMANDS])
