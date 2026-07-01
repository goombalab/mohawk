import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_component_repo_path_guard_preempts_external_utils(tmp_path):
    site_dir = tmp_path / "site"
    external_utils = site_dir / "utils"
    external_utils.mkdir(parents=True)
    (external_utils / "__init__.py").write_text("ORIGIN = 'external'\n")

    script = f"""
import sys
sys.path[:] = [{str(site_dir)!r}, {str(ROOT)!r}] + [
    p for p in sys.path if p not in ({str(site_dir)!r}, {str(ROOT)!r})
]
from components._repo_path import ensure_repo_root_on_path
ensure_repo_root_on_path()
import utils
assert sys.path[0] == {str(ROOT)!r}, sys.path[:3]
assert utils.__file__.startswith({str(ROOT / 'utils')!r}), utils.__file__
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_component_modules_guard_before_local_utils_imports():
    paths = [
        ROOT / "components" / "LMHeads" / "LayeredMambaLM.py",
        ROOT / "components" / "MixerModels" / "LlamaModel.py",
        ROOT / "components" / "MixerModels" / "Qwen2Model.py",
        ROOT / "components" / "cores" / "discrete_mamba2.py",
        ROOT / "components" / "cores" / "discrete_mamba2_rotary.py",
    ]
    for path in paths:
        text = path.read_text()
        assert "from components._repo_path import ensure_repo_root_on_path" in text
        assert "ensure_repo_root_on_path()" in text
        assert text.index("ensure_repo_root_on_path()") < text.index("from utils.")
