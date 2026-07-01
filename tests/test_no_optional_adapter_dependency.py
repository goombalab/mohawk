from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = [
    REPO_ROOT / name
    for name in (
        ".github",
        "components",
        "configs",
        "dataloaders",
        "distill",
        "evals",
        "external_models",
        "generation",
        "tests",
        "tools",
        "training_wrapper",
        "utils",
    )
]
SEARCH_FILES = [
    REPO_ROOT / "CONTRIBUTING.md",
    REPO_ROOT / "README.md",
    REPO_ROOT / "requirements.txt",
    REPO_ROOT / "requirements-ssm-cuda.txt",
]
TEXT_SUFFIXES = {".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}


def test_removed_adapter_dependency_is_not_referenced():
    package_name = "pe" + "ft"
    candidates = list(SEARCH_FILES)
    for root in SEARCH_ROOTS:
        if root.exists():
            candidates.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix in TEXT_SUFFIXES
            )

    offenders = [
        path.relative_to(REPO_ROOT)
        for path in candidates
        if package_name in path.read_text(encoding="utf-8").lower()
    ]

    assert offenders == []
