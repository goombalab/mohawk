"""Path guard for component-level imports.

Some optional CUDA/SSM dependency stacks install a top-level ``utils`` package.
Mohawk also has a top-level ``utils`` package, so direct component imports from
outside the repository can otherwise resolve local imports against the external
package when site-packages appears first on ``sys.path``.
"""

from pathlib import Path
import sys


def ensure_repo_root_on_path():
    repo_root = str(Path(__file__).resolve().parents[1])
    if sys.path and sys.path[0] == repo_root:
        return
    try:
        sys.path.remove(repo_root)
    except ValueError:
        pass
    sys.path.insert(0, repo_root)
