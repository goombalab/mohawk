import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Create a tiny local Mohawk checkpoint for run.py smoke validation."
    )
    parser.add_argument("--config", required=True, help="Smoke config containing ComponentsConfig.")
    parser.add_argument("--output", required=True, help="Output checkpoint directory.")
    parser.add_argument(
        "--rename-key",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rename one checkpoint state_dict key before saving; can be repeated.",
    )
    return parser


if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    build_arg_parser().print_help()
    raise SystemExit(0)


import torch

from utils.build_model import build_local_model
from utils.config import load_config


def main():
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config, CONSTANTS={"slurm_job_id": "0"})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    model = build_local_model(
        cfg.ComponentsConfig,
        dtype=torch.float32,
        device="cpu",
        attn_implementation="eager",
    )
    state_dict = model.state_dict()
    for rename in args.rename_key:
        if "=" not in rename:
            raise ValueError(f"--rename-key must be OLD=NEW, got {rename!r}")
        old, new = rename.split("=", 1)
        if old not in state_dict:
            raise KeyError(f"Cannot rename missing checkpoint key {old!r}")
        if new in state_dict and new != old:
            raise KeyError(f"Cannot rename {old!r} to existing checkpoint key {new!r}")
        state_dict[new] = state_dict.pop(old)
    torch.save(state_dict, output / "pytorch_model.bin")
    with open(output / "config.json", "w", encoding="utf-8") as handle:
        json.dump({"ComponentsConfig": cfg.ComponentsConfig.to_dict()}, handle, indent=2)
    print(f"Saved tiny smoke checkpoint to {output}")


if __name__ == "__main__":
    main()
