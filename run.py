import argparse
import sys


def _help_parser():
    parser = argparse.ArgumentParser(
        description="Train/distill a Mohawk model from one or more YAML configs."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config, or a comma-separated list of YAML configs.",
    )
    return parser


if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    _help_parser().print_help()
    raise SystemExit(0)

if "--config" not in sys.argv[1:]:
    _help_parser().parse_args()


import utils.set_basic_libraries # MUST BE FIRST after lightweight help handling
from distill.distill import distill

if __name__ == "__main__":
    distill()
