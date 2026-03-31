"""Convenient entrypoint for running one or many qmodel instances."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_benchmarks import discover_model_paths, run_models
from scripts.run_single import run_model


_DEFAULT_MODELS_ROOT = Path(__file__).resolve().parents[1] / "experiment_data" / "models"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--run-concrete",
        action="store_true",
        help="Also run the concrete reference backend. Disabled by default.",
    )
    common.add_argument(
        "--mode",
        choices=("trusted", "checked"),
        default="trusted",
        help="Abstract reconstruction mode. Default: trusted",
    )

    run_parser = subparsers.add_parser("run", parents=[common], help="Run one .qmodel file")
    run_parser.add_argument("model", help="Path to one .qmodel file")

    run_all_parser = subparsers.add_parser(
        "run-all",
        parents=[common],
        help="Run all discovered .qmodel files under experiment_data/models",
    )
    run_all_parser.add_argument(
        "--family",
        help="Optional family directory filter, for example GHZ or IQFT.",
    )
    run_all_parser.add_argument(
        "--models-root",
        default=str(_DEFAULT_MODELS_ROOT),
        help="Root directory used to discover .qmodel files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "run":
        payload = run_model(args.model, run_concrete=args.run_concrete, mode=args.mode)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    model_paths = discover_model_paths(Path(args.models_root), family=args.family)
    payload = run_models(model_paths, run_concrete=args.run_concrete, mode=args.mode)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
