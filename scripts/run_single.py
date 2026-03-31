"""Run one model instance through available backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from qmodel.abstract import (
    build_abstract_trace,
    evaluate_assertion,
    evaluate_terminal_probability_assertion_on_state,
    execute_abstract_to_final_state,
)
from qmodel.concrete import (
    build_comparison_payload,
    build_comparison_payload_from_stats,
    evaluate_assertion as evaluate_concrete_assertion,
)
from qmodel.parser import parse_qmodel_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Path to one .qmodel file")
    parser.add_argument(
        "--run-concrete",
        action="store_true",
        help="Also run the concrete reference backend. Disabled by default.",
    )
    parser.add_argument(
        "--mode",
        choices=("trusted", "checked"),
        default="trusted",
        help="Abstract reconstruction mode. Default: trusted",
    )
    return parser


def _parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _evaluate_concrete(spec) -> dict[str, Any]:
    return evaluate_concrete_assertion(spec)


def _abstract_state_bytes(state) -> int:
    return sum(int(unit.witness_rho.data.nbytes) for unit in state.units)


def run_model(
    model: str | Path,
    *,
    run_concrete: bool = False,
    mode: str = "trusted",
) -> dict[str, Any]:
    total_start = perf_counter()
    model_path = Path(model).resolve()
    spec = parse_qmodel_file(str(model_path))

    abstract_start = perf_counter()
    if spec.assertions[0].kind == "probability":
        execution = execute_abstract_to_final_state(spec, reconstruction_mode=mode)
        abstract_result = evaluate_terminal_probability_assertion_on_state(
            execution.final_state,
            spec,
            reconstruction_mode=mode,
        )
        abstract_elapsed = perf_counter() - abstract_start
        max_state_bytes = execution.stats.max_state_bytes
        max_transition_bytes = execution.stats.max_transition_bytes
        comparison_payload = build_comparison_payload_from_stats(execution.stats, spec)
    else:
        trace = build_abstract_trace(spec, reconstruction_mode=mode)
        abstract_result = evaluate_assertion(trace, spec, reconstruction_mode=mode)
        abstract_elapsed = perf_counter() - abstract_start
        max_state_bytes = max((_abstract_state_bytes(state) for state in trace.states), default=0)
        max_transition_bytes = max(
            (
                int(transition.metadata.get("transition_peak_bytes", 0))
                for transition in trace.transitions
            ),
            default=0,
        )
        comparison_payload = build_comparison_payload(trace, spec)

    max_execution_bytes = max(max_state_bytes, max_transition_bytes)
    abstract_result = {
        **abstract_result,
        "elapsed_seconds": abstract_elapsed,
        "max_state_bytes": max_state_bytes,
        "max_state_mib": max_state_bytes / (1024 * 1024),
        "max_transition_bytes": max_transition_bytes,
        "max_transition_mib": max_transition_bytes / (1024 * 1024),
        "max_execution_bytes": max_execution_bytes,
        "max_execution_mib": max_execution_bytes / (1024 * 1024),
    }
    concrete_result: dict[str, Any]
    if run_concrete:
        concrete_result = _evaluate_concrete(spec)
    else:
        concrete_result = {
            "status": "skipped",
            "reason": "Concrete backend disabled by default; pass --run-concrete to enable it.",
        }

    return {
        "model_path": str(model_path),
        "program_name": spec.program_name,
        "reconstruction_mode": mode,
        "run_concrete": run_concrete,
        "assertion_name": spec.assertions[0].name,
        "assertion_kind": spec.assertions[0].kind,
        "total_elapsed_seconds": perf_counter() - total_start,
        "concrete": concrete_result,
        "abstract": abstract_result,
        "comparison": comparison_payload,
    }


def main() -> int:
    args = _parse_args()
    payload = run_model(args.model, run_concrete=args.run_concrete, mode=args.mode)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
