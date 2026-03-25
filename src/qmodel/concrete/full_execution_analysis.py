"""Comparison helpers for abstract and full-execution resource baselines."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any, Sequence

from qmodel.abstract.state import AbstractUnitState
from qmodel.concrete.qiskit_backend import (
    ConcreteBackendError,
    _evaluate_probability_assertion_from_statevector,
    evaluate_assertion,
    simulate_statevector,
)
from qmodel.spec import QuantumProgramSpec

if TYPE_CHECKING:
    from qmodel.abstract.transition import AbstractExecutionStats, AbstractExecutionTrace


COMPLEX128_BYTES = 16
DEFAULT_FULL_EXECUTION_TIME_CUTOFF_QUBITS = 22


def _mib(byte_count: int) -> float:
    return byte_count / (1024 * 1024)


def _pure_statevector_bytes(width: int) -> int:
    return (2**width) * COMPLEX128_BYTES


def _density_matrix_bytes(width: int) -> int:
    return (2 ** (2 * width)) * COMPLEX128_BYTES


def _unit_label(unit: AbstractUnitState) -> str:
    return unit.name if unit.name is not None else "|".join(unit.qubits)


def _pure_units_bytes(units: Sequence[AbstractUnitState]) -> int:
    return sum(_pure_statevector_bytes(len(unit.qubits)) for unit in units)


def abstract_ideal_pure_lower_bound(trace: AbstractExecutionTrace) -> dict[str, float | int]:
    max_state_bytes = max((_pure_units_bytes(state.units) for state in trace.states), default=0)

    max_transition_bytes = 0
    for transition, target_state in zip(trace.transitions, trace.states[1:]):
        workspace_qubits = transition.metadata.get("workspace_qubits", ())
        affected_post_views = set(transition.metadata.get("affected_post_views", ()))
        unaffected_units = [
            unit for unit in target_state.units if _unit_label(unit) not in affected_post_views
        ]
        transition_bytes = _pure_statevector_bytes(len(workspace_qubits)) + _pure_units_bytes(
            unaffected_units
        )
        max_transition_bytes = max(max_transition_bytes, transition_bytes)

    max_execution_bytes = max(max_state_bytes, max_transition_bytes)
    return {
        "max_state_bytes": max_state_bytes,
        "max_state_mib": _mib(max_state_bytes),
        "max_transition_bytes": max_transition_bytes,
        "max_transition_mib": _mib(max_transition_bytes),
        "max_execution_bytes": max_execution_bytes,
        "max_execution_mib": _mib(max_execution_bytes),
    }


def abstract_ideal_pure_lower_bound_from_stats(
    stats: AbstractExecutionStats,
) -> dict[str, float | int]:
    max_state_bytes = stats.max_ideal_pure_state_bytes
    max_transition_bytes = stats.max_ideal_pure_transition_bytes
    max_execution_bytes = max(max_state_bytes, max_transition_bytes)
    return {
        "max_state_bytes": max_state_bytes,
        "max_state_mib": _mib(max_state_bytes),
        "max_transition_bytes": max_transition_bytes,
        "max_transition_mib": _mib(max_transition_bytes),
        "max_execution_bytes": max_execution_bytes,
        "max_execution_mib": _mib(max_execution_bytes),
    }


def full_execution_baseline(
    spec: QuantumProgramSpec,
    *,
    time_cutoff_qubits: int = DEFAULT_FULL_EXECUTION_TIME_CUTOFF_QUBITS,
) -> dict[str, Any]:
    qubit_count = len(spec.qubits)
    statevector_bytes = _pure_statevector_bytes(qubit_count)
    density_matrix_bytes = _density_matrix_bytes(qubit_count)

    time_benchmark: dict[str, Any]
    if qubit_count > time_cutoff_qubits:
        time_benchmark = {
            "mode": "skipped",
            "cutoff_qubits": time_cutoff_qubits,
            "reason": "qubit count exceeds automatic timing cutoff",
        }
    else:
        try:
            start = perf_counter()
            state = simulate_statevector(spec)
            statevector_elapsed = perf_counter() - start
            assertion_elapsed: float | None = None
            concrete_backend_elapsed: float

            if len(spec.assertions) == 1 and spec.assertions[0].kind == "probability":
                assertion_start = perf_counter()
                _evaluate_probability_assertion_from_statevector(spec, state)
                assertion_elapsed = perf_counter() - assertion_start
                concrete_backend_elapsed = statevector_elapsed + assertion_elapsed
            else:
                concrete_start = perf_counter()
                evaluate_assertion(spec)
                concrete_backend_elapsed = perf_counter() - concrete_start

            time_benchmark = {
                "mode": "measured",
                "cutoff_qubits": time_cutoff_qubits,
                "statevector_elapsed_seconds": statevector_elapsed,
                "concrete_backend_elapsed_seconds": concrete_backend_elapsed,
                "assertion_evaluation_seconds": assertion_elapsed,
            }
        except ConcreteBackendError as exc:
            time_benchmark = {
                "mode": "skipped",
                "cutoff_qubits": time_cutoff_qubits,
                "reason": str(exc),
            }

    return {
        "qubit_count": qubit_count,
        "statevector_bytes": statevector_bytes,
        "statevector_mib": _mib(statevector_bytes),
        "density_matrix_bytes": density_matrix_bytes,
        "density_matrix_mib": _mib(density_matrix_bytes),
        "time_benchmark": time_benchmark,
    }


def build_comparison_payload(
    trace: AbstractExecutionTrace,
    spec: QuantumProgramSpec,
    *,
    time_cutoff_qubits: int = DEFAULT_FULL_EXECUTION_TIME_CUTOFF_QUBITS,
) -> dict[str, Any]:
    return {
        "abstract_ideal_pure_lower_bound": abstract_ideal_pure_lower_bound(trace),
        "full_execution": full_execution_baseline(
            spec,
            time_cutoff_qubits=time_cutoff_qubits,
        ),
    }


def build_comparison_payload_from_stats(
    stats: AbstractExecutionStats,
    spec: QuantumProgramSpec,
    *,
    time_cutoff_qubits: int = DEFAULT_FULL_EXECUTION_TIME_CUTOFF_QUBITS,
) -> dict[str, Any]:
    return {
        "abstract_ideal_pure_lower_bound": abstract_ideal_pure_lower_bound_from_stats(stats),
        "full_execution": full_execution_baseline(
            spec,
            time_cutoff_qubits=time_cutoff_qubits,
        ),
    }
