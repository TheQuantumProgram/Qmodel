"""Property checking on abstract states."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from qiskit.quantum_info import DensityMatrix, partial_trace

from qmodel.abstract.state import AbstractState
from qmodel.abstract.transition import _ensure_checked_scope_consistency, _unit_identity
from qmodel.abstract.transition import (
    AbstractExecutionTrace,
    ReconstructionMode,
    reconstruct_scope_state,
)
from qmodel.spec import QuantumProgramSpec
from qmodel.validation import validate_program_spec


_COMPARISON_TOLERANCE = 1e-9


class AbstractPropertyCheckingError(ValueError):
    """Raised when an abstract property cannot be evaluated in the current stage."""


def state_scope_witness(
    trace: AbstractExecutionTrace,
    spec: QuantumProgramSpec,
    state_index: int,
    scope: list[str],
    reconstruction_mode: ReconstructionMode = "trusted",
) -> DensityMatrix:
    """Return the witness state for one trace state on a target scope."""

    if state_index < 0 or state_index >= len(trace.states):
        raise AbstractPropertyCheckingError(f"state_index out of range: {state_index}")

    try:
        return reconstruct_scope_state(
            trace.states[state_index],
            spec.qubits,
            scope,
            mode=reconstruction_mode,
        )
    except ValueError as exc:
        raise AbstractPropertyCheckingError(str(exc)) from exc


def final_scope_witness(
    trace: AbstractExecutionTrace,
    spec: QuantumProgramSpec,
    scope: list[str],
    reconstruction_mode: ReconstructionMode = "trusted",
) -> DensityMatrix:
    """Return the final witness state for a target scope."""

    return state_scope_witness(
        trace, spec, len(trace.states) - 1, scope, reconstruction_mode=reconstruction_mode
    )


def final_scope_witness_from_state(
    state: AbstractState,
    spec: QuantumProgramSpec,
    scope: list[str],
    reconstruction_mode: ReconstructionMode = "trusted",
) -> DensityMatrix:
    """Return the witness state for one final abstract state on a target scope."""

    try:
        return reconstruct_scope_state(
            state,
            spec.qubits,
            scope,
            mode=reconstruction_mode,
        )
    except ValueError as exc:
        raise AbstractPropertyCheckingError(str(exc)) from exc


def _basis_bitstring(index: int, width: int) -> str:
    return format(index, f"0{width}b")[::-1]


def _measurement_outcome_probability_from_witness(
    witness: DensityMatrix, outcomes: list[str]
) -> float:
    if not outcomes:
        raise AbstractPropertyCheckingError("measurement outcome set must be non-empty")

    width = witness.num_qubits
    normalized_outcomes = {outcome.strip() for outcome in outcomes}
    for outcome in normalized_outcomes:
        if len(outcome) != width or any(bit not in "01" for bit in outcome):
            raise AbstractPropertyCheckingError(
                f"Invalid measurement outcome {outcome!r} for scope width {width}"
            )

    probability = 0.0
    diagonal = witness.probabilities()
    for basis_index, basis_probability in enumerate(diagonal):
        if basis_probability == 0.0:
            continue
        if _basis_bitstring(basis_index, width) in normalized_outcomes:
            probability += float(basis_probability)
    return probability


def _single_bit_probability_from_witness(
    witness: DensityMatrix, expected_bit: str
) -> float:
    normalized_bit = expected_bit.strip()
    if normalized_bit not in {"0", "1"}:
        raise AbstractPropertyCheckingError(
            f"Invalid single-bit outcome {expected_bit!r}"
    )
    return _measurement_outcome_probability_from_witness(witness, [normalized_bit])


def _reduce_local_witness(
    witness: DensityMatrix,
    witness_qubits: list[str] | tuple[str, ...],
    scope: list[str] | tuple[str, ...],
) -> DensityMatrix:
    trace_out = [index for index, qubit in enumerate(witness_qubits) if qubit not in scope]
    return partial_trace(witness, trace_out) if trace_out else witness


def _basis_probability_on_local_scope(
    witness: DensityMatrix,
    witness_qubits: list[str] | tuple[str, ...],
    scope: list[str] | tuple[str, ...],
    outcome: str,
) -> float:
    reduced = _reduce_local_witness(witness, witness_qubits, scope)
    return _measurement_outcome_probability_from_witness(reduced, [outcome])


def _restricted_unit_entries(
    state: AbstractState,
    scope: list[str],
    outcome: str,
) -> list[dict[str, Any]]:
    bit_by_qubit = dict(zip(scope, outcome, strict=True))
    entries: list[dict[str, Any]] = []
    for unit in state.units:
        restricted_qubits = tuple(qubit for qubit in unit.qubits if qubit in bit_by_qubit)
        if not restricted_qubits:
            continue
        restricted_outcome = "".join(bit_by_qubit[qubit] for qubit in restricted_qubits)
        entries.append(
            {
                "label": unit.name if unit.name is not None else "|".join(unit.qubits),
                "identity": _unit_identity(unit.name, unit.qubits),
                "unit": unit,
                "restricted_qubits": restricted_qubits,
                "restricted_outcome": restricted_outcome,
            }
        )
    return entries


def _connected_components(entries: list[dict[str, Any]]) -> list[list[int]]:
    adjacency = {index: set() for index in range(len(entries))}
    for left_index in range(len(entries)):
        left_qubits = set(entries[left_index]["restricted_qubits"])
        for right_index in range(left_index + 1, len(entries)):
            right_qubits = set(entries[right_index]["restricted_qubits"])
            if left_qubits & right_qubits:
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)

    components: list[list[int]] = []
    remaining = set(range(len(entries)))
    while remaining:
        seed = min(remaining)
        queue = [seed]
        component = []
        remaining.remove(seed)
        while queue:
            current = queue.pop(0)
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _entry_probability(entry: dict[str, Any], qubits: tuple[str, ...]) -> float:
    outcome = "".join(
        bit for qubit, bit in zip(entry["restricted_qubits"], entry["restricted_outcome"], strict=True)
        if qubit in qubits
    )
    return _basis_probability_on_local_scope(
        entry["unit"].witness_rho,
        entry["unit"].qubits,
        qubits,
        outcome,
    )


def _factorized_basis_probability_from_state(
    state: AbstractState,
    spec: QuantumProgramSpec,
    scope: list[str],
    outcome: str,
    reconstruction_mode: ReconstructionMode = "trusted",
) -> float:
    entries = _restricted_unit_entries(state, scope, outcome)
    if not entries:
        raise AbstractPropertyCheckingError("No final abstract units intersect the requested scope")

    if reconstruction_mode == "checked":
        try:
            _ensure_checked_scope_consistency([entry["unit"] for entry in entries])
        except ValueError as exc:
            raise AbstractPropertyCheckingError(str(exc)) from exc

    covered = {qubit for entry in entries for qubit in entry["restricted_qubits"]}
    missing = [qubit for qubit in scope if qubit not in covered]
    if missing:
        raise AbstractPropertyCheckingError(
            "Final abstract units do not cover the requested scope: " + ", ".join(missing)
        )

    total_probability = 1.0
    seen_qubits: set[str] = set()
    for entry in entries:
        restricted_qubits = entry["restricted_qubits"]
        new_qubits = tuple(qubit for qubit in restricted_qubits if qubit not in seen_qubits)
        separator_qubits = tuple(qubit for qubit in restricted_qubits if qubit in seen_qubits)

        if not new_qubits:
            seen_qubits.update(restricted_qubits)
            continue

        local_probability = _entry_probability(entry, restricted_qubits)
        if local_probability <= _COMPARISON_TOLERANCE:
            return 0.0

        if not separator_qubits:
            total_probability *= local_probability
        else:
            separator_probability = _entry_probability(entry, separator_qubits)
            if separator_probability <= _COMPARISON_TOLERANCE:
                raise AbstractPropertyCheckingError(
                    "Separator probability is zero for a non-zero child assignment in final factorized evaluation"
                )
            total_probability *= local_probability / separator_probability

        seen_qubits.update(restricted_qubits)

    return total_probability


def _bitwise_measurement_outcome_probability_from_trace(
    trace: AbstractExecutionTrace,
    spec: QuantumProgramSpec,
    scope: list[str],
    outcome: str,
    reconstruction_mode: ReconstructionMode = "trusted",
) -> list[float]:
    normalized_outcome = outcome.strip()
    if len(normalized_outcome) != len(scope) or any(bit not in "01" for bit in normalized_outcome):
        raise AbstractPropertyCheckingError(
            f"Invalid bitwise measurement outcome {outcome!r} for scope width {len(scope)}"
        )

    bitwise_probabilities: list[float] = []
    for qubit, expected_bit in zip(scope, normalized_outcome, strict=True):
        witness = final_scope_witness(
            trace,
            spec,
            [qubit],
            reconstruction_mode=reconstruction_mode,
        )
        bitwise_probabilities.append(
            _single_bit_probability_from_witness(witness, expected_bit)
        )
    return bitwise_probabilities


def _bitwise_measurement_outcome_probability_from_state(
    state: AbstractState,
    spec: QuantumProgramSpec,
    scope: list[str],
    outcome: str,
    reconstruction_mode: ReconstructionMode = "trusted",
) -> list[float]:
    normalized_outcome = outcome.strip()
    if len(normalized_outcome) != len(scope) or any(bit not in "01" for bit in normalized_outcome):
        raise AbstractPropertyCheckingError(
            f"Invalid bitwise measurement outcome {outcome!r} for scope width {len(scope)}"
        )

    bitwise_probabilities: list[float] = []
    for qubit, expected_bit in zip(scope, normalized_outcome, strict=True):
        witness = final_scope_witness_from_state(
            state,
            spec,
            [qubit],
            reconstruction_mode=reconstruction_mode,
        )
        bitwise_probabilities.append(_single_bit_probability_from_witness(witness, expected_bit))
    return bitwise_probabilities


def _basis_state_projector(width: int, state: str) -> np.ndarray:
    normalized_state = state.strip()
    if len(normalized_state) != width or any(bit not in "01" for bit in normalized_state):
        raise AbstractPropertyCheckingError(
            f"Invalid basis state {state!r} for scope width {width}"
        )

    basis_vector = np.zeros(2**width, dtype=complex)
    basis_vector[int(normalized_state[::-1], 2)] = 1.0
    return np.outer(basis_vector, basis_vector.conj())


def _projector_overlap(witness: DensityMatrix, projector: np.ndarray) -> float:
    overlap = np.trace(projector @ witness.data)
    return float(np.real_if_close(overlap))


def evaluate_reachability_assertion(
    trace: AbstractExecutionTrace,
    spec: QuantumProgramSpec,
    reconstruction_mode: ReconstructionMode = "trusted",
) -> dict[str, Any]:
    """Evaluate the single reachability assertion on the abstract trace."""

    validate_program_spec(spec)

    if len(spec.assertions) != 1:
        raise AbstractPropertyCheckingError("Expected exactly one assertion")

    assertion = spec.assertions[0]
    if assertion.kind != "reachability":
        raise AbstractPropertyCheckingError("Only reachability assertions are supported")

    if assertion.target.get("type") != "basis_state":
        raise AbstractPropertyCheckingError("Only basis_state reachability targets are supported")

    scope = assertion.target["scope"]
    state = assertion.target.get("state")
    if not isinstance(state, str) or not state.strip():
        raise AbstractPropertyCheckingError(
            "basis_state reachability assertions require a non-empty 'state' field"
        )

    projector = _basis_state_projector(len(scope), state)
    max_overlap = 0.0
    first_reached_index: int | None = None

    for state_index in range(len(trace.states)):
        witness = state_scope_witness(
            trace, spec, state_index, scope, reconstruction_mode=reconstruction_mode
        )
        overlap = _projector_overlap(witness, projector)
        max_overlap = max(max_overlap, overlap)
        if overlap > _COMPARISON_TOLERANCE and first_reached_index is None:
            first_reached_index = state_index

    return {
        "max_overlap": max_overlap,
        "first_reached_index": first_reached_index,
        "judgment": "satisfied" if first_reached_index is not None else "violated",
    }


def evaluate_terminal_probability_assertion(
    trace: AbstractExecutionTrace,
    spec: QuantumProgramSpec,
    reconstruction_mode: ReconstructionMode = "trusted",
) -> dict[str, float | str]:
    """Evaluate the single terminal probability assertion on the abstract trace."""

    return evaluate_terminal_probability_assertion_on_state(
        trace.states[-1], spec, reconstruction_mode=reconstruction_mode
    )


def evaluate_terminal_probability_assertion_on_state(
    state: AbstractState,
    spec: QuantumProgramSpec,
    reconstruction_mode: ReconstructionMode = "trusted",
) -> dict[str, float | str]:
    """Evaluate the single terminal probability assertion on one final abstract state."""

    validate_program_spec(spec)

    if len(spec.assertions) != 1:
        raise AbstractPropertyCheckingError("Expected exactly one assertion")

    assertion = spec.assertions[0]
    if assertion.kind != "probability":
        raise AbstractPropertyCheckingError("Only probability assertions are supported")

    threshold = float(assertion.threshold)
    comparator = assertion.comparator

    target_type = assertion.target.get("type")
    if target_type == "measurement_outcome":
        scope = assertion.target["scope"]
        outcomes = assertion.target.get("outcomes")
        if not isinstance(outcomes, list) or not all(isinstance(item, str) for item in outcomes):
            raise AbstractPropertyCheckingError(
                "measurement_outcome probability assertions require a string-list 'outcomes' field"
            )

        if len(outcomes) == 1:
            probability = _factorized_basis_probability_from_state(
                state,
                spec,
                scope,
                outcomes[0].strip(),
                reconstruction_mode=reconstruction_mode,
            )
        else:
            witness = final_scope_witness_from_state(
                state, spec, scope, reconstruction_mode=reconstruction_mode
            )
            probability = _measurement_outcome_probability_from_witness(witness, outcomes)
    elif target_type == "bitwise_measurement_outcome":
        scope = assertion.target["scope"]
        outcome = assertion.target.get("outcome")
        if not isinstance(outcome, str) or not outcome.strip():
            raise AbstractPropertyCheckingError(
                "bitwise_measurement_outcome probability assertions require a non-empty 'outcome' field"
            )

        bitwise_probabilities = _bitwise_measurement_outcome_probability_from_state(
            state,
            spec,
            scope,
            outcome,
            reconstruction_mode=reconstruction_mode,
        )
        probability = min(bitwise_probabilities) if comparator != "<=" else max(bitwise_probabilities)
    else:
        raise AbstractPropertyCheckingError(
            "Only measurement_outcome and bitwise_measurement_outcome targets are supported in this stage"
        )

    if comparator == ">=":
        if target_type == "bitwise_measurement_outcome":
            satisfied = all(
                bit_probability >= threshold - _COMPARISON_TOLERANCE
                for bit_probability in bitwise_probabilities
            )
        else:
            satisfied = probability >= threshold - _COMPARISON_TOLERANCE
    elif comparator == "<=":
        if target_type == "bitwise_measurement_outcome":
            satisfied = all(
                bit_probability <= threshold + _COMPARISON_TOLERANCE
                for bit_probability in bitwise_probabilities
            )
        else:
            satisfied = probability <= threshold + _COMPARISON_TOLERANCE
    elif comparator == "=":
        if target_type == "bitwise_measurement_outcome":
            satisfied = all(
                math.isclose(
                    bit_probability,
                    threshold,
                    rel_tol=_COMPARISON_TOLERANCE,
                    abs_tol=_COMPARISON_TOLERANCE,
                )
                for bit_probability in bitwise_probabilities
            )
        else:
            satisfied = math.isclose(
                probability,
                threshold,
                rel_tol=_COMPARISON_TOLERANCE,
                abs_tol=_COMPARISON_TOLERANCE,
            )
    else:
        raise AbstractPropertyCheckingError(f"Unsupported comparator: {comparator!r}")

    result: dict[str, Any] = {
        "probability": probability,
        "threshold": threshold,
        "judgment": "satisfied" if satisfied else "violated",
    }
    if target_type == "bitwise_measurement_outcome":
        result["bitwise_probabilities"] = bitwise_probabilities
    return result


def evaluate_assertion(
    trace: AbstractExecutionTrace,
    spec: QuantumProgramSpec,
    reconstruction_mode: ReconstructionMode = "trusted",
) -> dict[str, Any]:
    """Dispatch the single v1 assertion to the supported abstract evaluator."""

    validate_program_spec(spec)

    if len(spec.assertions) != 1:
        raise AbstractPropertyCheckingError("Expected exactly one assertion")

    assertion = spec.assertions[0]
    if assertion.kind == "reachability":
        return evaluate_reachability_assertion(
            trace, spec, reconstruction_mode=reconstruction_mode
        )
    if assertion.kind == "probability":
        return evaluate_terminal_probability_assertion(
            trace, spec, reconstruction_mode=reconstruction_mode
        )

    raise AbstractPropertyCheckingError(f"Unsupported assertion kind: {assertion.kind!r}")
