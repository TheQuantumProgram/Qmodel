"""Validation helpers for QuantumProgramSpec."""

from __future__ import annotations

from .spec import QuantumProgramSpec


class SpecValidationError(ValueError):
    """Raised when a QuantumProgramSpec violates project constraints."""


_ALLOWED_INITIAL_STATES = {"zero"}
_ALLOWED_MEASUREMENT_BASES = {"computational"}
_ALLOWED_ASSERTION_KINDS = {"reachability", "probability"}
_ALLOWED_COMPARATORS = {">=", "<=", "="}
_SUPPORTED_GATES = {
    "I",
    "X",
    "Y",
    "Z",
    "H",
    "Ry",
    "S",
    "Sdg",
    "CX",
    "CP",
    "CZ",
    "SWAP",
    "T",
    "Tdg",
    "P",
    "CCX",
    "MCX",
}


def _ensure_known_qubits(names: list[str], known: set[str], context: str) -> None:
    unknown = [name for name in names if name not in known]
    if unknown:
        raise SpecValidationError(f"Unknown qubits in {context}: {unknown}")


def _ensure_gate_shape(name: str, controls: list[str], targets: list[str], context: str) -> None:
    if set(controls) & set(targets):
        raise SpecValidationError(f"{context} must not overlap controls and targets")

    if name in {"I", "Y", "Z", "H", "Ry", "S", "Sdg", "T", "Tdg", "P"}:
        if len(targets) != 1 or controls:
            raise SpecValidationError(
                f"{context} for gate {name!r} must have exactly 1 target and 0 controls"
            )
        return

    if name == "X":
        if len(targets) != 1:
            raise SpecValidationError(f"{context} for gate 'X' must have exactly 1 target")
        return

    if name in {"CX", "CP", "CZ"}:
        if len(targets) != 1 or len(controls) != 1:
            raise SpecValidationError(
                f"{context} for gate {name!r} must have exactly 1 target and 1 control"
            )
        return

    if name == "SWAP":
        if len(targets) != 2 or controls:
            raise SpecValidationError(
                f"{context} for gate 'SWAP' must have exactly 2 targets and 0 controls"
            )
        return

    if name == "CCX":
        if len(targets) != 1 or len(controls) != 2:
            raise SpecValidationError(
                f"{context} for gate 'CCX' must have exactly 1 target and 2 controls"
            )
        return

    if name == "MCX":
        if len(targets) != 1 or len(controls) < 1:
            raise SpecValidationError(
                f"{context} for gate 'MCX' must have exactly 1 target and at least 1 control"
            )
        return


def _validate_assertion_structure(kind: str, target: dict[str, object], context: str) -> None:
    if kind not in _ALLOWED_ASSERTION_KINDS:
        raise SpecValidationError(
            f"{context} kind must be one of {sorted(_ALLOWED_ASSERTION_KINDS)}, got {kind!r}"
        )

    target_type = target.get("type")
    if not isinstance(target_type, str) or not target_type.strip():
        raise SpecValidationError(f"{context} target.type must be a non-empty string")

    scope = target.get("scope")
    if not isinstance(scope, list) or not scope or not all(isinstance(item, str) for item in scope):
        raise SpecValidationError(f"{context} target.scope must be a non-empty list of strings")

    if kind == "reachability":
        if target_type != "basis_state":
            raise SpecValidationError(
                f"{context} reachability target.type must be 'basis_state' in v1"
            )
        state = target.get("state")
        if not isinstance(state, str) or not state.strip():
            raise SpecValidationError(
                f"{context} basis_state reachability targets must define a non-empty state"
            )
        if len(state.strip()) != len(scope) or any(bit not in "01" for bit in state.strip()):
            raise SpecValidationError(
                f"{context} basis_state reachability target.state must be a bitstring matching scope width"
            )
        return

    if kind == "probability" and target_type not in {
        "measurement_outcome",
        "bitwise_measurement_outcome",
    }:
        raise SpecValidationError(
            f"{context} probability target.type must be one of "
            "{'measurement_outcome', 'bitwise_measurement_outcome'} in v1"
        )
    if kind == "probability" and target_type == "measurement_outcome":
        outcomes = target.get("outcomes")
        if "outcome" in target:
            raise SpecValidationError(
                f"{context} measurement_outcome targets must not define outcome"
            )
        if not isinstance(outcomes, list) or not outcomes or not all(
            isinstance(item, str) for item in outcomes
        ):
            raise SpecValidationError(
                f"{context} probability targets must define a non-empty string-list outcomes field"
            )
        for outcome in outcomes:
            normalized = outcome.strip()
            if len(normalized) != len(scope) or any(bit not in "01" for bit in normalized):
                raise SpecValidationError(
                    f"{context} probability target outcomes must be bitstrings matching scope width"
                )
    if kind == "probability" and target_type == "bitwise_measurement_outcome":
        if "outcomes" in target:
            raise SpecValidationError(
                f"{context} bitwise_measurement_outcome targets must not define outcomes"
            )
        outcome = target.get("outcome")
        if not isinstance(outcome, str) or not outcome.strip():
            raise SpecValidationError(
                f"{context} bitwise_measurement_outcome targets must define a non-empty outcome"
            )
        normalized = outcome.strip()
        if len(normalized) != len(scope) or any(bit not in "01" for bit in normalized):
            raise SpecValidationError(
                f"{context} probability target outcome must be a bitstring matching scope width"
            )


def _validate_units(
    unit_specs: list,
    *,
    known_qubits: set[str],
    context: str,
) -> None:
    for index, unit in enumerate(unit_specs):
        if not unit.qubits:
            raise SpecValidationError(f"{context}[{index}] must contain at least one qubit")
        _ensure_known_qubits(unit.qubits, known_qubits, f"{context}[{index}].qubits")


def _validate_organization_schedule(
    spec: QuantumProgramSpec,
    known_qubits: set[str],
) -> None:
    schedule = spec.organization_schedule
    if schedule is None:
        return

    if not schedule.initial_state.strip():
        raise SpecValidationError("organization_schedule.initial_state must be non-empty")
    if not schedule.states:
        raise SpecValidationError("organization_schedule.states must be non-empty")

    states_by_name = {}
    for index, state in enumerate(schedule.states):
        if not state.name.strip():
            raise SpecValidationError(
                f"organization_schedule.states[{index}].name must be non-empty"
            )
        if state.name in states_by_name:
            raise SpecValidationError(
                f"organization_schedule state names must be unique, got duplicate {state.name!r}"
            )
        _validate_units(
            state.units,
            known_qubits=known_qubits,
            context=f"organization_schedule.states[{index}].units",
        )
        states_by_name[state.name] = state

    if schedule.initial_state not in states_by_name:
        raise SpecValidationError(
            "organization_schedule.initial_state must reference a defined state"
        )

    current_name = schedule.initial_state
    visited_states: set[str] = set()
    for gate_index in range(len(spec.gates)):
        if current_name in visited_states:
            raise SpecValidationError(
                "organization_schedule must form a linear acyclic state chain"
            )
        visited_states.add(current_name)
        current_state = states_by_name[current_name]
        transition = current_state.transition
        if transition is None:
            raise SpecValidationError(
                "organization_schedule must define one transition for each gate step"
            )
        if transition.gate_index != gate_index:
            raise SpecValidationError(
                "organization_schedule transitions must cover gates in sequential order"
            )
        if transition.next_state not in states_by_name:
            raise SpecValidationError(
                "organization_schedule transition next_state must reference a defined state"
            )
        current_name = transition.next_state

    terminal_state = states_by_name[current_name]
    if terminal_state.transition is not None:
        raise SpecValidationError(
            "organization_schedule terminal state must not define a transition"
        )
    visited_states.add(current_name)
    if visited_states != set(states_by_name):
        raise SpecValidationError(
            "organization_schedule must be one reachable linear chain with no extra states"
        )


def validate_program_spec(spec: QuantumProgramSpec) -> None:
    """Validate core structural constraints of a quantum program specification."""

    if not spec.program_name.strip():
        raise SpecValidationError("program_name must be non-empty")

    if not spec.qubits:
        raise SpecValidationError("qubits must be non-empty")

    if len(set(spec.qubits)) != len(spec.qubits):
        raise SpecValidationError("qubits must be unique and ordered")

    if spec.initial_state not in _ALLOWED_INITIAL_STATES:
        raise SpecValidationError(
            f"initial_state must be one of {_ALLOWED_INITIAL_STATES}, got {spec.initial_state!r}"
        )

    known_qubits = set(spec.qubits)

    for index, gate in enumerate(spec.gates):
        if not gate.name.strip():
            raise SpecValidationError(f"gate[{index}] must have a non-empty name")
        if gate.name not in _SUPPORTED_GATES:
            raise SpecValidationError(
                f"Unsupported gate in gate[{index}]: {gate.name!r}. "
                f"Supported gates are {sorted(_SUPPORTED_GATES)}"
            )
        if not gate.targets:
            raise SpecValidationError(f"gate[{index}] must have at least one target")
        _ensure_gate_shape(gate.name, gate.controls, gate.targets, f"gate[{index}]")
        _ensure_known_qubits(gate.targets, known_qubits, f"gate[{index}].targets")
        _ensure_known_qubits(gate.controls, known_qubits, f"gate[{index}].controls")

    if spec.measurement is not None:
        if spec.measurement.basis not in _ALLOWED_MEASUREMENT_BASES:
            raise SpecValidationError(
                "measurement.basis must be one of "
                f"{sorted(_ALLOWED_MEASUREMENT_BASES)}, got {spec.measurement.basis!r}"
            )
        _ensure_known_qubits(spec.measurement.qubits, known_qubits, "measurement.qubits")

    _validate_units(spec.units, known_qubits=known_qubits, context="unit")
    _validate_organization_schedule(spec, known_qubits)

    for index, assertion in enumerate(spec.assertions):
        if not assertion.kind.strip():
            raise SpecValidationError(f"assertion[{index}] must have a non-empty kind")
        if not isinstance(assertion.target, dict) or not assertion.target:
            raise SpecValidationError(f"assertion[{index}] must have a non-empty target mapping")
        _validate_assertion_structure(assertion.kind, assertion.target, f"assertion[{index}]")
        _ensure_known_qubits(assertion.target["scope"], known_qubits, f"assertion[{index}].target.scope")
        if assertion.kind == "probability":
            if assertion.comparator is None:
                raise SpecValidationError(
                    f"assertion[{index}] probability assertions must define comparator"
                )
            if assertion.comparator not in _ALLOWED_COMPARATORS:
                raise SpecValidationError(
                    f"assertion[{index}] probability comparator must be one of {sorted(_ALLOWED_COMPARATORS)}"
                )
            if assertion.threshold is None:
                raise SpecValidationError(
                    f"assertion[{index}] probability assertions must define threshold"
                )
