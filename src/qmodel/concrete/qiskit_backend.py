"""Qiskit-based concrete reference backend."""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Sequence
from typing import Any

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace

from qmodel.spec import QuantumProgramSpec
from qmodel.validation import validate_program_spec


class ConcreteBackendError(ValueError):
    """Raised when a concrete circuit cannot be built from a specification."""


_COMPARISON_TOLERANCE = 1e-9
ScopeStateProvider = Callable[[int, Sequence[str]], DensityMatrix]


def _eval_numeric_expression(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id == "pi":
        return math.pi
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_numeric_expression(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
    ):
        left = _eval_numeric_expression(node.left)
        right = _eval_numeric_expression(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        return left**right
    raise ConcreteBackendError("Unsupported numeric expression")


def _resolve_parameter(value: Any) -> float | Parameter:
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isidentifier():
            return Parameter(stripped)
        try:
            expression = ast.parse(stripped, mode="eval")
            return _eval_numeric_expression(expression.body)
        except (SyntaxError, ConcreteBackendError, ZeroDivisionError) as exc:
            raise ConcreteBackendError(
                f"Unsupported parameter expression: {value!r}"
            ) from exc

    raise ConcreteBackendError(f"Unsupported parameter value: {value!r}")


def _apply_gate(circuit: QuantumCircuit, qubit_index: dict[str, int], gate: Any) -> None:
    targets = [qubit_index[name] for name in gate.targets]
    controls = [qubit_index[name] for name in gate.controls]
    name = gate.name

    if name == "I":
        circuit.id(targets[0])
    elif name == "X":
        if controls:
            circuit.mcx(controls, targets[0])
        else:
            circuit.x(targets[0])
    elif name == "Y":
        circuit.y(targets[0])
    elif name == "Z":
        circuit.z(targets[0])
    elif name == "H":
        circuit.h(targets[0])
    elif name == "Ry":
        theta = gate.params.get("theta")
        if theta is None:
            raise ConcreteBackendError("Gate 'Ry' requires params['theta']")
        circuit.ry(_resolve_parameter(theta), targets[0])
    elif name == "S":
        circuit.s(targets[0])
    elif name == "Sdg":
        circuit.sdg(targets[0])
    elif name == "T":
        circuit.t(targets[0])
    elif name == "Tdg":
        circuit.tdg(targets[0])
    elif name == "P":
        theta = gate.params.get("theta")
        if theta is None:
            raise ConcreteBackendError("Gate 'P' requires params['theta']")
        circuit.p(_resolve_parameter(theta), targets[0])
    elif name == "CX":
        circuit.cx(controls[0], targets[0])
    elif name == "CP":
        theta = gate.params.get("theta")
        if theta is None:
            raise ConcreteBackendError("Gate 'CP' requires params['theta']")
        circuit.cp(_resolve_parameter(theta), controls[0], targets[0])
    elif name == "CZ":
        circuit.cz(controls[0], targets[0])
    elif name == "SWAP":
        circuit.swap(targets[0], targets[1])
    elif name == "CCX":
        circuit.ccx(controls[0], controls[1], targets[0])
    elif name == "MCX":
        circuit.mcx(controls, targets[0])
    else:
        raise ConcreteBackendError(f"Unsupported gate for circuit construction: {name!r}")


def build_circuit(spec: QuantumProgramSpec) -> QuantumCircuit:
    """Build a Qiskit circuit that serves as the concrete reference program."""

    validate_program_spec(spec)

    classical_width = 0
    if spec.measurement is not None:
        classical_width = len(spec.measurement.classical_bits or spec.measurement.qubits)
        if spec.measurement.classical_bits is not None and (
            len(spec.measurement.classical_bits) != len(spec.measurement.qubits)
        ):
            raise ConcreteBackendError(
                "measurement.classical_bits must have the same length as measurement.qubits"
            )

    circuit = QuantumCircuit(len(spec.qubits), classical_width, name=spec.program_name)
    qubit_index = {name: index for index, name in enumerate(spec.qubits)}

    for gate in spec.gates:
        _apply_gate(circuit, qubit_index, gate)

    if spec.measurement is not None:
        for classical_index, qubit_name in enumerate(spec.measurement.qubits):
            circuit.measure(qubit_index[qubit_name], classical_index)

    return circuit


def _build_unitary_circuit(spec: QuantumProgramSpec) -> QuantumCircuit:
    circuit = QuantumCircuit(len(spec.qubits), name=spec.program_name)
    qubit_index = {name: index for index, name in enumerate(spec.qubits)}
    for gate in spec.gates:
        _apply_gate(circuit, qubit_index, gate)
    return circuit


def simulate_statevector(spec: QuantumProgramSpec) -> Statevector:
    """Simulate the exact terminal pure state before terminal measurement."""

    validate_program_spec(spec)
    circuit = _build_unitary_circuit(spec)
    if circuit.parameters:
        raise ConcreteBackendError(
            "Cannot simulate an exact statevector with unresolved symbolic parameters"
    )
    return Statevector.from_instruction(circuit)


def simulate_statevector_trajectory(spec: QuantumProgramSpec) -> tuple[Statevector, ...]:
    """Simulate the exact pre-measurement state after each gate occurrence."""

    validate_program_spec(spec)
    trajectory = [Statevector.from_label("0" * len(spec.qubits))]

    for gate in spec.gates:
        step_spec = QuantumProgramSpec(
            program_name=gate.label or gate.name,
            qubits=spec.qubits,
            gates=[
                type(gate)(
                    name=gate.name,
                    targets=gate.targets[:],
                    controls=gate.controls[:],
                    params=dict(gate.params),
                    label=gate.label,
                )
            ],
        )
        step_circuit = _build_unitary_circuit(step_spec)
        if step_circuit.parameters:
            raise ConcreteBackendError(
                "Cannot simulate an exact state trajectory with unresolved symbolic parameters"
            )
        trajectory.append(trajectory[-1].evolve(step_circuit))

    return tuple(trajectory)


def _scope_density_from_statevector(
    state: Statevector, global_qubits: list[str], scope: Sequence[str]
) -> DensityMatrix:
    scope_list = list(scope)
    trace_out = [global_qubits.index(qubit) for qubit in global_qubits if qubit not in scope_list]
    if not trace_out:
        return DensityMatrix(state)
    reduced = partial_trace(state, trace_out)
    if not isinstance(reduced, DensityMatrix):
        raise ConcreteBackendError("Expected partial_trace to return a DensityMatrix")
    return reduced


def build_exact_scope_state_provider(spec: QuantumProgramSpec) -> ScopeStateProvider:
    """Build an exact scope-state oracle from the concrete reference trajectory."""

    validate_program_spec(spec)
    trajectory = simulate_statevector_trajectory(spec)
    qubit_index = {name: index for index, name in enumerate(spec.qubits)}

    def provider(state_index: int, scope: Sequence[str]) -> DensityMatrix:
        if state_index < 0 or state_index >= len(trajectory):
            raise ConcreteBackendError(f"state_index out of range: {state_index}")

        scope_list = list(scope)
        unknown_scope = [name for name in scope_list if name not in qubit_index]
        if unknown_scope:
            raise ConcreteBackendError(f"Unknown qubits in requested scope: {unknown_scope}")

        return _scope_density_from_statevector(trajectory[state_index], spec.qubits, scope_list)

    return provider


def _basis_bitstring(index: int, width: int) -> str:
    return format(index, f"0{width}b")[::-1]


def _basis_state_projector(width: int, state: str) -> DensityMatrix:
    normalized_state = state.strip()
    if len(normalized_state) != width or any(bit not in "01" for bit in normalized_state):
        raise ConcreteBackendError(
            f"Invalid basis state {state!r} for scope width {width}"
        )

    basis_vector = [0j] * (2**width)
    basis_vector[int(normalized_state[::-1], 2)] = 1.0
    return DensityMatrix(
        [[basis_vector[i] * complex(basis_vector[j]).conjugate() for j in range(2**width)] for i in range(2**width)]
    )


def _projector_overlap(witness: DensityMatrix, projector: DensityMatrix) -> float:
    overlap = (projector.data @ witness.data).trace()
    return float(overlap.real)


def measurement_outcome_probability(
    spec: QuantumProgramSpec, scope: list[str], outcomes: list[str]
) -> float:
    """Compute the exact probability mass of terminal computational outcomes on a scope."""

    if not outcomes:
        raise ConcreteBackendError("measurement outcome set must be non-empty")

    state = simulate_statevector(spec)
    qubit_index = {name: index for index, name in enumerate(spec.qubits)}

    unknown_scope = [name for name in scope if name not in qubit_index]
    if unknown_scope:
        raise ConcreteBackendError(f"Unknown qubits in assertion scope: {unknown_scope}")

    outcome_width = len(scope)
    normalized_outcomes = {outcome.strip() for outcome in outcomes}
    for outcome in normalized_outcomes:
        if len(outcome) != outcome_width or any(bit not in "01" for bit in outcome):
            raise ConcreteBackendError(
                f"Invalid measurement outcome {outcome!r} for scope width {outcome_width}"
            )

    probability = 0.0
    for basis_index, amplitude in enumerate(state.data):
        basis_probability = float(abs(amplitude) ** 2)
        if basis_probability == 0.0:
            continue
        full_bits = _basis_bitstring(basis_index, len(spec.qubits))
        scoped_bits = "".join(full_bits[qubit_index[name]] for name in scope)
        if scoped_bits in normalized_outcomes:
            probability += basis_probability

    return probability


def _measurement_outcome_probability_from_statevector(
    state: Statevector,
    spec: QuantumProgramSpec,
    scope: list[str],
    outcomes: list[str],
) -> float:
    if not outcomes:
        raise ConcreteBackendError("measurement outcome set must be non-empty")

    qubit_index = {name: index for index, name in enumerate(spec.qubits)}
    unknown_scope = [name for name in scope if name not in qubit_index]
    if unknown_scope:
        raise ConcreteBackendError(f"Unknown qubits in assertion scope: {unknown_scope}")

    outcome_width = len(scope)
    normalized_outcomes = {outcome.strip() for outcome in outcomes}
    for outcome in normalized_outcomes:
        if len(outcome) != outcome_width or any(bit not in "01" for bit in outcome):
            raise ConcreteBackendError(
                f"Invalid measurement outcome {outcome!r} for scope width {outcome_width}"
            )

    probability = 0.0
    for basis_index, amplitude in enumerate(state.data):
        basis_probability = float(abs(amplitude) ** 2)
        if basis_probability == 0.0:
            continue
        full_bits = _basis_bitstring(basis_index, len(spec.qubits))
        scoped_bits = "".join(full_bits[qubit_index[name]] for name in scope)
        if scoped_bits in normalized_outcomes:
            probability += basis_probability

    return probability


def _single_bit_probability_from_statevector(
    state: Statevector,
    qubit_index: int,
    expected_bit: str,
) -> float:
    normalized_bit = expected_bit.strip()
    if normalized_bit not in {"0", "1"}:
        raise ConcreteBackendError(f"Invalid single-bit outcome {expected_bit!r}")

    probability = 0.0
    for basis_index, amplitude in enumerate(state.data):
        basis_probability = float(abs(amplitude) ** 2)
        if basis_probability == 0.0:
            continue
        full_bits = _basis_bitstring(basis_index, state.num_qubits)
        if full_bits[qubit_index] == normalized_bit:
            probability += basis_probability
    return probability


def bitwise_measurement_outcome_probability(
    spec: QuantumProgramSpec, scope: list[str], outcome: str
) -> list[float]:
    """Compute the per-qubit probability mass for a bitwise terminal assertion."""

    state = simulate_statevector(spec)
    qubit_index = {name: index for index, name in enumerate(spec.qubits)}

    unknown_scope = [name for name in scope if name not in qubit_index]
    if unknown_scope:
        raise ConcreteBackendError(f"Unknown qubits in assertion scope: {unknown_scope}")

    normalized_outcome = outcome.strip()
    if len(normalized_outcome) != len(scope) or any(bit not in "01" for bit in normalized_outcome):
        raise ConcreteBackendError(
            f"Invalid bitwise measurement outcome {outcome!r} for scope width {len(scope)}"
        )

    return [
        _single_bit_probability_from_statevector(state, qubit_index[qubit_name], expected_bit)
        for qubit_name, expected_bit in zip(scope, normalized_outcome, strict=True)
    ]


def _bitwise_measurement_outcome_probability_from_statevector(
    state: Statevector,
    spec: QuantumProgramSpec,
    scope: list[str],
    outcome: str,
) -> list[float]:
    qubit_index = {name: index for index, name in enumerate(spec.qubits)}

    unknown_scope = [name for name in scope if name not in qubit_index]
    if unknown_scope:
        raise ConcreteBackendError(f"Unknown qubits in assertion scope: {unknown_scope}")

    normalized_outcome = outcome.strip()
    if len(normalized_outcome) != len(scope) or any(bit not in "01" for bit in normalized_outcome):
        raise ConcreteBackendError(
            f"Invalid bitwise measurement outcome {outcome!r} for scope width {len(scope)}"
        )

    return [
        _single_bit_probability_from_statevector(state, qubit_index[qubit_name], expected_bit)
        for qubit_name, expected_bit in zip(scope, normalized_outcome, strict=True)
    ]


def evaluate_reachability_assertion(spec: QuantumProgramSpec) -> dict[str, float | int | str | None]:
    """Evaluate the single reachability assertion stored in the specification."""

    validate_program_spec(spec)

    if len(spec.assertions) != 1:
        raise ConcreteBackendError("Reachability evaluation expects exactly one assertion")

    assertion = spec.assertions[0]
    if assertion.kind != "reachability":
        raise ConcreteBackendError("Only reachability assertions are supported in this stage")
    if assertion.target.get("type") != "basis_state":
        raise ConcreteBackendError("Only basis_state reachability targets are supported in this stage")

    scope = assertion.target["scope"]
    state = assertion.target.get("state")
    if not isinstance(state, str) or not state.strip():
        raise ConcreteBackendError("Reachability target.state must be a non-empty bitstring")

    projector = _basis_state_projector(len(scope), state)
    trajectory = simulate_statevector_trajectory(spec)
    max_overlap = 0.0
    first_reached_index: int | None = None
    for index, current in enumerate(trajectory):
        witness = _scope_density_from_statevector(current, spec.qubits, scope)
        overlap = _projector_overlap(witness, projector)
        max_overlap = max(max_overlap, overlap)
        if overlap > _COMPARISON_TOLERANCE and first_reached_index is None:
            first_reached_index = index

    return {
        "max_overlap": max_overlap,
        "first_reached_index": first_reached_index,
        "judgment": "satisfied" if first_reached_index is not None else "violated",
    }


def _evaluate_probability_assertion_from_statevector(
    spec: QuantumProgramSpec,
    state: Statevector,
) -> dict[str, float | str]:
    if len(spec.assertions) != 1:
        raise ConcreteBackendError("Probability evaluation expects exactly one assertion")

    assertion = spec.assertions[0]
    if assertion.kind != "probability":
        raise ConcreteBackendError("Only probability assertions are supported in this stage")

    target_type = assertion.target.get("type")
    if target_type not in {"measurement_outcome", "bitwise_measurement_outcome"}:
        raise ConcreteBackendError(
            "Only measurement_outcome and bitwise_measurement_outcome targets are supported in this stage"
        )

    scope = assertion.target["scope"]
    comparator = assertion.comparator
    threshold = assertion.threshold
    threshold_value = float(threshold)

    if target_type == "measurement_outcome":
        outcomes = assertion.target.get("outcomes")
        if not isinstance(outcomes, list) or not all(isinstance(item, str) for item in outcomes):
            raise ConcreteBackendError(
                "measurement_outcome probability assertions require a string-list 'outcomes' field"
            )
        probability = _measurement_outcome_probability_from_statevector(state, spec, scope, outcomes)
        bitwise_probabilities = None
    else:
        outcome = assertion.target.get("outcome")
        if not isinstance(outcome, str) or not outcome.strip():
            raise ConcreteBackendError(
                "bitwise_measurement_outcome probability assertions require a non-empty 'outcome' field"
            )
        bitwise_probabilities = _bitwise_measurement_outcome_probability_from_statevector(
            state, spec, scope, outcome
        )
        probability = min(bitwise_probabilities) if comparator != "<=" else max(bitwise_probabilities)

    if comparator == ">=":
        if bitwise_probabilities is None:
            satisfied = probability >= threshold_value - _COMPARISON_TOLERANCE
        else:
            satisfied = all(
                bit_probability >= threshold_value - _COMPARISON_TOLERANCE
                for bit_probability in bitwise_probabilities
            )
    elif comparator == "<=":
        if bitwise_probabilities is None:
            satisfied = probability <= threshold_value + _COMPARISON_TOLERANCE
        else:
            satisfied = all(
                bit_probability <= threshold_value + _COMPARISON_TOLERANCE
                for bit_probability in bitwise_probabilities
            )
    elif comparator == "=":
        if bitwise_probabilities is None:
            satisfied = math.isclose(
                probability,
                threshold_value,
                rel_tol=_COMPARISON_TOLERANCE,
                abs_tol=_COMPARISON_TOLERANCE,
            )
        else:
            satisfied = all(
                math.isclose(
                    bit_probability,
                    threshold_value,
                    rel_tol=_COMPARISON_TOLERANCE,
                    abs_tol=_COMPARISON_TOLERANCE,
                )
                for bit_probability in bitwise_probabilities
            )
    else:
        raise ConcreteBackendError(f"Unsupported comparator: {comparator!r}")

    result: dict[str, Any] = {
        "probability": probability,
        "threshold": threshold_value,
        "judgment": "satisfied" if satisfied else "violated",
    }
    if bitwise_probabilities is not None:
        result["bitwise_probabilities"] = bitwise_probabilities
    return result


def evaluate_probability_assertion(spec: QuantumProgramSpec) -> dict[str, float | str]:
    """Evaluate the single terminal probability assertion stored in the specification."""

    validate_program_spec(spec)
    state = simulate_statevector(spec)
    return _evaluate_probability_assertion_from_statevector(spec, state)


def evaluate_assertion(spec: QuantumProgramSpec) -> dict[str, Any]:
    """Dispatch the single v1 assertion to the supported concrete evaluator."""

    validate_program_spec(spec)

    if len(spec.assertions) != 1:
        raise ConcreteBackendError("Expected exactly one assertion")

    assertion = spec.assertions[0]
    if assertion.kind == "probability":
        return evaluate_probability_assertion(spec)
    if assertion.kind == "reachability":
        return evaluate_reachability_assertion(spec)
    raise ConcreteBackendError(f"Unsupported assertion kind: {assertion.kind!r}")
