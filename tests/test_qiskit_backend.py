import unittest
import math

from qmodel.concrete.qiskit_backend import (
    build_exact_scope_state_provider,
    build_circuit,
    evaluate_assertion,
    evaluate_reachability_assertion,
    evaluate_probability_assertion,
    measurement_outcome_probability,
    simulate_statevector,
    simulate_statevector_trajectory,
)
from qmodel.spec import AssertionSpec, GateSpec, MeasurementSpec, QuantumProgramSpec


class QiskitBackendTests(unittest.TestCase):
    def test_build_circuit_preserves_gate_order_and_measurement(self) -> None:
        spec = QuantumProgramSpec(
            program_name="bell",
            qubits=["q0", "q1"],
            gates=[
                GateSpec(name="H", targets=["q0"]),
                GateSpec(name="CX", controls=["q0"], targets=["q1"]),
            ],
            measurement=MeasurementSpec(qubits=["q0", "q1"]),
        )

        circuit = build_circuit(spec)

        self.assertEqual(circuit.name, "bell")
        self.assertEqual(circuit.num_qubits, 2)
        self.assertEqual(circuit.num_clbits, 2)
        self.assertEqual(
            [instruction.operation.name for instruction in circuit.data],
            ["h", "cx", "measure", "measure"],
        )

    def test_build_circuit_supports_symbolic_phase_parameter(self) -> None:
        spec = QuantumProgramSpec(
            program_name="phase_param",
            qubits=["q0"],
            gates=[GateSpec(name="P", targets=["q0"], params={"theta": "theta"})],
        )

        circuit = build_circuit(spec)

        self.assertEqual(circuit.data[0].operation.name, "p")
        self.assertEqual(str(circuit.data[0].operation.params[0]), "theta")

    def test_build_circuit_supports_ry_rotation(self) -> None:
        spec = QuantumProgramSpec(
            program_name="ry_param",
            qubits=["q0"],
            gates=[GateSpec(name="Ry", targets=["q0"], params={"theta": math.pi / 3})],
        )

        circuit = build_circuit(spec)

        self.assertEqual(circuit.data[0].operation.name, "ry")
        self.assertAlmostEqual(float(circuit.data[0].operation.params[0]), math.pi / 3)

    def test_build_circuit_supports_controlled_phase(self) -> None:
        spec = QuantumProgramSpec(
            program_name="controlled_phase",
            qubits=["q0", "q1"],
            gates=[
                GateSpec(
                    name="CP",
                    controls=["q0"],
                    targets=["q1"],
                    params={"theta": math.pi / 2},
                )
            ],
        )

        circuit = build_circuit(spec)

        self.assertEqual(circuit.data[0].operation.name, "cp")
        self.assertAlmostEqual(float(circuit.data[0].operation.params[0]), math.pi / 2)

    def test_build_circuit_supports_ccx_and_mcx(self) -> None:
        spec = QuantumProgramSpec(
            program_name="multi_control",
            qubits=["q0", "q1", "q2", "q3"],
            gates=[
                GateSpec(name="CCX", controls=["q0", "q1"], targets=["q2"]),
                GateSpec(name="MCX", controls=["q0", "q1", "q2"], targets=["q3"]),
            ],
        )

        circuit = build_circuit(spec)

        self.assertEqual(circuit.data[0].operation.name, "ccx")
        self.assertEqual(circuit.data[1].operation.name, "mcx")

    def test_build_circuit_supports_controlled_x_spelled_as_x(self) -> None:
        spec = QuantumProgramSpec(
            program_name="controlled_x_alias",
            qubits=["q0", "q1"],
            gates=[GateSpec(name="X", controls=["q0"], targets=["q1"])],
        )

        circuit = build_circuit(spec)

        self.assertEqual(circuit.data[0].operation.name, "cx")

    def test_simulate_statevector_returns_exact_terminal_state(self) -> None:
        spec = QuantumProgramSpec(
            program_name="one_qubit_h",
            qubits=["q0"],
            gates=[GateSpec(name="H", targets=["q0"])],
        )

        state = simulate_statevector(spec)
        probabilities = [float(value) for value in state.probabilities()]

        self.assertAlmostEqual(probabilities[0], 0.5)
        self.assertAlmostEqual(probabilities[1], 0.5)

    def test_simulate_statevector_supports_ry_probability_tuning(self) -> None:
        spec = QuantumProgramSpec(
            program_name="one_qubit_ry",
            qubits=["q0"],
            gates=[GateSpec(name="Ry", targets=["q0"], params={"theta": math.pi / 3})],
        )

        state = simulate_statevector(spec)
        probabilities = [float(value) for value in state.probabilities()]

        self.assertAlmostEqual(probabilities[0], 0.75, places=9)
        self.assertAlmostEqual(probabilities[1], 0.25, places=9)

    def test_simulate_statevector_supports_controlled_phase(self) -> None:
        spec = QuantumProgramSpec(
            program_name="controlled_phase_effect",
            qubits=["q0", "q1"],
            gates=[
                GateSpec(name="H", targets=["q0"]),
                GateSpec(name="H", targets=["q1"]),
                GateSpec(
                    name="CP",
                    controls=["q0"],
                    targets=["q1"],
                    params={"theta": math.pi},
                ),
            ],
        )

        state = simulate_statevector(spec)

        self.assertAlmostEqual(abs(state.data[0]), 0.5)
        self.assertAlmostEqual(abs(state.data[1]), 0.5)
        self.assertAlmostEqual(abs(state.data[2]), 0.5)
        self.assertAlmostEqual(abs(state.data[3]), 0.5)
        self.assertAlmostEqual(float(state.data[3].real), -0.5)
        self.assertAlmostEqual(float(state.data[3].imag), 0.0, places=9)

    def test_exact_scope_state_provider_returns_reduced_state_from_requested_step(self) -> None:
        spec = QuantumProgramSpec(
            program_name="one_qubit_h",
            qubits=["q0"],
            gates=[GateSpec(name="H", targets=["q0"])],
        )

        trajectory = simulate_statevector_trajectory(spec)
        provider = build_exact_scope_state_provider(spec)

        self.assertEqual(len(trajectory), 2)
        initial = provider(0, ["q0"])
        final = provider(1, ["q0"])

        self.assertAlmostEqual(float(initial.probabilities()[0]), 1.0)
        self.assertAlmostEqual(float(final.probabilities()[0]), 0.5)
        self.assertAlmostEqual(float(final.probabilities()[1]), 0.5)

    def test_measurement_outcome_probability_uses_scope_order(self) -> None:
        spec = QuantumProgramSpec(
            program_name="subset_measurement",
            qubits=["q0", "q1", "q2"],
            gates=[GateSpec(name="X", targets=["q2"])],
            measurement=MeasurementSpec(qubits=["q0", "q2"]),
        )

        probability = measurement_outcome_probability(spec, ["q0", "q2"], ["01"])

        self.assertAlmostEqual(probability, 1.0)

    def test_evaluate_probability_assertion_returns_probability_and_judgment(self) -> None:
        spec = QuantumProgramSpec(
            program_name="bell_assertion",
            qubits=["q0", "q1"],
            gates=[
                GateSpec(name="H", targets=["q0"]),
                GateSpec(name="CX", controls=["q0"], targets=["q1"]),
            ],
            measurement=MeasurementSpec(qubits=["q0", "q1"]),
            assertions=[
                AssertionSpec(
                    name="bell_mass",
                    kind="probability",
                    target={
                        "type": "measurement_outcome",
                        "scope": ["q0", "q1"],
                        "outcomes": ["00", "11"],
                    },
                    comparator=">=",
                    threshold=1.0,
                )
            ],
        )

        result = evaluate_probability_assertion(spec)

        self.assertAlmostEqual(result["probability"], 1.0)
        self.assertEqual(result["judgment"], "satisfied")

    def test_evaluate_bitwise_probability_assertion_returns_probability_and_judgment(self) -> None:
        spec = QuantumProgramSpec(
            program_name="bitwise_terminal_check",
            qubits=["q0", "q1"],
            gates=[GateSpec(name="X", targets=["q1"])],
            assertions=[
                AssertionSpec(
                    name="bitwise_mass",
                    kind="probability",
                    target={
                        "type": "bitwise_measurement_outcome",
                        "scope": ["q0", "q1"],
                        "outcome": "01",
                    },
                    comparator="=",
                    threshold=1.0,
                )
            ],
        )

        result = evaluate_probability_assertion(spec)

        self.assertAlmostEqual(result["probability"], 1.0)
        self.assertEqual(result["judgment"], "satisfied")

    def test_evaluate_assertion_dispatches_bitwise_probability_assertion(self) -> None:
        spec = QuantumProgramSpec(
            program_name="bitwise_terminal_check_dispatch",
            qubits=["q0", "q1"],
            gates=[GateSpec(name="X", targets=["q1"])],
            assertions=[
                AssertionSpec(
                    kind="probability",
                    target={
                        "type": "bitwise_measurement_outcome",
                        "scope": ["q0", "q1"],
                        "outcome": "01",
                    },
                    comparator="=",
                    threshold=1.0,
                )
            ],
        )

        self.assertEqual(evaluate_assertion(spec), evaluate_probability_assertion(spec))

    def test_evaluate_reachability_assertion_returns_overlap_and_first_index(self) -> None:
        spec = QuantumProgramSpec(
            program_name="single_h_reach",
            qubits=["q0"],
            gates=[GateSpec(name="H", targets=["q0"])],
            assertions=[
                AssertionSpec(
                    kind="reachability",
                    target={"type": "basis_state", "scope": ["q0"], "state": "1"},
                )
            ],
        )

        result = evaluate_reachability_assertion(spec)

        self.assertAlmostEqual(result["max_overlap"], 0.5)
        self.assertEqual(result["first_reached_index"], 1)
        self.assertEqual(result["judgment"], "satisfied")

    def test_evaluate_assertion_dispatches_reachability(self) -> None:
        spec = QuantumProgramSpec(
            program_name="single_h_reach",
            qubits=["q0"],
            gates=[GateSpec(name="H", targets=["q0"])],
            assertions=[
                AssertionSpec(
                    kind="reachability",
                    target={"type": "basis_state", "scope": ["q0"], "state": "1"},
                )
            ],
        )

        self.assertEqual(evaluate_assertion(spec), evaluate_reachability_assertion(spec))


if __name__ == "__main__":
    unittest.main()
