from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from qmodel.abstract.property_checking import (
    AbstractPropertyCheckingError,
    evaluate_assertion,
    evaluate_reachability_assertion,
    evaluate_terminal_probability_assertion,
    final_scope_witness,
)
from qmodel.abstract.state import AbstractState, abstract_local_state
from qmodel.abstract.transition import AbstractExecutionTrace, build_abstract_trace, reconstruct_scope_state
from qmodel.spec import AssertionSpec, GateSpec, QuantumProgramSpec, UnitSpec
from qmodel.concrete.qiskit_backend import measurement_outcome_probability
from qmodel.parser.qmodel_parser import parse_qmodel_file
from qiskit.quantum_info import DensityMatrix


_MODELS_DIR = Path(__file__).resolve().parent / "models"


class AbstractPropertyCheckingTests(unittest.TestCase):
    def test_final_scope_witness_uses_single_final_view_when_available(self) -> None:
        spec = parse_qmodel_file(str(_MODELS_DIR / "clifford_bell.qmodel"))
        trace = build_abstract_trace(spec)

        witness = final_scope_witness(trace, spec, ["q0", "q1"])

        expected = trace.states[-1].units[0].witness_rho.data
        np.testing.assert_allclose(witness.data, expected)

    def test_terminal_probability_assertion_matches_concrete_backend_for_single_view(self) -> None:
        spec = parse_qmodel_file(str(_MODELS_DIR / "clifford_bell.qmodel"))
        trace = build_abstract_trace(spec)

        result = evaluate_terminal_probability_assertion(trace, spec)
        expected_probability = measurement_outcome_probability(
            spec,
            spec.assertions[0].target["scope"],
            spec.assertions[0].target["outcomes"],
        )

        self.assertAlmostEqual(result["probability"], expected_probability)
        self.assertEqual(result["judgment"], "satisfied")

    def test_terminal_probability_assertion_matches_concrete_backend_for_multi_view_scope(self) -> None:
        spec = parse_qmodel_file(str(_MODELS_DIR / "clifford_gate_showcase.qmodel"))
        trace = build_abstract_trace(spec)

        result = evaluate_terminal_probability_assertion(trace, spec)
        expected_probability = measurement_outcome_probability(
            spec,
            spec.assertions[0].target["scope"],
            spec.assertions[0].target["outcomes"],
        )

        self.assertAlmostEqual(result["probability"], expected_probability)
        self.assertEqual(result["judgment"], "satisfied")

    def test_bitwise_probability_assertion_matches_single_qubit_witnesses(self) -> None:
        spec = QuantumProgramSpec(
            program_name="bitwise_terminal_check",
            qubits=["q0", "q1"],
            gates=[],
            units=[
                UnitSpec(name="u0", qubits=["q0"]),
                UnitSpec(name="u1", qubits=["q1"]),
            ],
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
        trace = AbstractExecutionTrace(
            states=(
                AbstractState(
                    units=(
                        abstract_local_state(DensityMatrix.from_label("0"), ["q0"], name="u0"),
                        abstract_local_state(DensityMatrix.from_label("1"), ["q1"], name="u1"),
                    ),
                    position=0,
                ),
            ),
            transitions=(),
        )

        result = evaluate_terminal_probability_assertion(trace, spec)

        self.assertAlmostEqual(result["probability"], 1.0)
        self.assertEqual(result["judgment"], "satisfied")

    def test_terminal_probability_assertion_checked_mode_accepts_consistent_overlap_join(self) -> None:
        spec = QuantumProgramSpec(
            program_name="overlap_terminal_check",
            qubits=["q0", "q1", "q2"],
            gates=[],
            units=[
                UnitSpec(name="u1", qubits=["q0", "q1"]),
                UnitSpec(name="u2", qubits=["q1", "q2"]),
            ],
            assertions=[
                AssertionSpec(
                    kind="probability",
                    target={
                        "type": "measurement_outcome",
                        "scope": ["q0", "q1", "q2"],
                        "outcomes": ["000"],
                    },
                    comparator=">=",
                    threshold=1.0,
                )
            ],
        )
        trace = AbstractExecutionTrace(
            states=(
                AbstractState(
                    units=(
                        abstract_local_state(DensityMatrix.from_label("00"), ["q0", "q1"], name="u1"),
                        abstract_local_state(DensityMatrix.from_label("00"), ["q1", "q2"], name="u2"),
                    ),
                    position=0,
                ),
            ),
            transitions=(),
        )

        result = evaluate_terminal_probability_assertion(trace, spec, reconstruction_mode="checked")

        self.assertAlmostEqual(result["probability"], 1.0)
        self.assertEqual(result["judgment"], "satisfied")

    def test_evaluate_assertion_dispatches_probability_assertion(self) -> None:
        spec = parse_qmodel_file(str(_MODELS_DIR / "clifford_bell.qmodel"))
        trace = build_abstract_trace(spec)

        dispatched = evaluate_assertion(trace, spec)
        direct = evaluate_terminal_probability_assertion(trace, spec)

        self.assertEqual(dispatched, direct)

    def test_evaluate_assertion_dispatches_bitwise_probability_assertion(self) -> None:
        spec = QuantumProgramSpec(
            program_name="bitwise_terminal_check_dispatch",
            qubits=["q0", "q1"],
            gates=[],
            units=[
                UnitSpec(name="u0", qubits=["q0"]),
                UnitSpec(name="u1", qubits=["q1"]),
            ],
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
        trace = AbstractExecutionTrace(
            states=(
                AbstractState(
                    units=(
                        abstract_local_state(DensityMatrix.from_label("0"), ["q0"], name="u0"),
                        abstract_local_state(DensityMatrix.from_label("1"), ["q1"], name="u1"),
                    ),
                    position=0,
                ),
            ),
            transitions=(),
        )

        dispatched = evaluate_assertion(trace, spec)
        direct = evaluate_terminal_probability_assertion(trace, spec)

        self.assertEqual(dispatched, direct)

    def test_evaluate_assertion_dispatches_reachability_assertion(self) -> None:
        spec = QuantumProgramSpec(
            program_name="single_h",
            qubits=["q0"],
            gates=[GateSpec(name="H", targets=["q0"])],
            units=[UnitSpec(name="u0", qubits=["q0"])],
            assertions=[
                AssertionSpec(
                    kind="reachability",
                    target={"type": "basis_state", "scope": ["q0"], "state": "1"},
                )
            ],
        )
        trace = build_abstract_trace(spec)

        dispatched = evaluate_assertion(trace, spec)
        direct = evaluate_reachability_assertion(trace, spec)

        self.assertEqual(dispatched, direct)

    def test_checked_mode_rejects_inconsistent_covering_units(self) -> None:
        state = AbstractState(
            units=(
                abstract_local_state(DensityMatrix.from_label("0"), ["q0"], name="u0"),
                abstract_local_state(DensityMatrix.from_label("01"), ["q0", "q1"], name="u01"),
            ),
            position=0,
        )

        with self.assertRaisesRegex(ValueError, "inconsistent"):
            reconstruct_scope_state(state, ["q0", "q1"], ["q0"], mode="checked")

    def test_checked_mode_terminal_probability_rejects_inconsistent_covering_units(self) -> None:
        spec = QuantumProgramSpec(
            program_name="bad_overlap_probability",
            qubits=["q0", "q1"],
            gates=[],
            units=[UnitSpec(name="u0", qubits=["q0"])],
            assertions=[
                AssertionSpec(
                    kind="probability",
                    target={
                        "type": "measurement_outcome",
                        "scope": ["q0"],
                        "outcomes": ["0"],
                    },
                    comparator=">=",
                    threshold=0.0,
                )
            ],
        )
        trace = AbstractExecutionTrace(
            states=(
                AbstractState(
                    units=(
                        abstract_local_state(DensityMatrix.from_label("0"), ["q0"], name="u0"),
                        abstract_local_state(DensityMatrix.from_label("01"), ["q0", "q1"], name="u01"),
                    ),
                    position=0,
                ),
            ),
            transitions=(),
        )

        with self.assertRaises(AbstractPropertyCheckingError):
            evaluate_terminal_probability_assertion(trace, spec, reconstruction_mode="checked")


if __name__ == "__main__":
    unittest.main()
