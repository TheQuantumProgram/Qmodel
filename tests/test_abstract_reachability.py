from __future__ import annotations

import unittest

import numpy as np

from qmodel.abstract.property_checking import (
    evaluate_reachability_assertion,
    state_scope_witness,
)
from qmodel.abstract.transition import build_abstract_trace
from qmodel.spec import AssertionSpec, GateSpec, QuantumProgramSpec, UnitSpec


class AbstractReachabilityTests(unittest.TestCase):
    def test_state_scope_witness_returns_intermediate_scope_state(self) -> None:
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
        witness = state_scope_witness(trace, spec, 1, ["q0"])
        expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)

        np.testing.assert_allclose(witness.data, expected)

    def test_reachability_assertion_is_satisfied_when_target_state_appears(self) -> None:
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
        result = evaluate_reachability_assertion(trace, spec)

        self.assertEqual(result["judgment"], "satisfied")
        self.assertEqual(result["first_reached_index"], 1)
        self.assertAlmostEqual(result["max_overlap"], 0.5)

    def test_reachability_assertion_is_violated_when_target_state_never_appears(self) -> None:
        spec = QuantumProgramSpec(
            program_name="single_idle",
            qubits=["q0"],
            gates=[],
            units=[UnitSpec(name="u0", qubits=["q0"])],
            assertions=[
                AssertionSpec(
                    kind="reachability",
                    target={"type": "basis_state", "scope": ["q0"], "state": "1"},
                )
            ],
        )

        trace = build_abstract_trace(spec)
        result = evaluate_reachability_assertion(trace, spec)

        self.assertEqual(result["judgment"], "violated")
        self.assertIsNone(result["first_reached_index"])
        self.assertAlmostEqual(result["max_overlap"], 0.0)


if __name__ == "__main__":
    unittest.main()
