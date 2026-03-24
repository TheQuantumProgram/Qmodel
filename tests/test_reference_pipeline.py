from __future__ import annotations

import unittest
from pathlib import Path

from qmodel.concrete.qiskit_backend import (
    build_circuit,
    measurement_outcome_probability,
    simulate_statevector,
)
from qmodel.parser.qmodel_parser import parse_qmodel_file


_MODELS_DIR = Path(__file__).resolve().parent / "models"


class ReferencePipelineTests(unittest.TestCase):
    def test_clifford_bell_model_generates_correct_reference_circuit(self) -> None:
        spec = parse_qmodel_file(str(_MODELS_DIR / "clifford_bell.qmodel"))

        circuit = build_circuit(spec)
        self.assertEqual(
            [instruction.operation.name for instruction in circuit.data],
            ["h", "cx", "measure", "measure"],
        )

    def test_clifford_bell_model_generates_expected_terminal_state(self) -> None:
        spec = parse_qmodel_file(str(_MODELS_DIR / "clifford_bell.qmodel"))

        state = simulate_statevector(spec)
        probabilities = {
            str(bits): float(value) for bits, value in state.probabilities_dict().items()
        }

        self.assertAlmostEqual(probabilities["00"], 0.5)
        self.assertAlmostEqual(probabilities["11"], 0.5)
        self.assertAlmostEqual(
            measurement_outcome_probability(spec, ["q0", "q1"], ["00", "11"]),
            1.0,
        )
        self.assertAlmostEqual(
            measurement_outcome_probability(spec, ["q0", "q1"], ["01", "10"]),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
