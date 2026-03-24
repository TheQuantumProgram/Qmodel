import unittest

from qmodel.spec import GateSpec, QuantumProgramSpec
from qmodel.validation import SpecValidationError, validate_program_spec


class QuantumProgramSpecValidationTests(unittest.TestCase):
    def test_minimal_program_spec_is_valid(self) -> None:
        spec = QuantumProgramSpec(
            program_name="minimal",
            qubits=["q0"],
            gates=[GateSpec(name="H", targets=["q0"])],
        )

        validate_program_spec(spec)

    def test_supported_clifford_gate_is_valid(self) -> None:
        spec = QuantumProgramSpec(
            program_name="clifford_showcase",
            qubits=["q0", "q1", "q2"],
            gates=[
                GateSpec(name="S", targets=["q0"]),
                GateSpec(name="CZ", controls=["q0"], targets=["q1"]),
                GateSpec(name="SWAP", targets=["q1", "q2"]),
            ],
        )

        validate_program_spec(spec)

    def test_unknown_gate_name_is_rejected(self) -> None:
        spec = QuantumProgramSpec(
            program_name="bad_gate",
            qubits=["q0"],
            gates=[GateSpec(name="FOO", targets=["q0"])],
        )

        with self.assertRaisesRegex(SpecValidationError, "Unsupported gate"):
            validate_program_spec(spec)


if __name__ == "__main__":
    unittest.main()
