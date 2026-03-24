from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from qmodel.parser.qmodel_parser import QModelParseError, parse_qmodel_file


_MODELS_DIR = Path(__file__).resolve().parent / "models"


class QModelParserTests(unittest.TestCase):
    def test_parse_clifford_bell_model(self) -> None:
        spec = parse_qmodel_file(str(_MODELS_DIR / "clifford_bell.qmodel"))

        self.assertEqual(spec.program_name, "clifford_bell")
        self.assertEqual(spec.qubits, ["q0", "q1"])
        self.assertEqual(len(spec.gates), 2)
        self.assertEqual(spec.gates[1].name, "CX")
        self.assertEqual(spec.gates[1].controls, ["q0"])
        self.assertEqual(spec.measurement.qubits, ["q0", "q1"])
        self.assertEqual(len(spec.units), 0)
        self.assertIsNotNone(spec.organization_schedule)
        self.assertEqual([state.name for state in spec.organization_schedule.states], ["s0", "s1", "s2"])
        self.assertEqual([unit.name for unit in spec.organization_schedule.states[0].units], ["bell_pair"])
        self.assertEqual(len(spec.assertions), 1)
        self.assertEqual(spec.assertions[0].kind, "probability")

    def test_parse_ccx_overlap_demo_model(self) -> None:
        spec = parse_qmodel_file(str(_MODELS_DIR / "ccx_overlap_demo.qmodel"))

        self.assertEqual(spec.program_name, "ccx_overlap_demo")
        self.assertEqual(spec.gates[4].name, "CCX")
        self.assertEqual(spec.gates[4].controls, ["q0", "q1"])
        self.assertEqual(spec.gates[4].targets, ["q2"])
        self.assertEqual(len(spec.units), 0)
        self.assertIsNotNone(spec.organization_schedule)
        self.assertEqual(spec.organization_schedule.initial_state, "s0")
        self.assertEqual([unit.name for unit in spec.organization_schedule.states[5].units], ["u012", "u3", "u4"])
        self.assertEqual([unit.name for unit in spec.organization_schedule.states[-1].units], ["u012", "u23", "u34"])

    def test_parse_model_with_organization_schedule(self) -> None:
        spec = parse_qmodel_file(str(_MODELS_DIR / "organization_schedule_chain.qmodel"))

        self.assertIsNotNone(spec.organization_schedule)
        self.assertEqual(spec.organization_schedule.initial_state, "s0")
        self.assertEqual([state.name for state in spec.organization_schedule.states], ["s0", "s1", "s2", "s3"])
        self.assertEqual(spec.organization_schedule.states[0].transition.gate_index, 0)
        self.assertEqual(spec.organization_schedule.states[0].transition.next_state, "s1")
        self.assertEqual([unit.name for unit in spec.organization_schedule.states[2].units], ["uq01", "uq2"])

    def test_reject_missing_format_field(self) -> None:
        source = textwrap.dedent(
            """
            program_name: broken
            qubits: [q0]
            initial_state: zero
            gates:
              - name: H
                targets: [q0]
            assertion:
              kind: probability
              target:
                type: measurement_outcome
                scope: [q0]
                outcomes: ["1"]
              comparator: ">="
              threshold: 0.5
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".qmodel", delete=False) as handle:
            handle.write(source)
            path = handle.name

        try:
            with self.assertRaisesRegex(QModelParseError, "format"):
                parse_qmodel_file(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_parse_model_with_ry_gate(self) -> None:
        source = textwrap.dedent(
            """
            format: qmodel-v1
            program_name: ry_demo
            qubits: [q0]
            initial_state: zero
            gates:
              - name: Ry
                targets: [q0]
                params:
                  theta: 1.0471975511965976
            units:
              - name: u0
                qubits: [q0]
            assertion:
              kind: probability
              target:
                type: measurement_outcome
                scope: [q0]
                outcomes: ["1"]
              comparator: "="
              threshold: 0.25
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".qmodel", delete=False) as handle:
            handle.write(source)
            path = handle.name

        try:
            spec = parse_qmodel_file(path)
        finally:
            Path(path).unlink(missing_ok=True)

        self.assertEqual(spec.gates[0].name, "Ry")
        self.assertEqual(spec.gates[0].params["theta"], 1.0471975511965976)

    def test_reject_gate_with_invalid_arity(self) -> None:
        source = textwrap.dedent(
            """
            format: qmodel-v1
            program_name: bad_ccx
            qubits: [q0, q1, q2]
            initial_state: zero
            gates:
              - name: CCX
                controls: [q0]
                targets: [q2]
            assertion:
              kind: probability
              target:
                type: measurement_outcome
                scope: [q2]
                outcomes: ["1"]
              comparator: ">="
              threshold: 0.5
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".qmodel", delete=False) as handle:
            handle.write(source)
            path = handle.name

        try:
            with self.assertRaisesRegex(QModelParseError, "CCX"):
                parse_qmodel_file(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_reject_noncomputational_measurement_basis(self) -> None:
        source = textwrap.dedent(
            """
            format: qmodel-v1
            program_name: bad_basis
            qubits: [q0]
            initial_state: zero
            gates:
              - name: H
                targets: [q0]
            measurement:
              qubits: [q0]
              basis: hadamard
            assertion:
              kind: probability
              target:
                type: measurement_outcome
                scope: [q0]
                outcomes: ["1"]
              comparator: ">="
              threshold: 0.5
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".qmodel", delete=False) as handle:
            handle.write(source)
            path = handle.name

        try:
            with self.assertRaisesRegex(QModelParseError, "computational"):
                parse_qmodel_file(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_reject_probability_assertion_without_threshold(self) -> None:
        source = textwrap.dedent(
            """
            format: qmodel-v1
            program_name: bad_assertion
            qubits: [q0]
            initial_state: zero
            gates:
              - name: H
                targets: [q0]
            assertion:
              kind: probability
              target:
                type: measurement_outcome
                scope: [q0]
                outcomes: ["1"]
              comparator: ">="
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".qmodel", delete=False) as handle:
            handle.write(source)
            path = handle.name

        try:
            with self.assertRaisesRegex(QModelParseError, "threshold"):
                parse_qmodel_file(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_reject_probability_assertion_without_outcomes(self) -> None:
        source = textwrap.dedent(
            """
            format: qmodel-v1
            program_name: bad_probability_target
            qubits: [q0]
            initial_state: zero
            gates: []
            units:
              - name: u0
                qubits: [q0]
            assertion:
              kind: probability
              target:
                type: measurement_outcome
                scope: [q0]
              comparator: ">="
              threshold: 0.0
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".qmodel", delete=False) as handle:
            handle.write(source)
            path = handle.name

        try:
            with self.assertRaisesRegex(QModelParseError, "outcomes"):
                parse_qmodel_file(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_reject_probability_assertion_with_unsupported_comparator(self) -> None:
        source = textwrap.dedent(
            """
            format: qmodel-v1
            program_name: bad_probability_comparator
            qubits: [q0]
            initial_state: zero
            gates: []
            units:
              - name: u0
                qubits: [q0]
            assertion:
              kind: probability
              target:
                type: measurement_outcome
                scope: [q0]
                outcomes: ["0"]
              comparator: ">>"
              threshold: 0.0
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".qmodel", delete=False) as handle:
            handle.write(source)
            path = handle.name

        try:
            with self.assertRaisesRegex(QModelParseError, "comparator"):
                parse_qmodel_file(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_reject_reachability_assertion_with_unsupported_target_type(self) -> None:
        source = textwrap.dedent(
            """
            format: qmodel-v1
            program_name: bad_reach
            qubits: [q0]
            initial_state: zero
            gates:
              - name: H
                targets: [q0]
            assertion:
              kind: reachability
              target:
                type: projector
                scope: [q0]
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".qmodel", delete=False) as handle:
            handle.write(source)
            path = handle.name

        try:
            with self.assertRaisesRegex(QModelParseError, "basis_state"):
                parse_qmodel_file(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_reject_organization_schedule_with_non_linear_gate_coverage(self) -> None:
        source = textwrap.dedent(
            """
            format: qmodel-v1
            program_name: bad_schedule
            qubits: [q0]
            initial_state: zero
            gates:
              - name: H
                targets: [q0]
            organization_schedule:
              initial_state: s0
              states:
                - name: s0
                  units:
                    - name: uq0
                      qubits: [q0]
                  transition:
                    gate_index: 0
                    next_state: s2
                - name: s1
                  units:
                    - name: uq0
                      qubits: [q0]
                - name: s2
                  units:
                    - name: uq0
                      qubits: [q0]
            assertion:
              kind: probability
              target:
                type: measurement_outcome
                scope: [q0]
                outcomes: ["1"]
              comparator: ">="
              threshold: 0.0
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".qmodel", delete=False) as handle:
            handle.write(source)
            path = handle.name

        try:
            with self.assertRaisesRegex(QModelParseError, "organization_schedule"):
                parse_qmodel_file(path)
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
