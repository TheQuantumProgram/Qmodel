from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from qmodel.abstract.transition import build_abstract_trace, execute_abstract_to_final_state
from qmodel.spec import AssertionSpec, GateSpec, QuantumProgramSpec, UnitSpec
from qmodel.parser.qmodel_parser import parse_qmodel_file


_MODELS_DIR = Path(__file__).resolve().parent / "models"


class AbstractTraceTests(unittest.TestCase):
    def test_build_abstract_trace_for_bell_model(self) -> None:
        spec = parse_qmodel_file(str(_MODELS_DIR / "clifford_bell.qmodel"))

        trace = build_abstract_trace(spec)

        self.assertEqual(len(trace.states), 3)
        self.assertEqual(len(trace.transitions), 2)
        self.assertEqual(trace.states[0].position, 0)
        self.assertEqual(trace.states[-1].position, 2)
        self.assertEqual(trace.transitions[0].label, "prepare-control")
        self.assertEqual(trace.transitions[1].label, "entangle-pair")
        self.assertEqual(trace.transitions[0].affected_views, ("bell_pair",))

        final_unit = trace.states[-1].units[0]
        expected = np.array(
            [
                [0.5, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.5],
            ],
            dtype=complex,
        )
        np.testing.assert_allclose(final_unit.witness_rho.data, expected)

    def test_build_abstract_trace_keeps_static_overlap_layout_by_default(self) -> None:
        spec = parse_qmodel_file(str(_MODELS_DIR / "clifford_gate_showcase.qmodel"))

        trace = build_abstract_trace(spec)

        self.assertEqual(len(trace.states), len(spec.gates) + 1)
        self.assertEqual(len(trace.transitions), len(spec.gates))
        self.assertEqual([unit.name for unit in trace.states[0].units], ["left_pair", "right_pair"])
        self.assertEqual([unit.name for unit in trace.states[-1].units], ["left_pair", "right_pair"])

    def test_build_abstract_trace_uses_organization_schedule_when_present(self) -> None:
        spec = parse_qmodel_file(str(_MODELS_DIR / "organization_schedule_chain.qmodel"))

        trace = build_abstract_trace(spec)

        self.assertEqual(len(trace.states), 4)
        self.assertEqual([unit.name for unit in trace.states[0].units], ["uq0", "uq1", "uq2"])
        self.assertEqual([unit.name for unit in trace.states[2].units], ["uq01", "uq2"])
        self.assertEqual([unit.name for unit in trace.states[3].units], ["uq01", "uq12"])
        self.assertEqual([transition.label for transition in trace.transitions], ["prepare-q0", "couple-q0-q1", "couple-q1-q2"])

    def test_build_abstract_trace_checked_mode_accepts_overlapping_zero_initial_units(self) -> None:
        spec = QuantumProgramSpec(
            program_name="overlap_init_checked",
            qubits=["q0", "q1", "q2"],
            gates=[GateSpec(name="H", targets=["q1"])],
            units=[
                UnitSpec(name="u01", qubits=["q0", "q1"]),
                UnitSpec(name="u12", qubits=["q1", "q2"]),
            ],
            assertions=[
                AssertionSpec(
                    kind="probability",
                    target={
                        "type": "measurement_outcome",
                        "scope": ["q1"],
                        "outcomes": ["0"],
                    },
                    comparator=">=",
                    threshold=0.0,
                )
            ],
        )

        trace = build_abstract_trace(spec, reconstruction_mode="checked")

        self.assertEqual(len(trace.states), 2)
        self.assertEqual(trace.transitions[0].affected_views, ("u01", "u12"))

    def test_initial_overlapping_units_do_not_create_component_witness_certificate(self) -> None:
        spec = QuantumProgramSpec(
            program_name="overlap_init_no_global_cert",
            qubits=["q0", "q1", "q2"],
            gates=[],
            units=[
                UnitSpec(name="u01", qubits=["q0", "q1"]),
                UnitSpec(name="u12", qubits=["q1", "q2"]),
            ],
        )

        trace = build_abstract_trace(spec)

        self.assertEqual(len(trace.states), 1)
        self.assertEqual(trace.states[0].certificates, ())
        self.assertEqual(trace.states[0].units[0].certificate_ids, ())
        self.assertEqual(trace.states[0].units[1].certificate_ids, ())
        self.assertEqual(trace.states[0].units[0].witness_rho.data.shape, (2**2, 2**2))
        self.assertEqual(trace.states[0].units[1].witness_rho.data.shape, (2**2, 2**2))

    def test_initial_fixed_overlapping_windows_do_not_create_global_component_certificate(self) -> None:
        qubits = [f"q{i}" for i in range(8)]
        units = [
            UnitSpec(name=f"w_{i}_{i+3}", qubits=[f"q{j}" for j in range(i, i + 4)])
            for i in range(5)
        ]
        spec = QuantumProgramSpec(
            program_name="fixed_overlap_init_no_global_cert",
            qubits=qubits,
            gates=[],
            units=units,
        )

        trace = build_abstract_trace(spec)

        self.assertEqual(len(trace.states), 1)
        self.assertEqual(trace.states[0].certificates, ())
        self.assertEqual(len(trace.states[0].units), len(units))
        for unit in trace.states[0].units:
            self.assertEqual(unit.certificate_ids, ())
            self.assertEqual(unit.witness_rho.data.shape, (2**4, 2**4))

    def test_execute_abstract_to_final_state_matches_trace_final_state_and_stats(self) -> None:
        spec = parse_qmodel_file(str(_MODELS_DIR / "organization_schedule_chain.qmodel"))

        trace = build_abstract_trace(spec)
        execution = execute_abstract_to_final_state(spec)

        self.assertEqual(execution.final_state.position, trace.states[-1].position)
        self.assertEqual(
            [unit.name for unit in execution.final_state.units],
            [unit.name for unit in trace.states[-1].units],
        )
        for actual, expected in zip(execution.final_state.units, trace.states[-1].units, strict=True):
            np.testing.assert_allclose(actual.witness_rho.data, expected.witness_rho.data)

        self.assertEqual(
            execution.stats.max_state_bytes,
            max(
                sum(int(unit.witness_rho.data.nbytes) for unit in state.units)
                for state in trace.states
            ),
        )
        self.assertEqual(
            execution.stats.max_transition_bytes,
            max(
                int(transition.metadata.get("transition_peak_bytes", 0))
                for transition in trace.transitions
            ),
        )


if __name__ == "__main__":
    unittest.main()
