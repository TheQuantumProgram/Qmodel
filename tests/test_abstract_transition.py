import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace

from qmodel.abstract.state import AbstractState, abstract_local_state
from qmodel.abstract.transition import (
    gate_support,
    merge_update_rewrite,
    reconstruct_scope_state,
    select_reconstruction_support_units,
    select_affected_views,
)
from qmodel.spec import GateSpec, UnitSpec


class AbstractTransitionTests(unittest.TestCase):
    def test_select_affected_views_for_single_qubit_gate_in_overlap(self) -> None:
        units = (
            abstract_local_state(DensityMatrix.from_label("000"), ["q0", "q1", "q2"], name="u1"),
            abstract_local_state(DensityMatrix.from_label("00"), ["q2", "q3"], name="u2"),
            abstract_local_state(DensityMatrix.from_label("00"), ["q3", "q4"], name="u3"),
        )
        gate = GateSpec(name="H", targets=["q2"])

        affected = select_affected_views(units, gate)

        self.assertEqual([unit.name for unit in affected], ["u1", "u2"])

    def test_select_affected_views_for_cx_q2_q3_hits_three_views(self) -> None:
        units = (
            abstract_local_state(DensityMatrix.from_label("000"), ["q0", "q1", "q2"], name="u1"),
            abstract_local_state(DensityMatrix.from_label("00"), ["q2", "q3"], name="u2"),
            abstract_local_state(DensityMatrix.from_label("00"), ["q3", "q4"], name="u3"),
        )
        gate = GateSpec(name="CX", controls=["q2"], targets=["q3"])

        affected = select_affected_views(units, gate)

        self.assertEqual([unit.name for unit in affected], ["u1", "u2", "u3"])
        self.assertEqual(gate_support(gate), ("q2", "q3"))

    def test_merge_update_rewrite_uses_post_view_layout(self) -> None:
        qc = QuantumCircuit(4)
        qc.x(2)
        global_witness = DensityMatrix.from_instruction(qc)

        pre_state = AbstractState(
            units=(
                abstract_local_state(partial_trace(global_witness, [3]), ["q0", "q1", "q2"], name="u1"),
                abstract_local_state(partial_trace(global_witness, [0, 1, 2]), ["q3"], name="u2"),
            ),
            position=0,
        )
        gate = GateSpec(name="CX", controls=["q2"], targets=["q3"], label="cx_23")
        post_units = [
            UnitSpec(name="u1", qubits=["q0", "q1", "q2"]),
            UnitSpec(name="u2", qubits=["q2", "q3"]),
        ]

        next_state = merge_update_rewrite(
            pre_state=pre_state,
            gate=gate,
            global_qubits=["q0", "q1", "q2", "q3"],
            post_units=post_units,
        )

        self.assertEqual(next_state.position, 1)
        self.assertEqual(next_state.metadata["affected_pre_views"], ("u1", "u2"))
        self.assertEqual(next_state.metadata["affected_post_views"], ("u1", "u2"))
        self.assertEqual(next_state.metadata["workspace_qubits"], ("q0", "q1", "q2", "q3"))

        units_by_name = {unit.name: unit for unit in next_state.units}
        np.testing.assert_allclose(
            units_by_name["u2"].witness_rho.data,
            np.diag([0.0, 0.0, 0.0, 1.0]).astype(complex),
        )

    def test_merge_update_rewrite_checked_mode_accepts_consistent_overlap_without_shared_certificate(self) -> None:
        pre_state = AbstractState(
            units=(
                abstract_local_state(DensityMatrix.from_label("000"), ["q0", "q1", "q2"], name="u1"),
                abstract_local_state(DensityMatrix.from_label("00"), ["q2", "q3"], name="u2"),
            ),
            position=0,
        )
        gate = GateSpec(name="H", targets=["q2"])

        next_state = merge_update_rewrite(
            pre_state=pre_state,
            gate=gate,
            global_qubits=["q0", "q1", "q2", "q3"],
            post_units=[
                UnitSpec(name="u1", qubits=["q0", "q1", "q2"]),
                UnitSpec(name="u2", qubits=["q2", "q3"]),
            ],
            reconstruction_mode="checked",
        )

        self.assertEqual(next_state.position, 1)
        self.assertEqual(next_state.metadata["affected_pre_views"], ("u1", "u2"))

    def test_reconstruct_scope_state_checked_mode_accepts_consistent_overlap_without_certificate(self) -> None:
        state = AbstractState(
            units=(
                abstract_local_state(DensityMatrix.from_label("00"), ["q0", "q1"], name="u01"),
                abstract_local_state(DensityMatrix.from_label("00"), ["q1", "q2"], name="u12"),
            ),
            position=0,
        )

        witness = reconstruct_scope_state(
            state,
            ["q0", "q1", "q2"],
            ["q0", "q1", "q2"],
            mode="checked",
        )

        np.testing.assert_allclose(
            witness.data,
            np.diag([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype(complex),
        )

    def test_merge_update_rewrite_does_not_truncate_static_fixed_windows(self) -> None:
        pre_state = AbstractState(
            units=tuple(
                abstract_local_state(
                    DensityMatrix.from_label("0000"),
                    [f"q{start + offset}" for offset in range(4)],
                    name=f"w_{start}_{start+3}",
                )
                for start in range(5)
            ),
            position=0,
        )
        post_units = [
            UnitSpec(name=f"w_{start}_{start+3}", qubits=[f"q{start + offset}" for offset in range(4)])
            for start in range(5)
        ]

        next_state = merge_update_rewrite(
            pre_state=pre_state,
            gate=GateSpec(name="H", targets=["q0"], label="prep-h-0"),
            global_qubits=[f"q{i}" for i in range(8)],
            post_units=post_units,
            reconstruction_mode="trusted",
        )

        by_name = {unit.name: unit for unit in next_state.units}
        self.assertEqual(by_name["w_4_7"].witness_rho.data.shape, (16, 16))
        np.testing.assert_allclose(
            by_name["w_4_7"].witness_rho.data,
            DensityMatrix.from_label("0000").data,
        )

    def test_select_reconstruction_support_units_is_not_transitive_on_fixed_windows(self) -> None:
        units = tuple(
            abstract_local_state(
                DensityMatrix.from_label("0000"),
                [f"q{start + offset}" for offset in range(4)],
                name=f"w_{start}_{start+3}",
            )
            for start in range(9)
        )

        selected = select_reconstruction_support_units(
            all_units=units,
            seed_units=units[2:6],
            workspace_qubits=["q2", "q3", "q4", "q5", "q6", "q7", "q8"],
            global_qubits=[f"q{i}" for i in range(12)],
        )

        self.assertEqual(
            [unit.name for unit in selected],
            ["w_2_5", "w_3_6", "w_4_7", "w_5_8"],
        )


if __name__ == "__main__":
    unittest.main()
