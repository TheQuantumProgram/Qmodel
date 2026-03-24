import unittest

import numpy as np
from qiskit.quantum_info import DensityMatrix

from qmodel.abstract.state import (
    AbstractState,
    AbstractTransition,
    AbstractUnitState,
    abstract_local_state,
    support_projector,
)


class AbstractStateTests(unittest.TestCase):
    def test_support_projector_of_pure_zero_state_is_rank_one(self) -> None:
        rho = DensityMatrix.from_label("0")

        projector = support_projector(rho)

        np.testing.assert_allclose(projector, np.array([[1.0, 0.0], [0.0, 0.0]]))

    def test_support_projector_of_hadamard_state_matches_plus_projector(self) -> None:
        rho = DensityMatrix(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex))

        projector = support_projector(rho)

        expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        np.testing.assert_allclose(projector, expected)

    def test_support_projector_of_maximally_mixed_state_is_identity(self) -> None:
        rho = DensityMatrix(np.eye(2, dtype=complex) / 2.0)

        projector = support_projector(rho)

        np.testing.assert_allclose(projector, np.eye(2, dtype=complex))

    def test_abstract_local_state_keeps_qubits_name_and_witness(self) -> None:
        rho = DensityMatrix.from_label("0")

        unit = abstract_local_state(rho, ["q2"], name="u2")

        self.assertIsInstance(unit, AbstractUnitState)
        self.assertEqual(unit.qubits, ("q2",))
        self.assertEqual(unit.name, "u2")
        np.testing.assert_allclose(unit.projector, np.array([[1.0, 0.0], [0.0, 0.0]]))
        np.testing.assert_allclose(unit.witness_rho.data, rho.data)

    def test_dataclasses_capture_minimal_abstract_state_and_transition(self) -> None:
        unit = abstract_local_state(DensityMatrix.from_label("0"), ["q0"], name="u0")
        state = AbstractState(units=(unit,), position=0, metadata={"gate": "init"})
        transition = AbstractTransition(
            source_id="s0",
            target_id="s1",
            label="g1",
            affected_views=("u0",),
            metadata={"workspace_size": 1},
        )

        self.assertEqual(state.position, 0)
        self.assertEqual(state.units[0].name, "u0")
        self.assertEqual(transition.label, "g1")
        self.assertEqual(transition.affected_views, ("u0",))


if __name__ == "__main__":
    unittest.main()
