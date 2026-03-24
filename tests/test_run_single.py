from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_MODELS_DIR = _ROOT / "tests" / "models"
_SCRIPT = _ROOT / "scripts" / "run_single.py"


class RunSingleScriptTests(unittest.TestCase):
    def _run(self, *args: str) -> dict[str, object]:
        completed = subprocess.run(
            [
                str(_ROOT / ".venv" / "bin" / "python"),
                str(_SCRIPT),
                *args,
            ],
            cwd=_ROOT.parent,
            check=True,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(_ROOT / "src")},
        )
        return json.loads(completed.stdout)

    def _run_temp_model(self, source: str, *args: str) -> dict[str, object]:
        with tempfile.NamedTemporaryFile("w", suffix=".qmodel", delete=False) as handle:
            handle.write(textwrap.dedent(source).strip())
            model_path = handle.name
        try:
            return self._run(*args, model_path)
        finally:
            Path(model_path).unlink(missing_ok=True)

    def test_run_single_default_mode_reports_matching_probability_results(self) -> None:
        result = self._run(str(_MODELS_DIR / "clifford_bell.qmodel"))

        self.assertEqual(result["program_name"], "clifford_bell")
        self.assertEqual(result["reconstruction_mode"], "trusted")
        self.assertEqual(result["assertion_kind"], "probability")
        self.assertFalse(result["run_concrete"])
        self.assertEqual(result["concrete"]["status"], "skipped")
        self.assertEqual(result["abstract"]["judgment"], "satisfied")
        self.assertGreater(result["abstract"]["elapsed_seconds"], 0.0)
        self.assertEqual(result["abstract"]["max_state_bytes"], 256)
        self.assertEqual(result["abstract"]["max_transition_bytes"], 256)
        self.assertEqual(result["abstract"]["max_execution_bytes"], 256)
        self.assertEqual(
            result["comparison"]["abstract_ideal_pure_lower_bound"]["max_state_bytes"],
            64,
        )
        self.assertEqual(
            result["comparison"]["abstract_ideal_pure_lower_bound"]["max_execution_bytes"],
            64,
        )
        self.assertEqual(result["comparison"]["full_execution"]["qubit_count"], 2)
        self.assertEqual(result["comparison"]["full_execution"]["statevector_bytes"], 64)
        self.assertEqual(result["comparison"]["full_execution"]["density_matrix_bytes"], 256)
        self.assertEqual(
            result["comparison"]["full_execution"]["time_benchmark"]["mode"],
            "measured",
        )
        self.assertAlmostEqual(
            result["abstract"]["max_execution_mib"],
            256 / (1024 * 1024),
        )

    def test_run_single_checked_mode_reports_success_for_single_view_model(self) -> None:
        result = self._run(
            "--mode",
            "checked",
            str(_MODELS_DIR / "clifford_bell.qmodel"),
        )

        self.assertEqual(result["reconstruction_mode"], "checked")
        self.assertEqual(result["abstract"]["judgment"], "satisfied")

    def test_run_single_can_optionally_run_concrete_backend(self) -> None:
        result = self._run(
            "--run-concrete",
            str(_MODELS_DIR / "clifford_bell.qmodel"),
        )

        self.assertTrue(result["run_concrete"])
        self.assertEqual(result["concrete"]["judgment"], "satisfied")
        self.assertAlmostEqual(result["concrete"]["probability"], result["abstract"]["probability"])
        self.assertGreater(result["abstract"]["elapsed_seconds"], 0.0)
        self.assertEqual(result["abstract"]["max_state_bytes"], 256)
        self.assertEqual(result["abstract"]["max_execution_bytes"], 256)

    def test_run_single_reports_organization_schedule_models(self) -> None:
        result = self._run(str(_MODELS_DIR / "organization_schedule_chain.qmodel"))

        self.assertEqual(result["program_name"], "organization_schedule_chain")
        self.assertEqual(result["assertion_kind"], "probability")
        self.assertEqual(result["abstract"]["judgment"], "satisfied")
        self.assertGreater(result["abstract"]["elapsed_seconds"], 0.0)
        self.assertEqual(result["abstract"]["max_state_bytes"], 512)
        self.assertEqual(result["abstract"]["max_transition_bytes"], 1024)
        self.assertEqual(result["abstract"]["max_execution_bytes"], 1024)
        self.assertEqual(
            result["comparison"]["abstract_ideal_pure_lower_bound"]["max_state_bytes"],
            128,
        )
        self.assertEqual(
            result["comparison"]["abstract_ideal_pure_lower_bound"]["max_transition_bytes"],
            128,
        )
        self.assertEqual(result["comparison"]["full_execution"]["qubit_count"], 3)
        self.assertEqual(result["comparison"]["full_execution"]["statevector_bytes"], 128)
        self.assertEqual(result["comparison"]["full_execution"]["density_matrix_bytes"], 1024)

    def test_run_single_reports_max_state_bytes_from_current_units_only(self) -> None:
        result = self._run_temp_model(
            """
            format: qmodel-v1
            program_name: state_size_demo
            qubits: [q0, q1, q2, q3, q4]
            initial_state: zero
            gates:
              - name: H
                targets: [q0]
                label: expand-left-block
            organization_schedule:
              initial_state: s0
              states:
                - name: s0
                  units:
                    - name: u0
                      qubits: [q0, q1, q2]
                    - name: u1
                      qubits: [q3]
                    - name: u2
                      qubits: [q4]
                  transition:
                    gate_index: 0
                    next_state: s1
                - name: s1
                  units:
                    - name: u0
                      qubits: [q0, q1, q2]
                    - name: u1
                      qubits: [q2, q3]
                    - name: u2
                      qubits: [q4]
            assertion:
              name: q0_probability
              kind: probability
              target:
                type: measurement_outcome
                scope: [q0]
                outcomes: ["1"]
              comparator: ">="
              threshold: 0.0
            """
        )

        self.assertEqual(result["program_name"], "state_size_demo")
        self.assertEqual(result["abstract"]["max_state_bytes"], 1344)
        self.assertEqual(result["abstract"]["max_transition_bytes"], 4160)
        self.assertEqual(result["abstract"]["max_execution_bytes"], 4160)
        self.assertEqual(
            result["comparison"]["abstract_ideal_pure_lower_bound"]["max_state_bytes"],
            224,
        )
        self.assertEqual(
            result["comparison"]["abstract_ideal_pure_lower_bound"]["max_transition_bytes"],
            288,
        )
        self.assertEqual(
            result["comparison"]["abstract_ideal_pure_lower_bound"]["max_execution_bytes"],
            288,
        )
        self.assertEqual(result["comparison"]["full_execution"]["qubit_count"], 5)
        self.assertEqual(result["comparison"]["full_execution"]["statevector_bytes"], 512)
        self.assertEqual(result["comparison"]["full_execution"]["density_matrix_bytes"], 16384)

    def test_run_single_skips_full_execution_timing_above_cutoff(self) -> None:
        qubits = ", ".join(f"q{i}" for i in range(26))
        units = "\n".join(
            [
                "  - name: u{0}\n    qubits: [q{0}]".format(i)
                for i in range(26)
            ]
        )
        result = self._run_temp_model(
            f"""
format: qmodel-v1
program_name: cutoff_skip_demo
qubits: [{qubits}]
initial_state: zero
gates: []
units:
{units}
assertion:
  name: q0_probability
  kind: probability
  target:
    type: measurement_outcome
    scope: [q0]
    outcomes: ["0"]
  comparator: "="
  threshold: 1.0
"""
        )

        self.assertEqual(result["comparison"]["full_execution"]["qubit_count"], 26)
        self.assertEqual(
            result["comparison"]["full_execution"]["time_benchmark"]["mode"],
            "skipped",
        )

    def test_run_single_can_optionally_run_concrete_backend_for_reachability(self) -> None:
        result = self._run_temp_model(
            """
            format: qmodel-v1
            program_name: single_h_reach
            qubits: [q0]
            initial_state: zero
            gates:
              - name: H
                targets: [q0]
            units:
              - name: u0
                qubits: [q0]
            assertion:
              name: reach_one
              kind: reachability
              target:
                type: basis_state
                scope: [q0]
                state: "1"
            """,
            "--run-concrete",
        )

        self.assertTrue(result["run_concrete"])
        self.assertEqual(result["assertion_kind"], "reachability")
        self.assertEqual(result["abstract"]["judgment"], "satisfied")
        self.assertEqual(result["concrete"]["judgment"], "satisfied")
        self.assertEqual(result["concrete"]["first_reached_index"], 1)


if __name__ == "__main__":
    unittest.main()
