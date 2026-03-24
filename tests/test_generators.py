from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
import io
import importlib.util

from qmodel.benchmarks.generators import (
    build_aiqft_family_payloads,
    build_aiqft_payload,
    build_ghz_family_payloads,
    build_ghz_staircase_payload,
    build_bv_family_payloads,
    build_bv_payload,
    emit_aiqft_family_models,
    emit_bv_family_models,
    write_qmodel_payload,
)
from qmodel.parser import parse_qmodel_file


class GHZGeneratorTests(unittest.TestCase):
    def test_build_standard_ghz_payload_for_n10(self) -> None:
        payload = build_ghz_staircase_payload(10)

        self.assertEqual(payload["program_name"], "ghz_10_staircase")
        self.assertEqual(payload["metadata"]["family"], "ghz")
        self.assertEqual(payload["metadata"]["n"], 10)
        self.assertEqual(len(payload["gates"]), 10)
        self.assertEqual(payload["gates"][0]["name"], "H")
        self.assertEqual(payload["gates"][1]["name"], "CX")
        self.assertEqual(payload["organization_schedule"]["initial_state"], "s0")
        self.assertEqual(len(payload["organization_schedule"]["states"]), 11)
        final_units = payload["organization_schedule"]["states"][-1]["units"]
        self.assertEqual(len(final_units), 9)
        self.assertTrue(all(len(unit["qubits"]) == 2 for unit in final_units))
        self.assertEqual(payload["assertion"]["threshold"], 0.5)

    def test_build_biased_ghz_payload_uses_ry_and_probability_threshold(self) -> None:
        payload = build_ghz_staircase_payload(20, root_probability=0.25)

        self.assertEqual(payload["program_name"], "ghz_20_root_p025")
        self.assertEqual(payload["metadata"]["variant"], "biased_root")
        self.assertEqual(payload["metadata"]["root_probability"], 0.25)
        self.assertEqual(payload["gates"][0]["name"], "Ry")
        self.assertAlmostEqual(payload["gates"][0]["params"]["theta"], 1.0471975511965976)
        self.assertEqual(payload["assertion"]["threshold"], 0.25)

    def test_write_generated_ghz_payload_round_trips_through_parser(self) -> None:
        payload = build_ghz_staircase_payload(10, root_probability=0.75)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ghz_generated.qmodel"
            write_qmodel_payload(payload, path)
            spec = parse_qmodel_file(str(path))

        self.assertEqual(spec.program_name, "ghz_10_root_p075")
        self.assertEqual(spec.gates[0].name, "Ry")
        self.assertEqual(len(spec.organization_schedule.states), 11)
        self.assertEqual(spec.assertions[0].threshold, 0.75)

    def test_build_ghz_family_payloads_covers_standard_and_biased_models(self) -> None:
        payloads = build_ghz_family_payloads()
        names = sorted(payload["program_name"] for payload in payloads)

        self.assertEqual(
            names,
            [
                "ghz_100_staircase",
                "ghz_10_staircase",
                "ghz_150_staircase",
                "ghz_200_staircase",
                "ghz_20_root_p025",
                "ghz_20_staircase",
                "ghz_50_root_p075",
                "ghz_50_staircase",
            ],
        )


class BVGeneratorTests(unittest.TestCase):
    def test_build_bv_payload_for_n10_uses_sparse_evenly_distributed_hidden_string(self) -> None:
        payload = build_bv_payload(10)

        self.assertEqual(payload["program_name"], "bv_10_sparse")
        self.assertEqual(payload["metadata"]["family"], "bv")
        self.assertEqual(payload["metadata"]["n"], 10)
        self.assertEqual(len(payload["qubits"]), 10)
        self.assertEqual(payload["gates"][0]["name"], "X")
        self.assertEqual(payload["gates"][1]["name"], "H")
        self.assertEqual(payload["gates"][-1]["name"], "H")
        self.assertEqual(payload["organization_schedule"]["initial_state"], "s0")
        self.assertEqual(
            len(payload["organization_schedule"]["states"]),
            len(payload["gates"]) + 1,
        )
        self.assertTrue(
            all(
                len(unit["qubits"]) == 1
                for state in payload["organization_schedule"]["states"]
                for unit in state["units"]
            )
        )
        self.assertEqual(payload["assertion"]["threshold"], 0.999999)
        self.assertEqual(payload["assertion"]["target"]["type"], "bitwise_measurement_outcome")
        hidden_string = payload["assertion"]["target"]["outcome"]
        self.assertEqual(len(hidden_string), 9)
        self.assertEqual(sum(1 for bit in hidden_string if bit == "1"), 3)
        ones = [index for index, bit in enumerate(hidden_string) if bit == "1"]
        self.assertGreaterEqual(ones[0], 0)
        self.assertLess(ones[-1], len(hidden_string))
        self.assertLess(ones[0], len(hidden_string) // 3)
        self.assertGreaterEqual(ones[-1], 2 * len(hidden_string) // 3)

    def test_build_bv_family_payloads_covers_all_expected_sizes(self) -> None:
        payloads = build_bv_family_payloads()
        names = [payload["program_name"] for payload in payloads]

        self.assertEqual(
            names,
            [
                "bv_10_sparse",
                "bv_20_sparse",
                "bv_50_sparse",
                "bv_100_sparse",
                "bv_150_sparse",
                "bv_200_sparse",
            ],
        )

    def test_write_generated_bv_payload_round_trips_through_parser(self) -> None:
        payload = build_bv_payload(20)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bv_generated.qmodel"
            write_qmodel_payload(payload, path)
            spec = parse_qmodel_file(str(path))

        self.assertEqual(spec.program_name, "bv_20_sparse")
        self.assertEqual(spec.gates[0].name, "X")
        self.assertEqual(spec.gates[1].name, "H")
        self.assertEqual(spec.organization_schedule.states[0].name, "s0")
        self.assertEqual(len(spec.organization_schedule.states), len(payload["gates"]) + 1)
        self.assertEqual(spec.assertions[0].threshold, 0.999999)
        self.assertEqual(spec.assertions[0].target["type"], "bitwise_measurement_outcome")
        self.assertEqual(spec.assertions[0].target["outcome"], payload["assertion"]["target"]["outcome"])
        self.assertNotIn("outcomes", spec.assertions[0].target)

    def test_reject_bitwise_probability_assertion_without_outcome_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad_bitwise_target.qmodel"
            path.write_text(
                "\n".join(
                    [
                        "format: qmodel-v1",
                        "program_name: bad_bitwise_target",
                        "qubits: [q0]",
                        "initial_state: zero",
                        "gates: []",
                        "units: []",
                        "assertion:",
                        "  kind: probability",
                        "  target:",
                        "    type: bitwise_measurement_outcome",
                        "    scope: [q0]",
                        "  comparator: '='",
                        "  threshold: 1.0",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(Exception):
                parse_qmodel_file(str(path))

    def test_emit_bv_family_models_writes_six_files_in_tempdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            written_paths = emit_bv_family_models(tmpdir)
            self.assertEqual(len(written_paths), 6)
            self.assertEqual(
                [path.name for path in written_paths],
                [
                    "bv_10_sparse.qmodel",
                    "bv_20_sparse.qmodel",
                    "bv_50_sparse.qmodel",
                    "bv_100_sparse.qmodel",
                    "bv_150_sparse.qmodel",
                    "bv_200_sparse.qmodel",
                ],
            )
            for path in written_paths:
                self.assertTrue(path.exists(), path)
                spec = parse_qmodel_file(str(path))
                self.assertTrue(spec.program_name.startswith("bv_"))

    def test_generate_bv_models_script_path_can_be_invoked(self) -> None:
        script_path = Path("/home/li/project/QCE-2026/project_code/scripts/generate_bv_models.py")
        spec = importlib.util.spec_from_file_location("generate_bv_models", script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                result = module.main(tmpdir)

            self.assertEqual(result, 0)
            output = [line for line in buffer.getvalue().splitlines() if line.strip()]
            self.assertEqual(len(output), 6)
            self.assertTrue(all(line.endswith(".qmodel") for line in output))

            written_paths = sorted(Path(tmpdir).glob("*.qmodel"))
            self.assertEqual(len(written_paths), 6)

    def test_formal_bv_models_parse(self) -> None:
        base = Path(
            "/home/li/project/QCE-2026/project_code/experiment_data/models/BV"
        )
        paths = sorted(base.glob("*.qmodel"))
        self.assertEqual(len(paths), 6)
        for path in paths:
            spec = parse_qmodel_file(str(path))
            self.assertTrue(spec.program_name.startswith("bv_"))
            self.assertEqual(spec.metadata["family"], "bv")
            self.assertEqual(len(spec.assertions), 1)


class AIQFTGeneratorTests(unittest.TestCase):
    def test_build_aiqft_payload_for_n10_window5(self) -> None:
        payload = build_aiqft_payload(10, 5)

        self.assertEqual(payload["program_name"], "aiqft_10_w5")
        self.assertEqual(payload["metadata"]["family"], "aiqft")
        self.assertEqual(payload["metadata"]["n"], 10)
        self.assertEqual(payload["metadata"]["window_size"], 5)
        self.assertEqual(payload["assertion"]["target"]["type"], "bitwise_measurement_outcome")
        self.assertEqual(payload["assertion"]["threshold"], 0.997)
        self.assertEqual(len(payload["assertion"]["target"]["scope"]), 10)
        self.assertEqual(len(payload["assertion"]["target"]["outcome"]), 10)
        self.assertEqual(payload["organization_schedule"]["initial_state"], "s0")
        self.assertEqual(
            len(payload["organization_schedule"]["states"]),
            len(payload["gates"]) + 1,
        )
        self.assertEqual(payload["gates"][0]["name"], "H")
        self.assertIn(payload["gates"][-1]["name"], {"H", "CP"})

    def test_build_aiqft_family_payloads_covers_10_and_20(self) -> None:
        payloads = build_aiqft_family_payloads()
        names = [payload["program_name"] for payload in payloads]

        self.assertEqual(names, ["aiqft_10_w5", "aiqft_20_w5"])

    def test_write_generated_aiqft_payload_round_trips_through_parser(self) -> None:
        payload = build_aiqft_payload(20, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "aiqft_generated.qmodel"
            write_qmodel_payload(payload, path)
            spec = parse_qmodel_file(str(path))

        self.assertEqual(spec.program_name, "aiqft_20_w5")
        self.assertEqual(spec.metadata["family"], "aiqft")
        self.assertEqual(spec.metadata["window_size"], 5)
        self.assertEqual(spec.assertions[0].threshold, 0.997)
        self.assertEqual(spec.assertions[0].target["type"], "bitwise_measurement_outcome")
        self.assertEqual(len(spec.organization_schedule.states), len(payload["gates"]) + 1)

    def test_emit_aiqft_family_models_writes_two_files_in_tempdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            written_paths = emit_aiqft_family_models(tmpdir)
            self.assertEqual(len(written_paths), 2)
            self.assertEqual(
                [path.name for path in written_paths],
                ["aiqft_10_w5.qmodel", "aiqft_20_w5.qmodel"],
            )
            for path in written_paths:
                self.assertTrue(path.exists(), path)
                spec = parse_qmodel_file(str(path))
                self.assertTrue(spec.program_name.startswith("aiqft_"))

    def test_generate_aiqft_models_script_path_can_be_invoked(self) -> None:
        script_path = Path("/home/li/project/QCE-2026/project_code/scripts/generate_aiqft_models.py")
        spec = importlib.util.spec_from_file_location("generate_aiqft_models", script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                result = module.main(tmpdir)

            self.assertEqual(result, 0)
            output = [line for line in buffer.getvalue().splitlines() if line.strip()]
            self.assertEqual(len(output), 2)
            self.assertTrue(all(line.endswith(".qmodel") for line in output))

            written_paths = sorted(Path(tmpdir).glob("*.qmodel"))
            self.assertEqual(len(written_paths), 2)

    def test_formal_aiqft_models_parse(self) -> None:
        base = Path(
            "/home/li/project/QCE-2026/project_code/experiment_data/models/AIQFT"
        )
        paths = sorted(path for path in base.glob("*.qmodel") if path.name != ".gitkeep")
        self.assertEqual(len(paths), 2)
        for path in paths:
            spec = parse_qmodel_file(str(path))
            self.assertTrue(spec.program_name.startswith("aiqft_"))
            self.assertEqual(spec.metadata["family"], "aiqft")
            self.assertEqual(len(spec.assertions), 1)


if __name__ == "__main__":
    unittest.main()
