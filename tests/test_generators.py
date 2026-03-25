from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qmodel.benchmarks.generators import (
    ADDER_STANDARD_SIZES,
    CUSTOM_MODEL_NAMES,
    build_adder_payload,
    build_adder_family_payloads,
    build_aiqft_family_payloads,
    build_aiqft_payload,
    build_bv_family_payloads,
    build_bv_payload,
    build_custom_family_payloads,
    build_grover_payload,
    build_ghz_family_payloads,
    build_ghz_staircase_payload,
    emit_adder_family_models,
    emit_aiqft_family_models,
    emit_bv_family_models,
    emit_custom_family_models,
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

    def test_write_qmodel_payload_uses_inline_top_level_qubits_without_yaml_aliases(self) -> None:
        payload = build_bv_payload(10)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bv_generated.qmodel"
            write_qmodel_payload(payload, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("qubits: [q0, q1, q2, q3, q4, q5, q6, q7, q8, q9]", text)
        self.assertNotIn("&id", text)
        self.assertNotIn("*id", text)

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

    def test_build_aiqft_family_payloads_covers_10_20_50_100_150_200(self) -> None:
        payloads = build_aiqft_family_payloads()
        names = [payload["program_name"] for payload in payloads]

        self.assertEqual(
            names,
            [
                "aiqft_10_w5",
                "aiqft_20_w5",
                "aiqft_50_w5",
                "aiqft_100_w5",
                "aiqft_150_w5",
                "aiqft_200_w5",
            ],
        )

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

    def test_emit_aiqft_family_models_writes_six_files_in_tempdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            written_paths = emit_aiqft_family_models(tmpdir)
            self.assertEqual(len(written_paths), 6)
            self.assertEqual(
                [path.name for path in written_paths],
                [
                    "aiqft_10_w5.qmodel",
                    "aiqft_20_w5.qmodel",
                    "aiqft_50_w5.qmodel",
                    "aiqft_100_w5.qmodel",
                    "aiqft_150_w5.qmodel",
                    "aiqft_200_w5.qmodel",
                ],
            )
            for path in written_paths:
                self.assertTrue(path.exists(), path)
                spec = parse_qmodel_file(str(path))
                self.assertTrue(spec.program_name.startswith("aiqft_"))

    def test_formal_aiqft_models_parse(self) -> None:
        base = Path(
            "/home/li/project/QCE-2026/project_code/experiment_data/models/AIQFT"
        )
        paths = sorted(path for path in base.glob("*.qmodel") if path.name != ".gitkeep")
        self.assertEqual(len(paths), 6)
        for path in paths:
            spec = parse_qmodel_file(str(path))
            self.assertTrue(spec.program_name.startswith("aiqft_"))
            self.assertEqual(spec.metadata["family"], "aiqft")
            self.assertEqual(len(spec.assertions), 1)


class AdderGeneratorTests(unittest.TestCase):
    def test_build_adder_payload_for_n10_uses_ccx_ripple_carry_structure(self) -> None:
        payload = build_adder_payload(10)

        self.assertEqual(payload["program_name"], "adder_10")
        self.assertEqual(payload["metadata"]["family"], "adder")
        self.assertEqual(payload["metadata"]["n"], 10)
        self.assertEqual(payload["metadata"]["register_bits"], 4)
        self.assertEqual(len(payload["qubits"]), 10)
        self.assertEqual(payload["gates"][0]["name"], "X")
        self.assertIn("CCX", {gate["name"] for gate in payload["gates"]})
        self.assertEqual(payload["organization_schedule"]["initial_state"], "s0")
        self.assertEqual(
            len(payload["organization_schedule"]["states"]),
            len(payload["gates"]) + 1,
        )
        self.assertTrue(
            all(len(unit["qubits"]) == 1 for unit in payload["organization_schedule"]["states"][0]["units"])
        )
        self.assertTrue(
            all(len(unit["qubits"]) == 1 for unit in payload["organization_schedule"]["states"][-1]["units"])
        )
        middle_unit_sizes = [
            len(unit["qubits"])
            for state in payload["organization_schedule"]["states"][1:-1]
            for unit in state["units"]
        ]
        self.assertIn(2, middle_unit_sizes)
        self.assertIn(3, middle_unit_sizes)
        self.assertEqual(payload["assertion"]["target"]["type"], "bitwise_measurement_outcome")
        self.assertEqual(payload["assertion"]["comparator"], ">=")
        self.assertEqual(payload["assertion"]["threshold"], 0.999999)
        self.assertEqual(len(payload["assertion"]["target"]["scope"]), 10)
        self.assertEqual(len(payload["assertion"]["target"]["outcome"]), 10)

    def test_write_generated_adder_payload_round_trips_through_parser(self) -> None:
        payload = build_adder_payload(50)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "adder_generated.qmodel"
            write_qmodel_payload(payload, path)
            spec = parse_qmodel_file(str(path))

        self.assertEqual(spec.program_name, "adder_50")
        self.assertEqual(spec.metadata["family"], "adder")
        self.assertEqual(spec.metadata["register_bits"], 24)
        self.assertEqual(spec.assertions[0].target["type"], "bitwise_measurement_outcome")
        self.assertEqual(spec.assertions[0].threshold, 0.999999)
        self.assertEqual(len(spec.organization_schedule.states), len(payload["gates"]) + 1)

    def test_build_adder_family_payloads_covers_all_expected_sizes(self) -> None:
        payloads = build_adder_family_payloads()
        names = [payload["program_name"] for payload in payloads]

        self.assertEqual(ADDER_STANDARD_SIZES, (10, 20, 50, 100, 150, 200))
        self.assertEqual(
            names,
            [
                "adder_10",
                "adder_20",
                "adder_50",
                "adder_100",
                "adder_150",
                "adder_200",
            ],
        )

    def test_emit_adder_family_models_writes_six_files_in_tempdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            written_paths = emit_adder_family_models(tmpdir)
            self.assertEqual(len(written_paths), 6)
            self.assertEqual(
                [path.name for path in written_paths],
                [
                    "adder_10.qmodel",
                    "adder_20.qmodel",
                    "adder_50.qmodel",
                    "adder_100.qmodel",
                    "adder_150.qmodel",
                    "adder_200.qmodel",
                ],
            )
            for path in written_paths:
                self.assertTrue(path.exists(), path)
                spec = parse_qmodel_file(str(path))
                self.assertTrue(spec.program_name.startswith("adder_"))

    def test_formal_adder_models_parse(self) -> None:
        base = Path(
            "/home/li/project/QCE-2026/project_code/experiment_data/models/Adder"
        )
        paths = sorted(
            (path for path in base.glob("*.qmodel") if path.name != ".gitkeep"),
            key=lambda path: int(path.stem.split("_")[1]),
        )
        self.assertEqual(
            [path.name for path in paths],
            [
                "adder_10.qmodel",
                "adder_20.qmodel",
                "adder_50.qmodel",
                "adder_100.qmodel",
                "adder_150.qmodel",
                "adder_200.qmodel",
            ],
        )
        for path in paths:
            spec = parse_qmodel_file(str(path))
            self.assertTrue(spec.program_name.startswith("adder_"))
            self.assertEqual(spec.metadata["family"], "adder")
            self.assertEqual(len(spec.assertions), 1)


class GroverGeneratorTests(unittest.TestCase):
    def test_build_grover_payload_for_n10_uses_tree_structured_controls(self) -> None:
        payload = build_grover_payload(10)

        self.assertEqual(payload["program_name"], "grover_10")
        self.assertEqual(payload["metadata"]["family"], "grover")
        self.assertEqual(payload["metadata"]["n"], 10)
        self.assertEqual(payload["metadata"]["search_bits"], 5)
        self.assertEqual(payload["metadata"]["mark_bits"], 2)
        self.assertEqual(payload["metadata"]["iterations"], 1)
        self.assertEqual(len(payload["qubits"]), 10)
        self.assertIn("H", {gate["name"] for gate in payload["gates"]})
        self.assertIn("X", {gate["name"] for gate in payload["gates"]})
        self.assertIn("CCX", {gate["name"] for gate in payload["gates"]})
        self.assertEqual(payload["assertion"]["target"]["type"], "measurement_outcome")
        self.assertEqual(payload["assertion"]["target"]["scope"], ["q0", "q1"])
        self.assertEqual(payload["assertion"]["target"]["outcomes"], ["11"])
        self.assertEqual(
            len(payload["organization_schedule"]["states"]),
            len(payload["gates"]) + 1,
        )

    def test_write_generated_grover_payload_round_trips_through_parser(self) -> None:
        payload = build_grover_payload(10)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "grover_generated.qmodel"
            write_qmodel_payload(payload, path)
            spec = parse_qmodel_file(str(path))

        self.assertEqual(spec.program_name, "grover_10")
        self.assertEqual(spec.metadata["family"], "grover")
        self.assertEqual(spec.metadata["search_bits"], 5)
        self.assertEqual(spec.metadata["mark_bits"], 2)
        self.assertEqual(spec.metadata["iterations"], 1)
        self.assertEqual(spec.assertions[0].target["type"], "measurement_outcome")
        self.assertEqual(spec.assertions[0].target["scope"], ["q0", "q1"])
        self.assertEqual(spec.assertions[0].target["outcomes"], ["11"])
        self.assertEqual(len(spec.organization_schedule.states), len(payload["gates"]) + 1)

    def test_build_grover_payload_for_n20_keeps_core_unit_bounded(self) -> None:
        payload = build_grover_payload(20)

        self.assertEqual(payload["program_name"], "grover_20")
        self.assertEqual(payload["metadata"]["search_bits"], 10)
        self.assertEqual(payload["metadata"]["mark_bits"], 3)
        self.assertEqual(payload["metadata"]["iterations"], 2)
        max_unit_width = max(
            len(unit["qubits"])
            for state in payload["organization_schedule"]["states"]
            for unit in state["units"]
        )
        self.assertLessEqual(max_unit_width, 5)

    def test_build_grover_payload_for_n50_keeps_core_unit_bounded(self) -> None:
        payload = build_grover_payload(50)

        self.assertEqual(payload["program_name"], "grover_50")
        self.assertEqual(payload["metadata"]["search_bits"], 25)
        self.assertEqual(payload["metadata"]["mark_bits"], 4)
        self.assertEqual(payload["metadata"]["iterations"], 3)
        max_unit_width = max(
            len(unit["qubits"])
            for state in payload["organization_schedule"]["states"]
            for unit in state["units"]
        )
        self.assertLessEqual(max_unit_width, 7)

    def test_build_grover_payload_for_n100_and_n200_keep_core_unit_bounded(self) -> None:
        expected = {
            100: (50, 5, 4, 9),
            150: (75, 6, 6, 11),
            200: (100, 6, 6, 11),
        }
        for n, (search_bits, mark_bits, iterations, max_width) in expected.items():
            payload = build_grover_payload(n)

            self.assertEqual(payload["program_name"], f"grover_{n}")
            self.assertEqual(payload["metadata"]["search_bits"], search_bits)
            self.assertEqual(payload["metadata"]["mark_bits"], mark_bits)
            self.assertEqual(payload["metadata"]["iterations"], iterations)
            max_unit_width = max(
                len(unit["qubits"])
                for state in payload["organization_schedule"]["states"]
                for unit in state["units"]
            )
            self.assertLessEqual(max_unit_width, max_width)

    def test_formal_grover_model_parses(self) -> None:
        expected = {
            "grover_10.qmodel": ("grover_10", 5, 2, 1, ["q0", "q1"], "11"),
            "grover_20.qmodel": ("grover_20", 10, 3, 2, ["q0", "q1", "q2"], "111"),
            "grover_50.qmodel": ("grover_50", 25, 4, 3, ["q0", "q1", "q2", "q3"], "1111"),
            "grover_100.qmodel": ("grover_100", 50, 5, 4, ["q0", "q1", "q2", "q3", "q4"], "11111"),
            "grover_150.qmodel": ("grover_150", 75, 6, 6, ["q0", "q1", "q2", "q3", "q4", "q5"], "111111"),
            "grover_200.qmodel": ("grover_200", 100, 6, 6, ["q0", "q1", "q2", "q3", "q4", "q5"], "111111"),
        }
        base = Path(
            "/home/li/project/QCE-2026/project_code/experiment_data/models/Grover"
        )

        for filename, (
            program_name,
            search_bits,
            mark_bits,
            iterations,
            scope,
            outcome,
        ) in expected.items():
            spec = parse_qmodel_file(str(base / filename))

            self.assertEqual(spec.program_name, program_name)
            self.assertEqual(spec.metadata["family"], "grover")
            self.assertEqual(spec.metadata["search_bits"], search_bits)
            self.assertEqual(spec.metadata["mark_bits"], mark_bits)
            self.assertEqual(spec.metadata["iterations"], iterations)
            self.assertEqual(spec.assertions[0].target["type"], "measurement_outcome")
            self.assertEqual(spec.assertions[0].target["scope"], scope)
            self.assertEqual(spec.assertions[0].target["outcomes"], [outcome])


class CustomGeneratorTests(unittest.TestCase):
    def test_build_custom_family_payloads_matches_expected_order_and_assertions(self) -> None:
        payloads = build_custom_family_payloads()
        names = [payload["program_name"] for payload in payloads]
        assertion_kinds = [payload["assertion"]["kind"] for payload in payloads]
        expected_judgments = [payload["metadata"]["expected_judgment"] for payload in payloads]

        self.assertEqual(names, list(CUSTOM_MODEL_NAMES))
        self.assertEqual(
            assertion_kinds,
            [
                "probability",
                "probability",
                "probability",
                "reachability",
                "reachability",
                "probability",
                "probability",
                "probability",
                "probability",
                "reachability",
                "reachability",
                "probability",
                "probability",
                "probability",
                "reachability",
                "probability",
            ],
        )
        self.assertEqual(
            expected_judgments,
            [
                "satisfied",
                "satisfied",
                "satisfied",
                "satisfied",
                "satisfied",
                "satisfied",
                "violated",
                "violated",
                "violated",
                "violated",
                "violated",
                "satisfied",
                "satisfied",
                "satisfied",
                "satisfied",
                "satisfied",
            ],
        )
        self.assertEqual(payloads[0]["assertion"]["target"]["type"], "measurement_outcome")
        self.assertEqual(payloads[3]["assertion"]["target"]["type"], "basis_state")

    def test_back_edge_and_disconnected_custom_models_have_expected_structure(self) -> None:
        payloads = {payload["program_name"]: payload for payload in build_custom_family_payloads()}

        back_edge = payloads["custom_back_edge_prob_6"]
        final_units = back_edge["organization_schedule"]["states"][-1]["units"]
        self.assertEqual([unit["name"] for unit in final_units], ["u012", "u234", "u45", "u15"])
        self.assertEqual(back_edge["assertion"]["target"]["outcomes"], ["111111"])

        disconnected = payloads["custom_disconnected_product_prob_10"]
        self.assertEqual(disconnected["assertion"]["threshold"], 0.125)
        self.assertEqual(
            [unit["name"] for unit in disconnected["organization_schedule"]["states"][-1]["units"]],
            ["u012", "u567", "u3", "u4", "u8", "u9"],
        )
        self.assertEqual(payloads["custom_overlap_chain_counter_6"]["metadata"]["expected_judgment"], "violated")
        self.assertEqual(payloads["custom_ccx_ladder_reach_15"]["metadata"]["expected_judgment"], "satisfied")

    def test_emit_custom_family_models_writes_all_files_in_tempdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            written_paths = emit_custom_family_models(tmpdir)
            self.assertEqual(len(written_paths), len(CUSTOM_MODEL_NAMES))
            self.assertEqual([path.stem for path in written_paths], list(CUSTOM_MODEL_NAMES))
            for path in written_paths:
                self.assertTrue(path.exists(), path)
                spec = parse_qmodel_file(str(path))
                self.assertEqual(spec.metadata["family"], "custom")

    def test_formal_custom_models_parse(self) -> None:
        base = Path(
            "/home/li/project/QCE-2026/project_code/experiment_data/models/Custom"
        )
        reachability_models = {
            "custom_ccx_ladder_reach_9",
            "custom_uncompute_reach_8",
            "custom_ccx_ladder_counter_9",
            "custom_uncompute_counter_8",
            "custom_ccx_ladder_reach_15",
        }
        paths = sorted(
            (path for path in base.glob("*.qmodel") if path.name != ".gitkeep"),
            key=lambda path: list(CUSTOM_MODEL_NAMES).index(path.stem),
        )
        self.assertEqual([path.stem for path in paths], list(CUSTOM_MODEL_NAMES))

        for path in paths:
            spec = parse_qmodel_file(str(path))
            self.assertEqual(spec.metadata["family"], "custom")
            self.assertEqual(len(spec.assertions), 1)
            if path.stem in reachability_models:
                self.assertEqual(spec.assertions[0].kind, "reachability")
            else:
                self.assertEqual(spec.assertions[0].kind, "probability")


if __name__ == "__main__":
    unittest.main()
