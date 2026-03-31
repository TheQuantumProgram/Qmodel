from __future__ import annotations

import unittest
from pathlib import Path


class QuickstartCliTests(unittest.TestCase):
    def test_run_subcommand_uses_expected_defaults(self) -> None:
        from scripts.qmodel_cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", "tests/models/clifford_bell.qmodel"])

        self.assertEqual(args.command, "run")
        self.assertEqual(args.model, "tests/models/clifford_bell.qmodel")
        self.assertFalse(args.run_concrete)
        self.assertEqual(args.mode, "trusted")

    def test_run_all_subcommand_accepts_family_filter(self) -> None:
        from scripts.qmodel_cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["run-all", "--family", "GHZ", "--mode", "checked"])

        self.assertEqual(args.command, "run-all")
        self.assertEqual(args.family, "GHZ")
        self.assertEqual(args.mode, "checked")
        self.assertFalse(args.run_concrete)

    def test_discover_models_filters_by_family_directory(self) -> None:
        from scripts.run_benchmarks import discover_model_paths

        models_root = Path(__file__).resolve().parents[1] / "experiment_data" / "models"
        discovered = discover_model_paths(models_root, family="GHZ")

        self.assertTrue(discovered)
        self.assertTrue(all(path.suffix == ".qmodel" for path in discovered))
        self.assertTrue(all(path.parent.name == "GHZ" for path in discovered))
        self.assertEqual(discovered, sorted(discovered))


if __name__ == "__main__":
    unittest.main()
