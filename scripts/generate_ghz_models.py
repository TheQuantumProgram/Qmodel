"""Generate the formal GHZ-family experiment models."""

from __future__ import annotations

from pathlib import Path

from qmodel.benchmarks import emit_ghz_family_models


def main() -> int:
    output_dir = (
        Path(__file__).resolve().parents[1] / "experiment_data" / "models" / "GHZ"
    )
    written_paths = emit_ghz_family_models(output_dir)
    for path in written_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
