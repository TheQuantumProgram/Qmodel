"""Generate the formal BV-family experiment models."""

from __future__ import annotations

from pathlib import Path

from qmodel.benchmarks import emit_bv_family_models


def main(output_dir: str | Path | None = None) -> int:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "experiment_data" / "models" / "BV"
    written_paths = emit_bv_family_models(output_dir)
    for path in written_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
