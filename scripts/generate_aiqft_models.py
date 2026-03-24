from __future__ import annotations

import sys
from pathlib import Path

from qmodel.benchmarks import emit_aiqft_family_models


def main(output_dir: str | None = None) -> int:
    target = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parents[1] / "experiment_data" / "models" / "AIQFT"
    written_paths = emit_aiqft_family_models(target)
    for path in written_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
