"""Run benchmark batches and write raw results."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_single import run_model


_DEFAULT_MODELS_ROOT = Path(__file__).resolve().parents[1] / "experiment_data" / "models"


def discover_model_paths(models_root: Path | None = None, family: str | None = None) -> list[Path]:
    root = (models_root or _DEFAULT_MODELS_ROOT).resolve()
    candidates = sorted(path.resolve() for path in root.rglob("*.qmodel"))
    if family is None:
        return candidates
    family_key = family.lower()
    return [path for path in candidates if path.parent.name.lower() == family_key]


def iter_model_results(
    model_paths: Iterable[Path],
    *,
    run_concrete: bool = False,
    mode: str = "trusted",
) -> Iterator[dict[str, Any]]:
    for model_path in model_paths:
        yield run_model(model_path, run_concrete=run_concrete, mode=mode)


def run_models(
    model_paths: Iterable[Path],
    *,
    run_concrete: bool = False,
    mode: str = "trusted",
) -> list[dict[str, Any]]:
    return list(iter_model_results(model_paths, run_concrete=run_concrete, mode=mode))
