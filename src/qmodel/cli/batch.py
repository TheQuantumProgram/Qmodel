"""Run batches of qmodel files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

from .run_single import run_model


def discover_model_paths(models_root: Path, family: str | None = None) -> list[Path]:
    root = models_root.resolve()
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
