"""Command-line entry points for qmodel."""

from .main import build_parser, main
from .run_single import run_model

__all__ = ["build_parser", "main", "run_model"]
