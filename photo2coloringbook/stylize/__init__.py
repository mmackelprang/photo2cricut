"""Stylizer factory."""

from __future__ import annotations

from .base import Stylizer
from .cv import CvStylizer

__all__ = ["Stylizer", "CvStylizer", "get_stylizer"]


def get_stylizer(name: str) -> Stylizer:
    if name == "cv":
        return CvStylizer()
    if name == "contour":
        raise NotImplementedError(
            "the 'contour' backend requires the [gpu] extra and arrives in Phase 2; "
            "use --backend cv for now")
    raise ValueError(f"unknown backend: {name!r} (use 'cv' or 'contour')")
