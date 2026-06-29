"""Stylizer factory."""

from __future__ import annotations

from .base import Stylizer
from .cv import CvStylizer

__all__ = ["Stylizer", "CvStylizer", "get_stylizer"]


def get_stylizer(name: str) -> Stylizer:
    if name == "cv":
        return CvStylizer()
    if name == "contour":
        try:
            from .contour import ContourStylizer
        except ImportError as e:
            raise RuntimeError(
                "the 'contour' backend needs the [gpu] extra "
                "(pip install '.[gpu]': torch, controlnet_aux). "
                f"Underlying import error: {e}") from e
        return ContourStylizer()
    raise ValueError(f"unknown backend: {name!r} (use 'cv' or 'contour')")
