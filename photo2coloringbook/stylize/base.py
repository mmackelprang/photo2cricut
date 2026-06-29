"""Stylizer interface: photo -> line art."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Stylizer(Protocol):
    def to_lineart(self, bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 photo -> grayscale uint8 line art (black lines=0, white=255)."""
        ...
