"""Subject isolation. Phase 1: 'keep' only; rembg-backed modes land in Phase 2."""

from __future__ import annotations

import numpy as np


def isolate(bgr: np.ndarray, mode: str = "keep") -> np.ndarray:
    if mode == "keep":
        return bgr
    if mode in ("auto", "remove"):
        raise NotImplementedError(
            f"--bg {mode} requires the [gpu] extra (rembg) and arrives in Phase 2; "
            "use --bg keep for now")
    raise ValueError(f"unknown bg mode: {mode!r} (use 'keep', 'auto', or 'remove')")
