"""Subject isolation. 'keep' is a no-op; 'remove'/'auto' use rembg (u2net)."""

from __future__ import annotations

import cv2
import numpy as np

_session = None  # cached rembg u2net session (load once)


def _get_session():
    global _session
    if _session is None:
        try:
            from rembg import new_session
        except ImportError as e:
            raise RuntimeError(
                "background removal needs the [gpu] extra "
                "(pip install '.[gpu]': rembg, onnxruntime). "
                f"Underlying import error: {e}") from e
        _session = new_session("u2net")
    return _session


def _remove_to_white(bgr: np.ndarray) -> np.ndarray:
    from PIL import Image
    sess = _get_session()       # raises the clean [gpu]-extra RuntimeError if rembg missing
    from rembg import remove    # safe: _get_session imported rembg successfully
    rgba = remove(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), session=sess)
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(white, rgba).convert("RGB")
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def _has_face(bgr: np.ndarray) -> bool:
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(60, 60))
    return len(faces) > 0


def isolate(bgr: np.ndarray, mode: str = "keep") -> np.ndarray:
    if mode == "keep":
        return bgr
    if mode == "remove":
        return _remove_to_white(bgr)
    if mode == "auto":
        return _remove_to_white(bgr) if _has_face(bgr) else bgr
    raise ValueError(f"unknown bg mode: {mode!r} (use 'keep', 'auto', or 'remove')")
