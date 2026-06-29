"""Load and EXIF-orient images from a directory."""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class LoadedImage:
    path: str
    bgr: np.ndarray  # H x W x 3, BGR uint8


def load_images(directory: str) -> list[LoadedImage]:
    """Load every image in `directory`, EXIF-oriented, sorted by filename.

    Skips unreadable files with a warning. Raises RuntimeError if none load.
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory}")
    out: list[LoadedImage] = []
    for name in sorted(os.listdir(directory)):
        if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
            continue
        path = os.path.join(directory, name)
        try:
            pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        except Exception as e:  # corrupt/unreadable
            print(f"warning: skipping unreadable image {path}: {e}")
            continue
        bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        out.append(LoadedImage(path=path, bgr=bgr))
    if not out:
        raise RuntimeError(f"No usable images found in {directory}")
    return out
