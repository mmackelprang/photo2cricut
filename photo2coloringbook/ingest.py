"""Load and EXIF-orient images from a directory."""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


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
            opened = ImageOps.exif_transpose(Image.open(path))
            if opened.mode in ("RGBA", "LA", "PA"):
                # Composite onto white instead of letting .convert("RGB") drop
                # alpha — transparent pixels usually carry black RGB, which would
                # otherwise become noise through the threshold.
                bg = Image.new("RGBA", opened.size, (255, 255, 255, 255))
                bg.alpha_composite(opened.convert("RGBA"))
                pil = bg.convert("RGB")
            else:
                pil = opened.convert("RGB")
        except Exception as e:  # corrupt/unreadable
            print(f"warning: skipping unreadable image {path}: {e}")
            continue
        bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        out.append(LoadedImage(path=path, bgr=bgr))
    if not out:
        raise RuntimeError(f"No usable images found in {directory}")
    return out
