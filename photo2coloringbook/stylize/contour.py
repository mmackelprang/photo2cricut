"""Neural line-art backend: controlnet_aux LineartDetector.

LineartDetector loads carolineec's "Informative Drawings" generator weights
(sk_model*.pth from the lllyasviel/Annotators repo). The model emits a soft
grayscale drawing with WHITE lines on BLACK; we invert + binarize to the
project's black-on-white uint8 convention so post.clean() keeps the strokes.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from controlnet_aux import LineartDetector


class ContourStylizer:
    def __init__(self, device: str | None = None, ink_level: int = 65,
                 max_resolution: int = 2048):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # load once; reused across every image in a convert_book run
        self.det = LineartDetector.from_pretrained("lllyasviel/Annotators").to(self.device)
        self.ink_level = ink_level            # model brightness >= this -> black ink
        self.max_resolution = max_resolution  # short-side cap (6 GB VRAM ceiling)

    def to_lineart(self, bgr: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = min(min(pil.size), self.max_resolution)
        out = self.det(pil, detect_resolution=res, image_resolution=res, coarse=False)
        model = np.asarray(out)[..., 0]       # white lines on black, 0..~230
        # invert + binarize so clean()'s 127 threshold keeps the (mid-gray) strokes
        return np.where(model >= self.ink_level, 0, 255).astype(np.uint8)
