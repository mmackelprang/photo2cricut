"""OpenCV adaptive-threshold line-art backend (no model/GPU)."""

from __future__ import annotations

import cv2
import numpy as np


class CvStylizer:
    def __init__(self, blur: int = 5, block_size: int = 9, c: int = 7):
        self.blur = blur
        self.block_size = block_size  # odd, >= 3
        self.c = c

    def to_lineart(self, bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, self.blur)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            self.block_size, self.c)
