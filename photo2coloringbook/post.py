"""Coloring-book cleanup: bold, closed, despeckled black-on-white line art."""

from __future__ import annotations

import cv2
import numpy as np


def clean(lineart_gray: np.ndarray, line_weight: int = 3, despeckle: int = 12) -> np.ndarray:
    """Grayscale black-on-white in/out. Binarize, close gaps, despeckle, thicken."""
    # lines -> 255 mask (THRESH_BINARY_INV: dark pixels become 255)
    _, mask = cv2.threshold(lineart_gray, 127, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if despeckle > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] > despeckle:
                keep[labels == i] = 255
        mask = keep

    if line_weight > 1:
        tk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_weight, line_weight))
        mask = cv2.dilate(mask, tk)

    return cv2.bitwise_not(mask)  # back to black-on-white
