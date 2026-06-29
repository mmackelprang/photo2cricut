"""Compose line art onto US-Letter page rasters, plus a title cover."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def fit_canvas(lineart_gray: np.ndarray, page_w: int, page_h: int, margin: int) -> np.ndarray:
    """Center `lineart_gray` on a white page, scaled to fit within margins."""
    canvas = np.full((page_h, page_w), 255, np.uint8)
    box_w, box_h = page_w - 2 * margin, page_h - 2 * margin
    h, w = lineart_gray.shape[:2]
    scale = min(box_w / w, box_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = np.asarray(
        Image.fromarray(lineart_gray).resize((new_w, new_h), Image.Resampling.LANCZOS))
    x, y = (page_w - new_w) // 2, (page_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def make_cover(title: str, page_w: int, page_h: int) -> np.ndarray:
    """White page with `title` centered."""
    img = Image.new("L", (page_w, page_h), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(24, page_w // 14))
    bbox = draw.textbbox((0, 0), title, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((page_w - tw) / 2, (page_h - th) / 2), title, fill=0, font=font)
    return np.asarray(img)
