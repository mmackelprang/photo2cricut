"""Generate a synthetic photo-like portrait for testing the pipeline.

No real/copyrighted photo is bundled; this produces a deterministic stand-in
with gradients, shading, and soft edges so the full flow can be exercised.

Usage:
  photo2cricut-makeimg examples/test_portrait.jpg
  python scripts/make_test_image.py examples/test_portrait.jpg --seed 7
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np


def make_portrait(seed: int = 0):
    rng = np.random.default_rng(seed)
    H, W = 900, 720
    img = np.zeros((H, W), np.float32)
    xx = np.mgrid[0:H, 0:W][1]
    img += 120 + 40 * np.sin(xx / 220.0)

    cv2.ellipse(img, (360, 360), (230, 260), 0, 180, 360, 40, -1)   # hair
    cv2.ellipse(img, (360, 330), (210, 250), 0, 0, 360, 55, -1)
    cv2.ellipse(img, (360, 430), (165, 210), 0, 0, 360, 200, -1)    # face
    cv2.circle(img, (300, 470), 60, 185, -1)
    cv2.circle(img, (430, 470), 60, 185, -1)
    for ex in (300, 430):
        cv2.ellipse(img, (ex, 400), (38, 22), 0, 0, 360, 235, -1)   # eye white
        cv2.circle(img, (ex, 402), 13, 35, -1)                      # pupil
        cv2.ellipse(img, (ex, 378), (40, 12), 0, 180, 360, 70, -1)  # brow
    cv2.line(img, (360, 420), (352, 485), 150, 3)                   # nose
    cv2.ellipse(img, (360, 492), (22, 12), 0, 0, 180, 150, 2)
    cv2.ellipse(img, (360, 545), (55, 26), 0, 10, 170, 90, 4)       # mouth
    cv2.rectangle(img, (315, 615), (405, 680), 190, -1)             # neck
    cv2.ellipse(img, (360, 700), (250, 90), 0, 180, 360, 150, -1)   # shoulders

    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    img += rng.normal(0, 6, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate a synthetic test portrait JPG.")
    ap.add_argument("output", nargs="?", default="examples/test_portrait.jpg")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    img = make_portrait(args.seed)
    if not cv2.imwrite(args.output, img, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        print(f"error: could not write {args.output}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}  {img.shape[1]}x{img.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
