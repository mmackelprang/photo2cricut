"""
photo2cricut.pipeline
=====================
Core image -> single-line SVG pipeline.

JPG -> grayscale + contrast -> line extraction (XDoG / Canny)
    -> despeckle -> skeletonize (1-px centerlines)
    -> trace skeleton graph into polylines -> smoothing
    -> intermediate SVG -> vpype (merge/simplify/filter/sort, scale)
    -> rewrite root size in inches.

A Cricut in Draw mode is a slow pen plotter; centerlines (one stroke per line)
are what you want, NOT outlines (which double every line).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize
import sknw


# ----------------------------------------------------------------------------- config

@dataclass
class Options:
    method: str = "xdog"          # "xdog" | "canny"
    detail: float = 0.5           # 0..1 line density / sensitivity
    width_in: float = 8.0         # final drawing width in inches
    max_px: int = 2000            # downscale longest side before processing
    despeckle: int = 12           # drop connected blobs <= this many px
    smooth: int = 2               # smoothing passes (0 = off)
    simplify_mm: float = 0.3      # vpype linesimplify tolerance
    merge_mm: float = 1.0         # bridge stroke gaps up to this distance
    min_line_mm: float = 1.0      # drop plotted segments shorter than this
    xdog_sigma: float = 0.8       # XDoG base sigma; smaller = finer lines
    clahe_clip: float = 2.0       # local-contrast strength; raise for flat/washed-out photos
    clahe_tiles: int = 8          # CLAHE grid; smaller = more local contrast


# ----------------------------------------------------------------------------- line extraction

def xdog(gray, sigma=0.8, k=1.6, tau=0.975, phi=200.0, eps=-0.1):
    """Extended Difference-of-Gaussians -> near-binary 'drawn' line image.
    Returns float image in [0,1] where lines are dark (~0)."""
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)
    dog = g1 - tau * g2
    out = np.ones_like(dog)
    mask = dog < eps
    out[mask] = 1.0 + np.tanh(phi * (dog[mask] - eps))
    return np.clip(out, 0.0, 1.0)


def extract_lines(gray, method: str, detail: float, sigma: float = 0.8):
    """Return a boolean array where True == a line pixel (foreground)."""
    if method == "xdog":
        tau = 0.965 + 0.02 * detail
        xd = xdog(gray, sigma=sigma, k=1.6, tau=tau, phi=200.0, eps=-0.1)
        thr = 0.5 - 0.15 * (detail - 0.5)
        return xd < thr
    if method == "canny":
        lo = int(40 + (1 - detail) * 60)
        hi = int(lo * 2.2)
        return cv2.Canny(gray, lo, hi) > 0
    raise ValueError(f"unknown method: {method!r} (use 'xdog' or 'canny')")


def despeckle(lines_bool, min_px: int):
    """Drop connected line-blobs with area <= min_px."""
    if min_px <= 0:
        return lines_bool
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        lines_bool.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(lines_bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > min_px:
            keep[labels == i] = True
    return keep


# ----------------------------------------------------------------------------- vectorize

def smooth_polyline(pts, passes: int):
    """Moving-average smoothing; endpoints fixed."""
    if passes <= 0 or len(pts) < 3:
        return pts
    p = pts.astype(float).copy()
    for _ in range(passes):
        q = p.copy()
        q[1:-1] = (p[:-2] + p[1:-1] + p[2:]) / 3.0
        p = q
    return p


def skeleton_to_polylines(lines_bool, smooth_passes: int):
    """Skeletonize, build a graph, return (polylines[(N,2) x,y], (H,W))."""
    skel = skeletonize(lines_bool)
    graph = sknw.build_sknw(skel.astype(np.uint8), multi=True)
    polylines = []
    for s, e, key in graph.edges(keys=True):
        pts = graph[s][e][key]["pts"]               # (N,2) as (row=y, col=x)
        if len(pts) < 2:
            continue
        xy = np.column_stack([pts[:, 1], pts[:, 0]]).astype(float)
        polylines.append(smooth_polyline(xy, smooth_passes))
    return polylines, skel.shape


# ----------------------------------------------------------------------------- svg io

def write_intermediate_svg(polylines, shape_hw, path, stroke_w=1.0):
    h, w = shape_hw
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">',
        f'<g fill="none" stroke="#000000" stroke-width="{stroke_w}" '
        f'stroke-linecap="round" stroke-linejoin="round">',
    ]
    for xy in polylines:
        d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in xy)
        parts.append(f'<path d="{d}"/>')
    parts.append("</g></svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts))


def run_vpype(in_svg, out_svg, opt: Options):
    """Optimize for plotting and collapse structure so CDS imports cleanly.
    Invoked as a module so it does not depend on PATH (venv/Docker/Windows safe)."""
    cmd = [
        sys.executable, "-m", "vpype_cli",
        "read", in_svg,
        # homogeneous scaleto into a tall box => width is the binding dimension
        "scaleto", f"{opt.width_in}in", "1000in",
        "linemerge", "--tolerance", f"{opt.merge_mm}mm",
        "linesimplify", "--tolerance", f"{opt.simplify_mm}mm",
        "filter", "--min-length", f"{opt.min_line_mm}mm",
        "linesort",
        # tighten the page to the geometry bbox so the declared size == drawn art
        "layout", "tight",
        "write", out_svg,
    ]
    subprocess.run(cmd, check=True)


def force_inches(svg_path, width_in: float):
    """Rewrite root width/height in inches so Cricut sizes the art correctly."""
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ET.parse(svg_path)
    root = tree.getroot()
    vb = root.get("viewBox")
    if vb:
        _, _, vw, vh = (float(v) for v in vb.split())
        height_in = width_in * (vh / vw) if vw else width_in
        root.set("width", f"{width_in}in")
        root.set("height", f"{height_in:.4f}in")
    tree.write(svg_path, xml_declaration=True, encoding="utf-8")


# ----------------------------------------------------------------------------- public API

def convert(input_path: str, output_path: str, opt: Options | None = None) -> dict:
    """Convert a photo to a Cricut-ready single-line SVG.

    Returns a small dict of stats. Raises on unreadable input / no lines found.
    """
    opt = opt or Options()
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    h, w = img.shape[:2]
    scale = opt.max_px / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    gray = cv2.createCLAHE(clipLimit=opt.clahe_clip,
                           tileGridSize=(opt.clahe_tiles, opt.clahe_tiles)).apply(gray)

    lines = extract_lines(gray, opt.method, opt.detail, opt.xdog_sigma)
    lines = despeckle(lines, opt.despeckle)

    polylines, shape_hw = skeleton_to_polylines(lines, opt.smooth)
    if not polylines:
        raise RuntimeError("No lines found. Try a higher --detail or a higher-contrast photo.")

    with tempfile.TemporaryDirectory() as td:
        mid = os.path.join(td, "mid.svg")
        write_intermediate_svg(polylines, shape_hw, mid)
        run_vpype(mid, output_path, opt)
    force_inches(output_path, opt.width_in)

    return {"raw_strokes": len(polylines), "method": opt.method, "width_in": opt.width_in,
            "output": output_path}
