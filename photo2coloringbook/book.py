"""Assemble page rasters into a single US-Letter PDF."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdfcanvas


def build_pdf(page_rasters: list[np.ndarray], out_path: str) -> None:
    """One Letter page per raster (each stretched to fill the page)."""
    if not page_rasters:
        raise ValueError("no pages to write")
    page_w_pt, page_h_pt = letter  # 612 x 792 pt
    c = pdfcanvas.Canvas(out_path, pagesize=letter)
    for raster in page_rasters:
        buf = io.BytesIO()
        Image.fromarray(raster).convert("L").save(buf, format="PNG")
        buf.seek(0)
        c.drawImage(ImageReader(buf), 0, 0, width=page_w_pt, height=page_h_pt)
        c.showPage()
    c.save()
