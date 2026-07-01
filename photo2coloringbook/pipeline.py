"""Orchestrate: folder of photos -> US-Letter PDF coloring book."""

from __future__ import annotations

from dataclasses import dataclass

from .book import build_pdf
from .ingest import load_images
from .isolate import isolate
from .layout import fit_canvas, make_cover
from .post import clean
from .stylize import get_stylizer

LETTER_IN = (8.5, 11.0)


@dataclass
class BookOptions:
    title: str = "Coloring Book"
    paper: str = "letter"
    dpi: int = 300
    margin_in: float = 0.5
    bg: str = "keep"        # keep | auto | remove   (auto/remove: Phase 2)
    backend: str = "cv"     # cv | contour           (contour: Phase 2)
    line_weight: int = 3
    ink_level: int = 110    # contour-only: higher = less ink (drops soft shading)


def convert_book(input_dir: str, output_pdf: str, opt: BookOptions | None = None) -> dict:
    opt = opt or BookOptions()
    if opt.paper != "letter":
        # Guard the hardcoded LETTER_IN below: other sizes (A4, ...) arrive in
        # Phase 3 and must route real dimensions, not silently produce Letter.
        raise ValueError(f"paper={opt.paper!r} not yet supported; use 'letter'")
    page_w = int(LETTER_IN[0] * opt.dpi)
    page_h = int(LETTER_IN[1] * opt.dpi)
    margin = int(opt.margin_in * opt.dpi)

    stylizer = get_stylizer(opt.backend, ink_level=opt.ink_level)
    images = load_images(input_dir)

    pages = [make_cover(opt.title, page_w, page_h)]
    for img in images:
        subject = isolate(img.bgr, opt.bg)
        lineart = stylizer.to_lineart(subject)
        cleaned = clean(lineart, line_weight=opt.line_weight)
        pages.append(fit_canvas(cleaned, page_w, page_h, margin))

    build_pdf(pages, output_pdf)
    return {"pages": len(pages), "backend": opt.backend, "output": output_pdf}
