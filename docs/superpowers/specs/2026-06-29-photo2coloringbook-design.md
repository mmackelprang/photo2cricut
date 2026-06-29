# photo2coloringbook — Design Spec

**Date:** 2026-06-29
**Status:** Approved (design) — pending implementation plan
**Author:** brainstormed with Mark

## 1. Goal

Turn a folder of family photos into a single **print-ready US-Letter (8.5×11") PDF coloring book**: bold, clean **black-on-white line art** that a young child can color inside, with **recognizable likeness** of the family members in the photos. The coloring book itself is the deliverable; text on pages is strictly secondary.

This is a **new pipeline**, not a change to `photo2cricut`. photo2cricut produces *centerlines* (one pen stroke per line, for a plotter); a coloring book needs the opposite — **bold closed outlines** enclosing large fillable regions, rasterized for print. The two tools share a repo and scaffolding but nothing in the image pipeline.

## 2. Context & constraints

- **Likeness is the top priority.** Favor methods that bind output tightly to the source photo (line *extraction*) over generative redrawing that can soften a specific face.
- **Cloud is allowed but not needed.** Family photos of children — local processing is preferred for privacy; the chosen path runs locally.
- **Execution host: `appserver`** (Ubuntu 24.04, NVIDIA GTX 1660 SUPER **6 GB VRAM**, Python 3.12, 16 cores, 31 GB RAM, 560 GB free; `docker`, `ollama`, `git` present; no ML libs yet). The 6 GB VRAM is the binding constraint — the chosen likeness backend runs in <2 GB, with ample headroom.
- The Windows dev box has no GPU; the tool runs on appserver. Development happens in this repo; the tool is deployed and executed on the Linux box.

## 3. Packaging & repository

- **Same repo, separate package** `photo2coloringbook/` with its own console entry point `photo2coloringbook` and its own optional-dependency groups, so `photo2cricut` stays lean (no torch dragged into it).
- **Dependency groups** (keep the test path light):
  - Base (`cv` backend + layout + PDF): `opencv-python-headless`, `numpy`, `Pillow`, `reportlab`.
  - `[gpu]` (the `contour` backend): `torch`, `torchvision`, `rembg`, `onnxruntime` (installed on appserver, cu121 wheels).
  - `[dev]`: `pytest`, `pypdf` (page-count assertions).
- The repo is brought under **git** (init `main`) as part of this work, so implementation can proceed branch-per-change per workflow policy. Large local sample media and `examples/uat/` scratch are git-ignored.

## 4. Architecture

A linear pipeline of small, independently testable units. Each takes/returns plain data (numpy image arrays, paths) and has one job.

```
INPUT_DIR ─► ingest ─► isolate(bg) ─► stylize(backend) ─► post ─► layout ─► book(PDF) ─► OUTPUT.pdf
```

### Modules

| Module | Responsibility | Interface (conceptual) |
|---|---|---|
| `ingest.py` | Load images from a dir; EXIF auto-rotate; validate; deterministic order. | `load_images(dir) -> list[LoadedImage(path, bgr)]` |
| `isolate.py` | Subject isolation: `rembg`/U²-Net → subject composited on white. Modes `auto\|keep\|remove`. `auto` removes when a dominant person/face is detected, else keeps. Failure → fall back to `keep`. | `isolate(bgr, mode) -> bgr` |
| `stylize/base.py` | `Stylizer` interface: photo → line art (black lines on white). | `to_lineart(bgr) -> gray` |
| `stylize/cv.py` | OpenCV adaptive-threshold/XDoG backend. No model/GPU. Test + fallback. | implements `Stylizer` |
| `stylize/contour.py` | Likeness-first neural extractor — **Informative Drawings "contour"** (or `controlnet_aux` `LineartDetector`, whichever integrates cleanest; both are pure line extraction). Loads weights once; runs on GPU. | implements `Stylizer` |
| `stylize/__init__.py` | `get_stylizer(name, **opts) -> Stylizer` factory. | — |
| `post.py` | Coloring-book cleanup: binarize to clean black-on-white, **morphological close** so regions enclose, despeckle, **normalize line weight** to a bold kid-friendly stroke. | `clean(gray, line_weight) -> gray` |
| `layout.py` | Fit one line-art image into a Letter page region (margins, centered, aspect preserved) at print DPI; optional bottom caption. | `place(gray, page, caption?) -> page_raster` |
| `book.py` | Assemble **cover page** (title) + one page per photo into a single PDF via `reportlab`. | `build_pdf(pages, title, out)` |
| `pipeline.py` | Orchestrate the above. Public API. | `convert_book(input_dir, output_pdf, BookOptions) -> dict` |
| `cli.py` | argparse → `BookOptions` → `convert_book`. | `photo2coloringbook` |

### `BookOptions` (config surface)

`title`, `paper="letter"`, `dpi=300`, `margin_in=0.5`, `bg="auto"` (`auto\|keep\|remove`), `backend="contour"` (`contour\|cv`), `line_weight` (stroke thickness in px), `caption_mode="off"` (`off\|sidecar` — sidecar reads `<image>.txt` next to a photo for an optional location/caption line).

### CLI

```
photo2coloringbook INPUT_DIR OUTPUT.pdf --title "Our Family Summer" \
    [--bg auto|keep|remove] [--backend contour|cv] [--line-weight N] [--paper letter]
```

## 5. Data flow & rendering notes

- Work in BGR/gray numpy arrays through the pipeline; line art is grayscale (black=line, white=paper), binarized in `post`.
- Render at **print resolution** (Letter @ 300 DPI = 2550×3300 px) so lines are crisp on paper. `layout` scales the cleaned line art to fit the printable area; `book` places the raster on the page.
- The cover page is title text (and optionally one hero line-art image) — `reportlab` draws text + image.

## 6. Error handling

- Unreadable/corrupt image → log a warning, skip it; continue.
- Zero usable images → fail with a clear message.
- `rembg`/model weights missing → clear error naming the install step (`pip install '.[gpu]'`) or, for `isolate`, fall back to `keep` with a warning.
- Output PDF must exist and contain `cover + N` pages on success; `convert_book` returns stats `{pages, backend, output}`.

## 7. Testing

Same ethos as `photo2cricut` — **no real/copyrighted photos bundled**.

- A synthetic generator produces a few deterministic test images (reuse/extend `photo2cricut.testimage` style).
- End-to-end test runs the full pipeline with `backend="cv"` (no model/GPU/network) and asserts: output PDF exists, is a valid PDF, has **`N+1` pages** (cover + N images), via `pypdf`.
- Unit tests for `post` (closed-region / despeckle behavior on a synthetic shape), `layout` (aspect/margin math), and the `get_stylizer` factory.
- The `contour` backend is exercised manually on appserver (GPU + weights), not in the default suite.

## 8. Build order (phased, de-risks early)

1. **Phase 1 — skeleton on `cv` backend.** ingest → (bg keep) → cv stylize → post → layout → book. Produces a real multi-page Letter PDF from a folder. Full test suite green. *Deliverable: a working (if rough) coloring book.*
2. **Phase 2 — quality leap on appserver.** Add `rembg` isolation (`auto`/`remove`) and the `contour` neural backend; wire `[gpu]` extras; validate on real family photos on the GPU.
3. **Phase 3 — polish.** Cover/title page, optional sidecar captions, line-weight tuning, page layout refinements.

Each phase is its own PR.

## 9. Out of scope / deferred (YAGNI)

- **Automatic location detection.** Reliable auto-location needs EXIF-GPS → reverse geocoding (cloud) or landmark AI — fragile, and explicitly secondary. v1 honors the request cheaply via an **optional manual sidecar caption** (`<image>.txt`). Auto EXIF-GPS geocoding is a possible later follow-up.
- **Generative "illustrated/storybook" backend** (SD 1.5 + ControlNet lineart). Feasible on the 6 GB GPU (512–768 px, batch) but risks facial-likeness drift. Deferred behind the stable `Stylizer` interface; can be added as a third backend if the extracted-line look proves too plain.
- **SDXL / cloud image APIs.** Not needed; reserved as a future escape hatch for higher polish.
- Per-image interactive editing / GUI.

## 10. Assumptions to confirm at implementation

- Informative Drawings weights are obtainable and license-compatible for personal use; if integration is heavy, `controlnet_aux.LineartDetector` is the drop-in substitute (same `Stylizer` interface, pip-installable).
- appserver gets a fresh `torch` cu121 venv (or a Docker image, since Docker is present); CUDA toolkit not required (torch ships its own runtime; driver 595 is current).
