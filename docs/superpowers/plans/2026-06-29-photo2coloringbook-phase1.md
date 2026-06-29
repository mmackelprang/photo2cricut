# photo2coloringbook Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the end-to-end `photo2coloringbook` skeleton on the no-GPU `cv` backend — a folder of photos → a print-ready US-Letter PDF coloring book (cover + one line-art page per photo).

**Architecture:** A new sibling package `photo2coloringbook/` in this repo, structured as a linear pipeline of small, independently testable units: `ingest → isolate → stylize → post → layout → book`. Phase 1 implements the `cv` (OpenCV adaptive-threshold) stylizer and `keep` (no-op) isolation so the whole flow runs with zero models/GPU; the GPU `contour` backend and `rembg` isolation are Phase 2 drop-ins behind the same interfaces.

**Tech Stack:** Python 3.9+ (appserver runs 3.12), OpenCV (`opencv-python-headless`), NumPy, Pillow ≥10.1, ReportLab (PDF), pytest + pypdf (tests).

## Global Constraints

- **Work on a branch:** all implementation on `feat/coloringbook-phase1`, merged via PR. Never commit to `main` directly.
- **Package isolation:** `photo2cricut` must stay importable without any new deps — coloring-book deps live in optional-dependency groups only.
- **No real/copyrighted photos in tests:** reuse `photo2cricut.testimage.make_portrait(seed)` to synthesize test images.
- **Line-art convention everywhere:** grayscale `uint8`, **black lines = 0, white paper = 255**.
- **Backends this phase:** `cv` only is functional; `contour` must raise a clear "Phase 2 / [gpu] extra" error, never silently no-op.
- **Isolation this phase:** `keep` only is functional; `auto`/`remove` must raise a clear "Phase 2 / [gpu] extra" error.
- **Page target:** US Letter, default 300 DPI (2550×3300 px), 0.5" margins.
- Pillow pinned `>=10.1` (needs `ImageFont.load_default(size=...)`); use `Image.Resampling.LANCZOS`.

---

### Task 1: Package scaffold + packaging

**Files:**
- Create: `photo2coloringbook/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/coloringbook/test_package.py`

**Interfaces:**
- Consumes: nothing.
- Produces: importable package `photo2coloringbook` with `__version__`; console script `photo2coloringbook`; optional-deps groups `coloringbook`, `gpu`; `dev` extra gains `pypdf`, `reportlab`, `Pillow>=10.1`.

- [ ] **Step 0: Create the branch**

Run:
```bash
git checkout -b feat/coloringbook-phase1
```

- [ ] **Step 1: Write the failing test**

`tests/coloringbook/test_package.py`:
```python
def test_package_imports_and_has_version():
    import photo2coloringbook as cb
    assert isinstance(cb.__version__, str) and cb.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_package.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'photo2coloringbook'`

- [ ] **Step 3: Write minimal implementation**

`photo2coloringbook/__init__.py`:
```python
"""photo2coloringbook -- photos to a print-ready PDF coloring book."""

__version__ = "0.1.0"
```

Modify `pyproject.toml` — add the package, console script, and dependency groups. Replace the `[project.optional-dependencies]`, `[project.scripts]`, and `[tool.setuptools]` sections with:
```toml
[project.optional-dependencies]
dev = ["pytest>=7", "cairosvg>=2.5", "pypdf>=4.0", "reportlab>=4.0", "Pillow>=10.1"]
coloringbook = ["reportlab>=4.0", "Pillow>=10.1"]
gpu = ["torch", "torchvision", "rembg", "onnxruntime"]  # Phase 2 backend

[project.scripts]
photo2cricut = "photo2cricut.cli:main"
photo2cricut-validate = "photo2cricut.validate:main"
photo2cricut-makeimg = "photo2cricut.testimage:main"
photo2coloringbook = "photo2coloringbook.cli:main"

[tool.setuptools]
packages = ["photo2cricut", "photo2coloringbook", "photo2coloringbook.stylize"]
```

Then install the updated dev extras into the venv so tests can run:
```bash
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_package.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml photo2coloringbook/__init__.py tests/coloringbook/test_package.py
git commit -m "feat(coloringbook): package scaffold + packaging"
```

---

### Task 2: Ingest — load + EXIF-orient images

**Files:**
- Create: `photo2coloringbook/ingest.py`
- Test: `tests/coloringbook/test_ingest.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `LoadedImage` dataclass with fields `path: str`, `bgr: np.ndarray` (H×W×3 BGR uint8).
  - `load_images(directory: str) -> list[LoadedImage]` — sorted by filename; skips unreadable files with a warning; raises `RuntimeError` if no usable images; raises `NotADirectoryError` if the path isn't a directory.

- [ ] **Step 1: Write the failing test**

`tests/coloringbook/test_ingest.py`:
```python
import os
import cv2
import pytest
from photo2cricut.testimage import make_portrait
from photo2coloringbook.ingest import load_images, LoadedImage


def _write_imgs(d, n):
    for i in range(n):
        cv2.imwrite(os.path.join(d, f"p{i}.jpg"), make_portrait(seed=i),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])


def test_loads_all_images_sorted(tmp_path):
    _write_imgs(str(tmp_path), 3)
    (tmp_path / "notes.txt").write_text("ignore me")
    imgs = load_images(str(tmp_path))
    assert len(imgs) == 3
    assert all(isinstance(x, LoadedImage) for x in imgs)
    assert imgs[0].bgr.ndim == 3 and imgs[0].bgr.shape[2] == 3
    assert [os.path.basename(x.path) for x in imgs] == ["p0.jpg", "p1.jpg", "p2.jpg"]


def test_empty_dir_raises(tmp_path):
    with pytest.raises(RuntimeError):
        load_images(str(tmp_path))


def test_not_a_directory_raises(tmp_path):
    f = tmp_path / "x.jpg"
    f.write_bytes(b"not an image")
    with pytest.raises(NotADirectoryError):
        load_images(str(f))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'photo2coloringbook.ingest'`

- [ ] **Step 3: Write minimal implementation**

`photo2coloringbook/ingest.py`:
```python
"""Load and EXIF-orient images from a directory."""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
            pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        except Exception as e:  # corrupt/unreadable
            print(f"warning: skipping unreadable image {path}: {e}")
            continue
        bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        out.append(LoadedImage(path=path, bgr=bgr))
    if not out:
        raise RuntimeError(f"No usable images found in {directory}")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_ingest.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add photo2coloringbook/ingest.py tests/coloringbook/test_ingest.py
git commit -m "feat(coloringbook): ingest images with EXIF orientation"
```

---

### Task 3: Stylize — base interface, cv backend, factory

**Files:**
- Create: `photo2coloringbook/stylize/__init__.py`, `photo2coloringbook/stylize/base.py`, `photo2coloringbook/stylize/cv.py`
- Test: `tests/coloringbook/test_stylize.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Stylizer` Protocol with `to_lineart(bgr: np.ndarray) -> np.ndarray` (grayscale uint8, black-on-white).
  - `CvStylizer` implementing it via OpenCV adaptive threshold.
  - `get_stylizer(name: str) -> Stylizer` — `"cv"` → `CvStylizer`; `"contour"` → raises `NotImplementedError`; else `ValueError`.

- [ ] **Step 1: Write the failing test**

`tests/coloringbook/test_stylize.py`:
```python
import numpy as np
import pytest
from photo2cricut.testimage import make_portrait
from photo2coloringbook.stylize import get_stylizer


def test_cv_stylizer_returns_black_on_white_gray():
    s = get_stylizer("cv")
    out = s.to_lineart(make_portrait(seed=0))
    assert out.dtype == np.uint8 and out.ndim == 2
    assert out.shape == make_portrait(seed=0).shape[:2]
    assert (out == 0).any() and (out == 255).any()  # has lines and paper


def test_contour_backend_not_yet_available():
    with pytest.raises(NotImplementedError):
        get_stylizer("contour")


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_stylizer("bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_stylize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'photo2coloringbook.stylize'`

- [ ] **Step 3: Write minimal implementation**

`photo2coloringbook/stylize/base.py`:
```python
"""Stylizer interface: photo -> line art."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Stylizer(Protocol):
    def to_lineart(self, bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 photo -> grayscale uint8 line art (black lines=0, white=255)."""
        ...
```

`photo2coloringbook/stylize/cv.py`:
```python
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
```

`photo2coloringbook/stylize/__init__.py`:
```python
"""Stylizer factory."""

from __future__ import annotations

from .base import Stylizer
from .cv import CvStylizer

__all__ = ["Stylizer", "CvStylizer", "get_stylizer"]


def get_stylizer(name: str) -> Stylizer:
    if name == "cv":
        return CvStylizer()
    if name == "contour":
        raise NotImplementedError(
            "the 'contour' backend requires the [gpu] extra and arrives in Phase 2; "
            "use --backend cv for now")
    raise ValueError(f"unknown backend: {name!r} (use 'cv' or 'contour')")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_stylize.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add photo2coloringbook/stylize/ tests/coloringbook/test_stylize.py
git commit -m "feat(coloringbook): stylizer interface + cv backend + factory"
```

---

### Task 4: Post — coloring-book cleanup

**Files:**
- Create: `photo2coloringbook/post.py`
- Test: `tests/coloringbook/test_post.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `clean(lineart_gray: np.ndarray, line_weight: int = 3, despeckle: int = 12) -> np.ndarray` — grayscale uint8 black-on-white in and out: binarize, morphological close (enclose regions), despeckle tiny blobs, thicken lines to `line_weight`.

- [ ] **Step 1: Write the failing test**

`tests/coloringbook/test_post.py`:
```python
import numpy as np
from photo2coloringbook.post import clean


def test_clean_removes_speckles_keeps_shapes_and_is_black_on_white():
    img = np.full((120, 120), 255, np.uint8)      # white page
    img[20:100, 20:100] = 0                        # solid black block (big component)
    img[5, 5] = 0                                  # 1-px speckle (should be removed)
    out = clean(img, line_weight=1, despeckle=12)
    assert out.dtype == np.uint8 and out.ndim == 2
    assert out[5, 5] == 255                         # speckle gone
    assert (out[20:100, 20:100] == 0).any()         # block survives
    assert (out == 255).any() and (out == 0).any()  # black-on-white


def test_line_weight_thickens():
    img = np.full((60, 60), 255, np.uint8)
    img[30, 10:50] = 0                              # 1-px horizontal line
    thin = clean(img, line_weight=1, despeckle=0)
    thick = clean(img, line_weight=5, despeckle=0)
    assert (thick == 0).sum() > (thin == 0).sum()   # more ink after thickening
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_post.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'photo2coloringbook.post'`

- [ ] **Step 3: Write minimal implementation**

`photo2coloringbook/post.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_post.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add photo2coloringbook/post.py tests/coloringbook/test_post.py
git commit -m "feat(coloringbook): coloring-book cleanup pass"
```

---

### Task 5: Layout — page fit + cover

**Files:**
- Create: `photo2coloringbook/layout.py`
- Test: `tests/coloringbook/test_layout.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `fit_canvas(lineart_gray, page_w: int, page_h: int, margin: int) -> np.ndarray` — white page-sized gray raster (page_h×page_w) with the art centered, aspect-fit within margins.
  - `make_cover(title: str, page_w: int, page_h: int) -> np.ndarray` — white page-sized gray raster with the title centered.

- [ ] **Step 1: Write the failing test**

`tests/coloringbook/test_layout.py`:
```python
import numpy as np
from photo2coloringbook.layout import fit_canvas, make_cover


def test_fit_canvas_sizes_and_centers():
    art = np.zeros((100, 200), np.uint8)            # 2:1 black art
    page = fit_canvas(art, page_w=1000, page_h=1000, margin=50)
    assert page.shape == (1000, 1000)
    assert page[0, 0] == 255 and page[-1, -1] == 255  # corners are white margin
    assert (page == 0).any()                          # art placed


def test_make_cover_draws_title():
    cover = make_cover("Hello", page_w=800, page_h=1000)
    assert cover.shape == (1000, 800)
    assert (cover < 128).any()                        # some dark title pixels
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_layout.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'photo2coloringbook.layout'`

- [ ] **Step 3: Write minimal implementation**

`photo2coloringbook/layout.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_layout.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add photo2coloringbook/layout.py tests/coloringbook/test_layout.py
git commit -m "feat(coloringbook): page layout + cover composition"
```

---

### Task 6: Book — assemble PDF

**Files:**
- Create: `photo2coloringbook/book.py`
- Test: `tests/coloringbook/test_book.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `build_pdf(page_rasters: list[np.ndarray], out_path: str) -> None` — one US-Letter page per raster; raises `ValueError` if empty.

- [ ] **Step 1: Write the failing test**

`tests/coloringbook/test_book.py`:
```python
import numpy as np
import pytest
from pypdf import PdfReader
from photo2coloringbook.book import build_pdf


def test_build_pdf_one_page_per_raster(tmp_path):
    rasters = [np.full((330, 255), 255, np.uint8) for _ in range(3)]
    out = str(tmp_path / "book.pdf")
    build_pdf(rasters, out)
    assert len(PdfReader(out).pages) == 3


def test_build_pdf_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        build_pdf([], str(tmp_path / "x.pdf"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_book.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'photo2coloringbook.book'`

- [ ] **Step 3: Write minimal implementation**

`photo2coloringbook/book.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_book.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add photo2coloringbook/book.py tests/coloringbook/test_book.py
git commit -m "feat(coloringbook): assemble Letter PDF from page rasters"
```

---

### Task 7: Isolate — subject isolation (keep no-op)

**Files:**
- Create: `photo2coloringbook/isolate.py`
- Test: `tests/coloringbook/test_isolate.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `isolate(bgr: np.ndarray, mode: str = "keep") -> np.ndarray` — `"keep"` returns input unchanged; `"auto"`/`"remove"` raise `NotImplementedError`; else `ValueError`.

- [ ] **Step 1: Write the failing test**

`tests/coloringbook/test_isolate.py`:
```python
import numpy as np
import pytest
from photo2coloringbook.isolate import isolate


def test_keep_is_identity():
    img = np.zeros((10, 10, 3), np.uint8)
    assert isolate(img, "keep") is img


def test_remove_and_auto_not_yet_available():
    img = np.zeros((10, 10, 3), np.uint8)
    for mode in ("auto", "remove"):
        with pytest.raises(NotImplementedError):
            isolate(img, mode)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        isolate(np.zeros((10, 10, 3), np.uint8), "bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_isolate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'photo2coloringbook.isolate'`

- [ ] **Step 3: Write minimal implementation**

`photo2coloringbook/isolate.py`:
```python
"""Subject isolation. Phase 1: 'keep' only; rembg-backed modes land in Phase 2."""

from __future__ import annotations

import numpy as np


def isolate(bgr: np.ndarray, mode: str = "keep") -> np.ndarray:
    if mode == "keep":
        return bgr
    if mode in ("auto", "remove"):
        raise NotImplementedError(
            f"--bg {mode} requires the [gpu] extra (rembg) and arrives in Phase 2; "
            "use --bg keep for now")
    raise ValueError(f"unknown bg mode: {mode!r} (use 'keep', 'auto', or 'remove')")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_isolate.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add photo2coloringbook/isolate.py tests/coloringbook/test_isolate.py
git commit -m "feat(coloringbook): subject isolation interface (keep no-op)"
```

---

### Task 8: Pipeline — orchestrate end-to-end

**Files:**
- Create: `photo2coloringbook/pipeline.py`
- Modify: `photo2coloringbook/__init__.py`
- Test: `tests/coloringbook/test_pipeline.py`

**Interfaces:**
- Consumes: `load_images` (T2), `isolate` (T7), `get_stylizer` (T3), `clean` (T4), `fit_canvas`/`make_cover` (T5), `build_pdf` (T6).
- Produces:
  - `BookOptions` dataclass: `title="Coloring Book"`, `paper="letter"`, `dpi=300`, `margin_in=0.5`, `bg="keep"`, `backend="cv"`, `line_weight=3`.
  - `convert_book(input_dir: str, output_pdf: str, opt: BookOptions | None = None) -> dict` returning `{"pages": int, "backend": str, "output": str}` (pages = cover + N).

- [ ] **Step 1: Write the failing test**

`tests/coloringbook/test_pipeline.py`:
```python
import os
import cv2
from pypdf import PdfReader
from photo2cricut.testimage import make_portrait
from photo2coloringbook import BookOptions, convert_book


def _make_dir(d, n):
    for i in range(n):
        cv2.imwrite(os.path.join(d, f"p{i}.jpg"), make_portrait(seed=i),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])


def test_convert_book_produces_cover_plus_pages(tmp_path):
    src = tmp_path / "photos"; src.mkdir()
    _make_dir(str(src), 2)
    out = str(tmp_path / "book.pdf")
    stats = convert_book(str(src), out, BookOptions(title="Fam", backend="cv", dpi=100))
    assert stats == {"pages": 3, "backend": "cv", "output": out}
    assert os.path.exists(out)
    assert len(PdfReader(out).pages) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_pipeline.py -v`
Expected: FAIL — `ImportError: cannot import name 'BookOptions'`

- [ ] **Step 3: Write minimal implementation**

`photo2coloringbook/pipeline.py`:
```python
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


def convert_book(input_dir: str, output_pdf: str, opt: BookOptions | None = None) -> dict:
    opt = opt or BookOptions()
    page_w = int(LETTER_IN[0] * opt.dpi)
    page_h = int(LETTER_IN[1] * opt.dpi)
    margin = int(opt.margin_in * opt.dpi)

    stylizer = get_stylizer(opt.backend)
    images = load_images(input_dir)

    pages = [make_cover(opt.title, page_w, page_h)]
    for img in images:
        subject = isolate(img.bgr, opt.bg)
        lineart = stylizer.to_lineart(subject)
        cleaned = clean(lineart, line_weight=opt.line_weight)
        pages.append(fit_canvas(cleaned, page_w, page_h, margin))

    build_pdf(pages, output_pdf)
    return {"pages": len(pages), "backend": opt.backend, "output": output_pdf}
```

Modify `photo2coloringbook/__init__.py` to export the public API:
```python
"""photo2coloringbook -- photos to a print-ready PDF coloring book."""

from .pipeline import BookOptions, convert_book

__version__ = "0.1.0"
__all__ = ["BookOptions", "convert_book", "__version__"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add photo2coloringbook/pipeline.py photo2coloringbook/__init__.py tests/coloringbook/test_pipeline.py
git commit -m "feat(coloringbook): end-to-end pipeline (convert_book)"
```

---

### Task 9: CLI + docs

**Files:**
- Create: `photo2coloringbook/cli.py`
- Modify: `README.md`
- Test: `tests/coloringbook/test_cli.py`

**Interfaces:**
- Consumes: `BookOptions`, `convert_book` (T8).
- Produces: `main(argv=None) -> int` (0 success, 1 on handled errors); `build_parser() -> argparse.ArgumentParser`.

- [ ] **Step 1: Write the failing test**

`tests/coloringbook/test_cli.py`:
```python
import os
import cv2
from photo2cricut.testimage import make_portrait
from photo2coloringbook.cli import main


def test_cli_generates_pdf(tmp_path):
    src = tmp_path / "photos"; src.mkdir()
    cv2.imwrite(str(src / "a.jpg"), make_portrait(seed=0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    out = str(tmp_path / "out.pdf")
    rc = main([str(src), out, "--title", "Test", "--backend", "cv"])
    assert rc == 0
    assert os.path.exists(out)


def test_cli_bad_dir_returns_1(tmp_path, capsys):
    rc = main([str(tmp_path / "nope"), str(tmp_path / "o.pdf")])
    assert rc == 1
    assert "error:" in capsys.readouterr().err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'photo2coloringbook.cli'`

- [ ] **Step 3: Write minimal implementation**

`photo2coloringbook/cli.py`:
```python
"""photo2coloringbook -- CLI entry point."""

from __future__ import annotations

import argparse
import sys

from .pipeline import BookOptions, convert_book


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="photo2coloringbook",
        description="Turn a folder of photos into a print-ready US-Letter PDF coloring book.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("input_dir", help="folder of photos")
    ap.add_argument("output", help="output .pdf path")
    ap.add_argument("--title", default="Coloring Book", help="cover-page title")
    ap.add_argument("--bg", choices=["keep", "auto", "remove"], default="keep",
                    help="subject isolation (auto/remove need the [gpu] extra, Phase 2)")
    ap.add_argument("--backend", choices=["cv", "contour"], default="cv",
                    help="line-art backend (contour needs the [gpu] extra, Phase 2)")
    ap.add_argument("--line-weight", type=int, default=3, dest="line_weight",
                    help="stroke thickness in px")
    ap.add_argument("--paper", choices=["letter"], default="letter", help="page size")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    opt = BookOptions(title=args.title, bg=args.bg, backend=args.backend,
                      line_weight=args.line_weight, paper=args.paper)
    try:
        stats = convert_book(args.input_dir, args.output, opt)
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError,
            NotImplementedError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"OK -> {stats['output']}  ({stats['pages']} pages, backend={stats['backend']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Add a section to `README.md` (after the photo2cricut content) documenting the new tool:
```markdown
## photo2coloringbook (sibling tool)

Turn a **folder of photos** into a print-ready **US-Letter PDF coloring book**
(bold black-on-white line art, one page per photo + a title cover).

```bash
pip install -e ".[coloringbook]"
photo2coloringbook ./photos book.pdf --title "Our Family"
```

Phase 1 ships the local, no-GPU `cv` backend. The likeness-first neural
`contour` backend and `rembg` background removal (`--bg auto`) arrive in
Phase 2 (`pip install -e ".[gpu]"`, run on a CUDA GPU).
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_cli.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full suite and commit**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass (photo2cricut + photo2coloringbook).

```bash
git add photo2coloringbook/cli.py tests/coloringbook/test_cli.py README.md
git commit -m "feat(coloringbook): CLI entry point + README"
```

---

## Self-Review

**1. Spec coverage:**
- §3 packaging (separate package, extras, git) → Task 1 ✓
- §4 ingest/isolate/stylize(cv+factory)/post/layout/book/pipeline/cli → Tasks 2–9 ✓
- §4 `BookOptions` config surface → Task 8 ✓
- §5 print-DPI Letter rendering → Task 5/8 (`dpi`, `fit_canvas`) ✓
- §6 error handling (skip unreadable, zero-images raise, clear backend errors, return stats) → Tasks 2/3/7/8/9 ✓
- §7 testing (synthetic images, cv-backend e2e, pypdf page count, unit tests) → all tasks ✓
- §8 Phase 1 scope (cv backend end-to-end) → whole plan ✓
- §9 deferrals (contour, rembg auto/remove, auto-location) → raised-not-implemented in Tasks 3/7; captions not in Phase 1 ✓ (sidecar caption is Phase 3)

**2. Placeholder scan:** No TBD/TODO; every code step has complete code. ✓

**3. Type consistency:** `to_lineart(bgr)->gray`, `clean(gray,line_weight,despeckle)->gray`, `fit_canvas(gray,page_w,page_h,margin)->gray`, `make_cover(title,page_w,page_h)->gray`, `build_pdf(list,out)`, `isolate(bgr,mode)->bgr`, `convert_book(input_dir,output_pdf,opt)->dict`, `get_stylizer(name)->Stylizer` — names/signatures match across Tasks 2–9. ✓
