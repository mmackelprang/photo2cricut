# photo2coloringbook Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the likeness-first neural `contour` backend (controlnet_aux LineartDetector) and `rembg` background removal to `photo2coloringbook`, so the production recipe `--backend contour --bg auto` runs on appserver's GPU and produces recognizable, clean coloring pages.

**Architecture:** Drop-in implementations behind the Phase 1 interfaces — `ContourStylizer` implements `Stylizer.to_lineart`; `isolate()` gains `auto`/`remove` via rembg. Heavy deps (torch, controlnet_aux, rembg) are **lazy-imported** so the base install and the no-torch dev-box test suite are unaffected. The model runs on appserver; the Windows dev box only runs the cv path + guarded tests.

**Tech Stack:** controlnet_aux 0.0.10 (LineartDetector = carolineec Informative Drawings weights, from `lllyasviel/Annotators`), torch 2.5.1+cu121 / torchvision 0.20.1+cu121, rembg 2.0.76 + onnxruntime 1.27.0, OpenCV (Haar face detect, already a base dep), Pillow.

## Global Constraints

- **Work on a branch:** `feat/coloringbook-phase2`, merged via PR/`--no-ff`. Never commit to `main` directly.
- **Base install must never import torch:** all torch/controlnet_aux/rembg imports are lazy (inside the factory/function call), wrapped so a missing dep raises a clear `RuntimeError` naming the `[gpu]` extra — never a raw `ModuleNotFoundError`.
- **Default suite stays green on the dev box** (no torch): model-dependent tests use `pytest.importorskip(...)`; missing-dependency-error tests `skip` when the dep IS present. The real validation is the appserver UAT (Task 4).
- **VRAM ceiling:** the GTX 1660 SUPER has ~5.7 GB free; `ContourStylizer` MUST cap `detect_resolution = min(short_side, 2048)` (native 24 MP OOMs).
- **Line-art convention everywhere:** grayscale uint8, **black lines = 0, white = 255**. The neural model emits white-on-black soft grayscale — invert + binarize inside `to_lineart`.
- **Defaults unchanged:** `BookOptions` keeps `backend="cv"`, `bg="keep"` (base-install-safe). The quality path is the explicit `--backend contour --bg auto`, documented for appserver. Do NOT flip defaults (would break base installs lacking torch).
- **Run dev-box tests with:** `.venv/Scripts/python.exe -m pytest`.

---

### Task 1: `contour` neural backend

**Files:**
- Create: `photo2coloringbook/stylize/contour.py`
- Modify: `photo2coloringbook/stylize/__init__.py`, `pyproject.toml`
- Test: `tests/coloringbook/test_contour.py`

**Interfaces:**
- Consumes: the `Stylizer` contract from Phase 1 (`to_lineart(bgr) -> gray`, black=0/white=255).
- Produces: `ContourStylizer(device=None, ink_level=65, max_resolution=2048)` implementing `Stylizer`; `get_stylizer("contour")` returns it (lazily) or raises a clear `RuntimeError` if the `[gpu]` extra is missing.

- [ ] **Step 0: Branch**

Run: `git checkout -b feat/coloringbook-phase2`

- [ ] **Step 1: Write the failing tests**

`tests/coloringbook/test_contour.py`:
```python
import importlib.util
import numpy as np
import pytest
from photo2cricut.testimage import make_portrait
from photo2coloringbook.stylize import get_stylizer

_HAS_CONTROLNET = importlib.util.find_spec("controlnet_aux") is not None


def test_contour_requires_gpu_extra_when_absent():
    if _HAS_CONTROLNET:
        pytest.skip("controlnet_aux installed; cannot test the missing-dep path")
    with pytest.raises(RuntimeError, match="gpu"):
        get_stylizer("contour")


@pytest.mark.skipif(not _HAS_CONTROLNET, reason="needs the [gpu] extra (controlnet_aux)")
def test_contour_to_lineart_is_black_on_white():
    out = get_stylizer("contour").to_lineart(make_portrait(seed=0))
    assert out.dtype == np.uint8 and out.ndim == 2
    assert (out == 0).any() and (out == 255).any()
    assert set(np.unique(out)).issubset({0, 255})  # binarized
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_contour.py -v`
Expected (dev box, no controlnet_aux): the skipif test is skipped; `test_contour_requires_gpu_extra_when_absent` FAILS because `get_stylizer("contour")` currently raises `NotImplementedError`, not a `RuntimeError` matching "gpu".

- [ ] **Step 3: Write the implementation**

`photo2coloringbook/stylize/contour.py`:
```python
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
        self.ink_level = ink_level          # model brightness >= this -> black ink
        self.max_resolution = max_resolution  # short-side cap (6 GB VRAM ceiling)

    def to_lineart(self, bgr: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = min(min(pil.size), self.max_resolution)
        out = self.det(pil, detect_resolution=res, image_resolution=res, coarse=False)
        model = np.asarray(out)[..., 0]     # white lines on black, 0..~230
        # invert + binarize so clean()'s 127 threshold keeps the (mid-gray) strokes
        return np.where(model >= self.ink_level, 0, 255).astype(np.uint8)
```

Modify `photo2coloringbook/stylize/__init__.py` — replace the `contour` branch with a lazy, dep-guarded import:
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
        try:
            from .contour import ContourStylizer
        except ImportError as e:
            raise RuntimeError(
                "the 'contour' backend needs the [gpu] extra "
                "(pip install '.[gpu]': torch, controlnet_aux). "
                f"Underlying import error: {e}") from e
        return ContourStylizer()
    raise ValueError(f"unknown backend: {name!r} (use 'cv' or 'contour')")
```

Modify `pyproject.toml` — add `controlnet_aux` to the `[gpu]` extra:
```toml
gpu = ["torch", "torchvision", "controlnet_aux>=0.0.10", "rembg", "onnxruntime"]  # Phase 2 backend
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_contour.py -v`
Expected (dev box): `test_contour_requires_gpu_extra_when_absent` PASSES; the real test is SKIPPED. (On appserver with `[gpu]` installed, the real test runs and passes.)

- [ ] **Step 5: Commit**

```bash
git add photo2coloringbook/stylize/contour.py photo2coloringbook/stylize/__init__.py pyproject.toml tests/coloringbook/test_contour.py
git commit -m "feat(coloringbook): contour neural line-art backend (controlnet_aux)"
```

---

### Task 2: `rembg` background removal in `isolate`

**Files:**
- Modify: `photo2coloringbook/isolate.py`
- Test: `tests/coloringbook/test_isolate.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `isolate(bgr, mode)` where `keep` is identity (unchanged), `remove` cuts the background and composites onto white via rembg, `auto` does `remove` only when an OpenCV Haar detector finds a face (else `keep`); a missing rembg raises a clear `RuntimeError` naming `[gpu]`. The rembg session is cached module-level (load u2net once).

- [ ] **Step 1: Replace the failing tests**

Replace the body of `tests/coloringbook/test_isolate.py` with:
```python
import importlib.util
import numpy as np
import pytest
from photo2coloringbook.isolate import isolate

_HAS_REMBG = importlib.util.find_spec("rembg") is not None


def test_keep_is_identity():
    img = np.zeros((10, 10, 3), np.uint8)
    assert isolate(img, "keep") is img


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        isolate(np.zeros((10, 10, 3), np.uint8), "bogus")


def test_auto_without_face_is_keep():
    # A flat gray image has no detectable face -> auto must not touch rembg.
    img = np.full((120, 120, 3), 128, np.uint8)
    assert isolate(img, "auto") is img


def test_remove_requires_rembg_when_absent():
    if _HAS_REMBG:
        pytest.skip("rembg installed; cannot test the missing-dep path")
    with pytest.raises(RuntimeError, match="gpu"):
        isolate(np.full((120, 120, 3), 128, np.uint8), "remove")


@pytest.mark.skipif(not _HAS_REMBG, reason="needs the [gpu] extra (rembg)")
def test_remove_returns_valid_bgr():
    from photo2cricut.testimage import make_portrait
    out = isolate(make_portrait(seed=0), "remove")
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_isolate.py -v`
Expected (dev box): `test_remove_requires_rembg_when_absent` FAILS (current code raises `NotImplementedError`, not a `RuntimeError` matching "gpu"); `test_auto_without_face_is_keep` FAILS (current `auto` raises `NotImplementedError`).

- [ ] **Step 3: Write the implementation**

Replace `photo2coloringbook/isolate.py` with:
```python
"""Subject isolation. 'keep' is a no-op; 'remove'/'auto' use rembg (u2net)."""

from __future__ import annotations

import cv2
import numpy as np

_session = None  # cached rembg u2net session (load once)


def _get_session():
    global _session
    if _session is None:
        try:
            from rembg import new_session
        except ImportError as e:
            raise RuntimeError(
                "background removal needs the [gpu] extra "
                "(pip install '.[gpu]': rembg, onnxruntime). "
                f"Underlying import error: {e}") from e
        _session = new_session("u2net")
    return _session


def _remove_to_white(bgr: np.ndarray) -> np.ndarray:
    from PIL import Image
    from rembg import remove
    sess = _get_session()
    rgba = remove(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), session=sess)
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(white, rgba).convert("RGB")
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def _has_face(bgr: np.ndarray) -> bool:
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(60, 60))
    return len(faces) > 0


def isolate(bgr: np.ndarray, mode: str = "keep") -> np.ndarray:
    if mode == "keep":
        return bgr
    if mode == "remove":
        return _remove_to_white(bgr)
    if mode == "auto":
        return _remove_to_white(bgr) if _has_face(bgr) else bgr
    raise ValueError(f"unknown bg mode: {mode!r} (use 'keep', 'auto', or 'remove')")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/coloringbook/test_isolate.py -v`
Expected (dev box): keep/unknown/auto-no-face/remove-requires-rembg all PASS; `test_remove_returns_valid_bgr` SKIPPED.

- [ ] **Step 5: Commit**

```bash
git add photo2coloringbook/isolate.py tests/coloringbook/test_isolate.py
git commit -m "feat(coloringbook): rembg background removal (auto/remove)"
```

---

### Task 3: appserver setup script + docs

**Files:**
- Create: `scripts/setup-appserver.sh`
- Modify: `README.md`, `CLAUDE.md`
- Test: shell syntax check (no pytest — this is infra/docs)

**Interfaces:** none (operational).

- [ ] **Step 1: Write the setup script**

`scripts/setup-appserver.sh`:
```bash
#!/usr/bin/env bash
# Set up the photo2coloringbook [gpu] backend on a CUDA host (e.g. appserver).
# Handles hosts with no system pip / no python3-venv ensurepip / no sudo:
# bootstraps pip via get-pip.py into a --without-pip venv.
#
# Usage:  scripts/setup-appserver.sh [venv_dir]   (default: .venv-gpu)
set -euo pipefail
cd "$(dirname "$0")/.."
ENVDIR="${1:-.venv-gpu}"

python3 -m venv --without-pip "$ENVDIR"
# shellcheck disable=SC1091
. "$ENVDIR/bin/activate"
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python /tmp/get-pip.py

# Torch with the CUDA 12.1 wheels (NOT on PyPI — needs the dedicated index).
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install controlnet_aux==0.0.10 rembg==2.0.76 onnxruntime==1.27.0
pip install -e .

echo ">> Verifying CUDA + GPU..."
python -c "import torch; print('cuda', torch.cuda.is_available(), \
  torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
echo ">> Done. Generate a book with:"
echo ">>   photo2coloringbook <photos_dir> book.pdf --backend contour --bg auto"
```

- [ ] **Step 2: Verify shell syntax**

Run: `bash -n scripts/setup-appserver.sh && echo OK`
Expected: `OK`

- [ ] **Step 3: Update docs**

In `README.md`, replace the Phase-1 note in the photo2coloringbook section with:
```markdown
**Quality backend (GPU).** The likeness-first `contour` backend (a neural
line-art model) plus `rembg` background removal run on a CUDA box. Set it up:

```bash
scripts/setup-appserver.sh           # bootstraps pip + torch(cu121) + models
photo2coloringbook ./photos book.pdf --backend contour --bg auto --title "Our Family"
```

`--bg auto` removes busy backgrounds when a face is detected. Tune `contour`
ink with the model's `ink_level` if hair over-inks on harsh-lit shots.
Needs ~3.5 GB VRAM (short side is capped at 2048 px).
```

In `CLAUDE.md`, update the photo2coloringbook section: change "Phase 2 (not yet built)" to note that the `contour` backend (controlnet_aux `LineartDetector`) and `rembg` isolation are **shipped**, run via `scripts/setup-appserver.sh` on a CUDA host, and that heavy deps are lazy-imported (base install/test suite never import torch).

- [ ] **Step 4: Commit**

```bash
git add scripts/setup-appserver.sh README.md CLAUDE.md
git commit -m "docs(coloringbook): appserver GPU setup script + Phase 2 usage"
```

---

### Task 4: appserver UAT + lock `ink_level` default (manual gate)

**Files:** none committed unless `ink_level` default is changed in `photo2coloringbook/stylize/contour.py`.

This task is a **running-app validation on appserver** (not a pytest). It is the Phase 2 quality gate, standing in for unit tests of the non-deterministic model path.

- [ ] **Step 1: Deploy the branch to appserver**

From the dev box: push the branch state to appserver (rsync the repo working tree, excluding `.git`/`.venv*`/`images` if already present), or `git fetch` if a remote is reachable there. Then on appserver run `scripts/setup-appserver.sh` (Task 3).

- [ ] **Step 2: Stage a representative photo set**

Ensure ~8–12 of the real `images/` photos (mix of flat-lit selfie, 24 MP studio, harsh sunlit, small portraits) are on appserver under a `photos/` dir.

- [ ] **Step 3: Generate the book on the contour backend**

Run on appserver:
```bash
photo2coloringbook ./photos ./contour_book.pdf --backend contour --bg auto --title "Family"
python -c "from pypdf import PdfReader; print('pages:', len(PdfReader('contour_book.pdf').pages))"
```
Expected: exit 0; pages == 1 cover + N photos. Confirm it runs without OOM (resolution cap working).

- [ ] **Step 4: Eyeball + tune `ink_level`**

Pull the PDF (and/or per-page PNGs) back to the dev box for review. Check faces are recognizable and hair/dark regions are not over-inked. If harsh-lit shots over-ink (per the spike's `PXL_121716408` case), raise the `ContourStylizer` default `ink_level` (try 75–90) and regenerate until the full set looks good. Lock the chosen default in `contour.py` and commit:
```bash
git add photo2coloringbook/stylize/contour.py
git commit -m "tune(coloringbook): lock contour ink_level default after appserver UAT"
```

- [ ] **Step 5: Record the result**

Note the final `ink_level`, per-image timing, peak VRAM, and the page count in the PR/merge description. Save a couple of `--backend contour` sample pages under `examples/uat-phase2/` (git-ignored) for reference.

---

## Self-Review

**1. Spec coverage:** §4 `contour` backend → Task 1; §4 `isolate` auto/remove (rembg) → Task 2; §3 `[gpu]` extra + appserver execution → Tasks 1/3; §7 "contour exercised manually on appserver" → Task 4; §10 assumptions (weights obtainable, controlnet_aux substitute) → resolved by the spike, baked into Task 1. ✓

**2. Placeholder scan:** No TBD/TODO; every code step has complete code; the one manual task (UAT) has exact commands + acceptance criteria. ✓

**3. Type consistency:** `ContourStylizer.to_lineart(bgr)->gray uint8 black/white` matches the `Stylizer` contract and `convert_book`'s usage; `isolate(bgr, mode)->bgr` unchanged from Phase 1; `get_stylizer("contour")` return type matches the factory contract. Lazy-import error type is `RuntimeError` in both the factory and `isolate`, matching the tests. ✓
