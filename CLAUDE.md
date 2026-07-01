# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`photo2cricut` converts a JPG/PNG photo into a **Cricut-ready single-line (centerline) SVG** for the Cricut's Draw/Pen mode (a slow pen plotter). The defining constraint: produce **centerlines** (one pen stroke per line) rather than **outlines** (which double every line into a closed loop and look bad with a pen). Everything runs **fully local** — no API keys, no web service. Read `README.md` for the user-facing tuning guide; this file is the developer map.

## Second package in this repo: `photo2coloringbook/`

This repo now hosts a **second, independent sibling tool** alongside `photo2cricut`:
`photo2coloringbook/` turns a **folder of photos** into a print-ready **US-Letter PDF
coloring book** — bold **black-on-white** line art (the opposite of centerlines: closed
outlines enclosing fillable regions), one page per photo plus a title cover. Console
entry point: `photo2coloringbook` (also runnable as `python -m photo2coloringbook.cli`).

- **Pipeline** (linear, in `photo2coloringbook/`): `ingest → isolate → stylize → post →
  layout → book`, orchestrated by `pipeline.convert_book(input_dir, output_pdf, BookOptions)`.
- **Phase 1 (shipped):** the no-GPU `cv` backend (OpenCV adaptive threshold) with `keep`
  (no-op) isolation — the whole flow runs with zero models/GPU.
- **Phase 2 (shipped):** the GPU `contour` neural backend (`stylize/contour.py` —
  `controlnet_aux` `LineartDetector`, = Informative Drawings weights) and `rembg`
  background removal (`--bg auto`/`remove` in `isolate.py`). The model emits soft
  white-on-black; `to_lineart` inverts + binarizes (`ink_level`) and caps the short side
  at 2048 px (6 GB VRAM ceiling). These live behind the `[gpu]` extra and are
  **lazy-imported** — the base install and the dev-box suite never load torch; a missing
  dep raises a clear `[gpu]`-extra `RuntimeError`. Set up on a CUDA host with
  `scripts/setup-appserver.sh`, then run `--backend contour --bg auto`. Defaults stay
  `backend="cv"`/`bg="keep"` so the base install works everywhere.
- **Testing the GPU paths:** `contour`/`remove` tests `importorskip` their deps, so they
  skip on the no-torch dev box and run only where `[gpu]` is installed (appserver). The
  real quality validation is a manual appserver UAT.
- **Deps are isolated:** coloring-book deps (`reportlab`, `Pillow>=10.1`) live in the
  `coloringbook` extra; `photo2cricut` stays lean (no torch). Tests use `pypdf` (dev extra)
  and the same synthetic `photo2cricut.testimage.make_portrait` images — no real photos.
- **Line-art convention everywhere:** grayscale uint8, **black lines = 0, white = 255**.
- Coloring-book tests live in `tests/coloringbook/`; pytest runs with
  `--import-mode=importlib` so both suites can share test-file basenames.

## Commands

Setup (creates `.venv`, installs editable, runs a smoke test):
```bash
./install.sh            # or ./install.sh --dev  (adds pytest+cairosvg, runs tests)
source .venv/bin/activate
```

Day-to-day (Makefile targets wrap these):
```bash
pytest -q                                   # full suite
pytest tests/test_pipeline.py::test_no_fills_present -q   # single test
pytest -k canny -q                          # by keyword (e.g. one method param)

photo2cricut-makeimg examples/test.jpg      # generate synthetic test portrait
photo2cricut examples/test.jpg out.svg --method xdog --width-in 8   # convert
photo2cricut-validate out.svg               # check Cricut-readiness (exit 0/1)
```

There is no linter configured. `vpype` is **not** called via PATH — the pipeline shells out to `python -m vpype_cli`, so it only works inside the installed venv.

## Architecture

The whole program is a single linear image→SVG pipeline plus thin wrappers. There is no framework, no state, no I/O beyond files and one subprocess.

**`photo2cricut/pipeline.py` — the entire core.** `convert(input, output, Options)` is the one public entry point and orchestrates this fixed sequence:

1. `cv2.imread` → downscale longest side to `max_px` → grayscale → `bilateralFilter` → CLAHE (local contrast).
2. `extract_lines()` — `xdog()` (Extended Difference-of-Gaussians, "drawn" look, stays connected) or Canny (thin/crisp, more broken). Returns a boolean foreground mask.
3. `despeckle()` — drop connected blobs ≤ `despeckle` px (OpenCV connected components).
4. `skeleton_to_polylines()` — `skimage.skeletonize` → `sknw.build_sknw` graph → each graph **edge** becomes a polyline. Note the coordinate swap: sknw gives `(row=y, col=x)`; we emit `(x, y)`. Then `smooth_polyline()` (moving average, endpoints fixed).
5. `write_intermediate_svg()` — raw polylines to a temp SVG (pixel units).
6. `run_vpype()` — subprocess `python -m vpype_cli`: `scaleto width_in×1000in` (homogeneous scale, width is the binding dimension) → `linemerge` → `linesimplify` → `filter --min-length` → `linesort` → `layout tight`. This is where plotting optimization and physical sizing happen.
7. `force_inches()` — re-parse the vpype output and rewrite root `width`/`height` in **inches** (derived from the viewBox aspect ratio) so Cricut Design Space sizes the art correctly. vpype alone doesn't guarantee this.

**`Options` (dataclass in pipeline.py) is the single config surface.** Every tuning knob lives here. The CLI maps argparse flags 1:1 onto `Options` fields — watch the renames via `dest=`: `--simplify`→`simplify_mm`, `--merge-mm`→`merge_mm`, `--min-line-mm`→`min_line_mm`, `--clahe`→`clahe_clip`, `--xdog-sigma`→`xdog_sigma`. Add a new parameter in `Options`, then wire it through `cli.py`.

**Three console commands, three thin modules** (defined in `pyproject.toml [project.scripts]`):
- `cli.py` → `photo2cricut` (wraps `convert`)
- `validate.py` → `photo2cricut-validate` (wraps `validate_svg`)
- `testimage.py` → `photo2cricut-makeimg` (wraps `make_portrait`)

**`validate.py` defines what "Cricut-ready" means** (and the test suite asserts against it): parses as SVG, root `width`/`height` carry explicit `in`/`mm`/`cm` units, has a `viewBox`, contains stroke geometry (`path`/`polyline`/`line`/`polygon`), has **no real fills** (fills draw badly with a pen), and stroke count is under `MAX_RECOMMENDED_STROKES` (~3000, a Design Space performance ceiling — over it is a WARN, not a FAIL).

## Testing conventions

- **No real/copyrighted photo is bundled.** `testimage.make_portrait(seed)` deterministically renders a synthetic shaded portrait with OpenCV primitives; tests generate it on the fly. Use it (not the loose `.jpg`/`.svg` files in the repo root) for any new test.
- Tests are **end-to-end**: photo → `convert` → `validate_svg`, parametrized over both methods. They assert the output *validates as Cricut-ready* rather than checking pixel values — keep new tests behavior-level against the validator.
- `tests/conftest.py` injects the repo root on `sys.path` so the package imports from a bare checkout without install.
- Loose images/SVGs in the repo root are scratch inputs; generated art under `examples/` is git-ignored.
