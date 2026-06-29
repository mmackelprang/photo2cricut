# photo2cricut

Turn a JPG photo into a **Cricut-ready single-line (centerline) SVG** you can import
into Cricut Design Space and set to **Draw / Pen**.

A Cricut in Draw mode is, effectively, a slow pen plotter. The trick to good pen
line-art is **centerline** tracing — one pen stroke per line — instead of the
**outline** tracing that consumer tools (and Design Space's own image upload)
produce, which doubles every line into a thin closed loop that looks bad with a pen.
This tool does the centerline path end-to-end and runs **fully local** (no API keys,
no web service).

```
JPG ─► grayscale + contrast ─► line extraction (XDoG | Canny)
    ─► despeckle ─► skeletonize (1-px centerlines)
    ─► trace skeleton into polylines ─► smoothing
    ─► vpype (merge / simplify / filter / sort, scale to inches)
    ─► Cricut-ready SVG
```

---

## Requirements

- Python 3.9+
- The Python packages in `requirements.txt` (installed for you by the scripts below).
  `opencv-python-headless` is used so it works on servers/containers with no display.
- Rendering PNG previews (optional) needs `cairosvg`, included in the `[dev]` extras.

## Install

Clone, then run the installer for your platform. It creates a `.venv`, installs the
package, and runs a smoke test (generate a test image → convert → validate).

**Linux / macOS**
```bash
git clone https://github.com/YOUR_USERNAME/photo2cricut.git
cd photo2cricut
./install.sh            # or: ./install.sh --dev   (also installs + runs tests)
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
git clone https://github.com/YOUR_USERNAME/photo2cricut.git
cd photo2cricut
./install.ps1           # or: ./install.ps1 -Dev
.\.venv\Scripts\Activate.ps1
```
If PowerShell blocks the script, run once in that session:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

**Docker**
```bash
docker build -t photo2cricut .
docker run --rm -v "$PWD:/work" photo2cricut /work/photo.jpg /work/out.svg --method xdog
```

## Usage

```bash
photo2cricut input.jpg output.svg
photo2cricut portrait.jpg out.svg --method xdog --width-in 8
photo2cricut portrait.jpg out.svg --method canny --width-in 6 --merge-mm 1.5
```

| Flag | Default | What it does |
|------|---------|--------------|
| `--method {xdog,canny}` | `xdog` | `xdog` = drawn-looking, stays connected on big contours; `canny` = thin/crisp but more broken |
| `--detail` | `0.5` | 0–1 line density / sensitivity |
| `--width-in` | `8.0` | final drawing width in inches (sets physical size in Design Space) |
| `--max-px` | `2000` | resolution cap; **raise to 2400–2800 to recover fine detail** from large photos |
| `--xdog-sigma` | `0.8` | XDoG line scale; **lower to 0.5–0.6 for finer facial detail** |
| `--clahe` | `2.0` | local-contrast boost; **raise to 3–4 for flat / sun-washed photos** |
| `--clahe-tiles` | `8` | CLAHE grid; smaller = stronger local contrast |
| `--merge-mm` | `1.0` | bridge gaps between stroke ends; raise to reconnect dashed lines |
| `--simplify` | `0.3` | vpype linesimplify tolerance (mm) — higher = smoother, fewer nodes |
| `--min-line-mm` | `1.0` | drop plotted segments shorter than this (kills noise) |
| `--despeckle` | `12` | remove line-blobs smaller than N pixels |
| `--smooth` | `2` | moving-average smoothing passes |
| `--detail` | `0.5` | 0–1 line density / sensitivity |

### Tuning cheat-sheet
- **Too busy / hairy / noisy** → lower `--detail`, raise `--min-line-mm`, raise `--despeckle`
- **Missing detail (faces look empty)** → see the recipe below
- **Lines come out dashed** → raise `--merge-mm` (try 1.5–2.0)
- **Jaggy / staircase lines** → raise `--smooth` and/or `--simplify`
- **Wrong physical size in Cricut** → set `--width-in` to the inches you want it drawn

### Recovering detail (large or sun-washed photos)
Edge detection on a big, brightly lit photo loses fine detail two ways: the default
resolution cap downsamples it before tracing, and flat lighting gives faces little
local contrast. Counter both:
```bash
photo2cricut photo.jpg out.svg \
  --max-px 2600 --xdog-sigma 0.6 --clahe 3.5 \
  --despeckle 4 --min-line-mm 0.5 --simplify 0.25 --detail 0.55
```
`--max-px` is the biggest lever (process at full resolution); `--xdog-sigma 0.6`
captures finer lines; `--clahe 3.5` rescues washed-out faces. Keep `--detail` modest
in busy scenes — cranking it adds background texture (rock, sand, foliage) faster
than facial detail. **Trade-off:** more detail = more strokes = longer draw time and
heavier Design Space import. The validator warns past ~3000 strokes.

For the cleanest *portrait* faces specifically, a flat-lit selfie has a hard ceiling
for edge detection — the two-stage AI approach below beats any settings here.

### Getting the cleanest portraits (optional two-stage)
Classic CV (XDoG/Canny) is good but not magic on faces. For the cleanest result,
swap the **front** of the pipeline: run your photo through an AI "minimalist outline"
line-art tool first to get a clean black-on-white PNG, then feed **that** into this
tool. The skeletonize → centerline → vpype tail only needs black lines on white, so
it doesn't care whether they came from XDoG or an AI model:
```bash
photo2cricut clean_lineart.png out.svg --method canny --detail 0.7
```

## Import into Cricut Design Space
1. **Upload** → select the generated `.svg`.
2. Select the layer, change the operation from **Cut** to **Draw / Pen**.
3. Pick a fine-point pen (a 0.4 mm draws cleanest; gel pens lay smoother ink).
4. Test on cardstock first; tape the paper to the mat (friction feed drifts on detail).

Expect ~2–5 minutes of touch-up in Inkscape around the eyes/mouth on portraits —
that's where the likeness lives and where automated tracing is least forgiving.

## Validate an output
```bash
photo2cricut-validate out.svg
```
Checks the SVG parses, has real inch/mm units + a viewBox, contains stroke geometry,
has no pen-unfriendly fills, and isn't so dense it will choke Design Space.

## Testing
```bash
./install.sh --dev      # installs dev deps and runs the suite
# or, with the venv active:
pytest -q
```
The tests generate a synthetic portrait (no real/copyrighted image is bundled),
run both methods end-to-end, and assert the output validates as Cricut-ready.

## Project layout
```
photo2cricut/
├── README.md
├── LICENSE                  (MIT — fill in your name)
├── requirements.txt / requirements-dev.txt
├── pyproject.toml           (installable; defines the console commands)
├── install.sh / install.ps1 (venv + deps + smoke test)
├── Dockerfile / Makefile
├── photo2cricut/            (the package)
│   ├── pipeline.py          (core convert() flow)
│   ├── cli.py               (photo2cricut command)
│   ├── validate.py          (photo2cricut-validate command)
│   └── testimage.py         (photo2cricut-makeimg command)
├── scripts/
│   └── make_test_image.py   (shim to run the generator from a checkout)
├── tests/
│   └── test_pipeline.py
└── examples/                (generated images/SVGs land here; git-ignored)
```

## Console commands installed
- `photo2cricut` — convert a photo to SVG
- `photo2cricut-validate` — check an SVG is Cricut-ready
- `photo2cricut-makeimg` — generate the synthetic test portrait

## Troubleshooting
- **`vpype` not found** — the pipeline calls it as `python -m vpype_cli`, so just make
  sure you're inside the `.venv` the installer created.
- **Design Space is slow to import / hangs** — the SVG has too many strokes. Raise
  `--simplify`, `--min-line-mm`, and `--despeckle`; the validator warns past ~3000.
- **PNG preview tools fail** — `cairosvg` (and a system cairo lib) are only needed for
  previews, not for producing the SVG.

## License
MIT — see `LICENSE` (replace `YOUR NAME`).
