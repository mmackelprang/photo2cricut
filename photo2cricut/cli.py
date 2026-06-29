"""photo2cricut.cli -- command-line entry point for the converter."""

from __future__ import annotations

import argparse
import sys

from .pipeline import Options, convert


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="photo2cricut",
        description="Turn a JPG photo into a Cricut-ready single-line (centerline) SVG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", help="input image (jpg/png/...)")
    ap.add_argument("output", help="output .svg path")
    ap.add_argument("--method", choices=["xdog", "canny"], default="xdog",
                    help="xdog = drawn-looking lines; canny = thin crisp edges")
    ap.add_argument("--detail", type=float, default=0.5, help="0..1 line density / sensitivity")
    ap.add_argument("--width-in", type=float, default=8.0, help="final drawing width in inches")
    ap.add_argument("--max-px", type=int, default=2000, help="downscale longest side before processing")
    ap.add_argument("--despeckle", type=int, default=12, help="drop line-blobs smaller than N px")
    ap.add_argument("--smooth", type=int, default=2, help="smoothing passes (0=off)")
    ap.add_argument("--simplify", type=float, default=0.3, dest="simplify_mm",
                    help="vpype linesimplify tolerance (mm)")
    ap.add_argument("--merge-mm", type=float, default=1.0, dest="merge_mm",
                    help="bridge stroke gaps up to this distance (mm)")
    ap.add_argument("--min-line-mm", type=float, default=1.0, dest="min_line_mm",
                    help="drop plotted segments shorter than this (mm)")
    ap.add_argument("--xdog-sigma", type=float, default=0.8, dest="xdog_sigma",
                    help="XDoG base sigma; LOWER (0.5-0.6) = finer detail")
    ap.add_argument("--clahe", type=float, default=2.0, dest="clahe_clip",
                    help="local-contrast strength; RAISE (3-4) for flat/washed-out photos")
    ap.add_argument("--clahe-tiles", type=int, default=8, dest="clahe_tiles",
                    help="CLAHE grid size; smaller = more local contrast")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    opt = Options(
        method=args.method, detail=args.detail, width_in=args.width_in,
        max_px=args.max_px, despeckle=args.despeckle, smooth=args.smooth,
        simplify_mm=args.simplify_mm, merge_mm=args.merge_mm, min_line_mm=args.min_line_mm,
        xdog_sigma=args.xdog_sigma, clahe_clip=args.clahe_clip, clahe_tiles=args.clahe_tiles,
    )
    try:
        stats = convert(args.input, args.output, opt)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"OK -> {stats['output']}  ({stats['raw_strokes']} raw strokes, "
          f"method={stats['method']}, {stats['width_in']}in wide)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
