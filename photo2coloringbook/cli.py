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
    ap.add_argument("--ink-level", type=int, default=110, dest="ink_level",
                    help="contour backend: higher = less ink / cleaner faces (drops soft shading)")
    ap.add_argument("--paper", choices=["letter"], default="letter", help="page size")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    opt = BookOptions(title=args.title, bg=args.bg, backend=args.backend,
                      line_weight=args.line_weight, ink_level=args.ink_level, paper=args.paper)
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
