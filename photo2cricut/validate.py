"""photo2cricut.validate -- check that an SVG is Cricut Draw-ready.

Pass criteria:
  * parses as XML/SVG
  * root has explicit physical units (in / mm / cm) on width & height
  * has a viewBox
  * contains stroke geometry (path / polyline / line / polygon)
  * no filled regions (fill is none / unset) -- fills draw badly with a pen
  * stroke count is within a sane range for Design Space import

Usage:
  photo2cricut-validate out.svg
Exit code 0 on pass, 1 on fail.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET

GEOM_TAGS = {"path", "polyline", "line", "polygon"}
UNIT_RE = re.compile(r"^\s*[\d.]+\s*(in|mm|cm)\s*$", re.I)
# A very high element count makes Design Space slow/unstable on import.
MAX_RECOMMENDED_STROKES = 3000


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def validate_svg(path: str):
    """Return (ok: bool, report: list[str])."""
    report = []
    ok = True

    try:
        tree = ET.parse(path)
    except (ET.ParseError, FileNotFoundError) as e:
        return False, [f"FAIL  cannot parse SVG: {e}"]

    root = tree.getroot()
    if _localname(root.tag) != "svg":
        return False, ["FAIL  root element is not <svg>"]

    w, h, vb = root.get("width"), root.get("height"), root.get("viewBox")

    if w and h and UNIT_RE.match(w) and UNIT_RE.match(h):
        report.append(f"PASS  physical size: {w} x {h}")
    else:
        ok = False
        report.append(f"FAIL  width/height need explicit in/mm/cm units (got {w!r} x {h!r})")

    if vb:
        report.append(f"PASS  viewBox present: {vb}")
    else:
        ok = False
        report.append("FAIL  no viewBox")

    geom = [el for el in root.iter() if _localname(el.tag) in GEOM_TAGS]
    if geom:
        report.append(f"PASS  {len(geom)} stroke element(s) found")
    else:
        ok = False
        report.append("FAIL  no stroke geometry (path/polyline/line) found")

    # detect real fills (a literal color, not 'none')
    filled = 0
    for el in root.iter():
        f = (el.get("fill") or "").strip().lower()
        if f and f != "none":
            filled += 1
    if filled == 0:
        report.append("PASS  no filled regions (pen-friendly)")
    else:
        report.append(f"WARN  {filled} element(s) have a fill -- set fill:none for clean pen draw")

    if len(geom) > MAX_RECOMMENDED_STROKES:
        report.append(f"WARN  {len(geom)} strokes exceeds ~{MAX_RECOMMENDED_STROKES}; "
                      "Design Space may be slow. Raise --simplify / --min-line-mm / --despeckle.")
    return ok, report


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: photo2cricut-validate <file.svg>", file=sys.stderr)
        return 2
    ok, report = validate_svg(argv[0])
    print(f"\nValidating: {argv[0]}")
    for line in report:
        print("  " + line)
    print("\nRESULT:", "READY for Cricut Design Space" if ok else "NOT ready (see FAILs)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
