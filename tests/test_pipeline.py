"""End-to-end tests: photo -> SVG -> Cricut-readiness validation."""

import os

import cv2
import pytest

from photo2cricut import Options, convert, validate_svg
from photo2cricut.testimage import make_portrait


@pytest.fixture(scope="module")
def test_jpg(tmp_path_factory):
    p = tmp_path_factory.mktemp("img") / "portrait.jpg"
    cv2.imwrite(str(p), make_portrait(seed=0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    assert p.exists()
    return str(p)


@pytest.mark.parametrize("method", ["xdog", "canny"])
def test_convert_produces_ready_svg(test_jpg, tmp_path, method):
    out = str(tmp_path / f"out_{method}.svg")
    stats = convert(test_jpg, out, Options(method=method, width_in=8.0))
    assert os.path.exists(out)
    assert stats["raw_strokes"] > 0

    ok, report = validate_svg(out)
    assert ok, "validation failed:\n" + "\n".join(report)


def test_width_controls_physical_size(test_jpg, tmp_path):
    out = str(tmp_path / "sized.svg")
    convert(test_jpg, out, Options(width_in=6.0))
    text = open(out).read()
    assert 'width="6.0in"' in text
    assert "height=" in text and "in" in text


def test_no_fills_present(test_jpg, tmp_path):
    out = str(tmp_path / "nofill.svg")
    convert(test_jpg, out, Options())
    text = open(out).read().lower()
    # the only 'fill' allowed is fill="none"
    assert "fill=\"none\"" in text
    assert "fill=\"#" not in text


def test_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert(str(tmp_path / "nope.jpg"), str(tmp_path / "x.svg"), Options())


def test_validate_rejects_outline_svg(tmp_path):
    bad = tmp_path / "bad.svg"
    bad.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
        '<rect width="10" height="10" fill="#000"/></svg>'
    )
    ok, _ = validate_svg(str(bad))
    assert not ok  # no units, no viewBox, no stroke geometry
