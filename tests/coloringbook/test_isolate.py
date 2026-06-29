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
