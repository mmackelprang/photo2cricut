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
