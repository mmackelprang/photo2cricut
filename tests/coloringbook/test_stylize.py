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
