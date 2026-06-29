import numpy as np
import pytest
from photo2coloringbook.isolate import isolate


def test_keep_is_identity():
    img = np.zeros((10, 10, 3), np.uint8)
    assert isolate(img, "keep") is img


def test_remove_and_auto_not_yet_available():
    img = np.zeros((10, 10, 3), np.uint8)
    for mode in ("auto", "remove"):
        with pytest.raises(NotImplementedError):
            isolate(img, mode)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        isolate(np.zeros((10, 10, 3), np.uint8), "bogus")
