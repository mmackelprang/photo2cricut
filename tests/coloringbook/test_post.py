import numpy as np
from photo2coloringbook.post import clean


def test_clean_removes_speckles_keeps_shapes_and_is_black_on_white():
    img = np.full((120, 120), 255, np.uint8)      # white page
    img[20:100, 20:100] = 0                        # solid black block (big component)
    img[5, 5] = 0                                  # 1-px speckle (should be removed)
    out = clean(img, line_weight=1, despeckle=12)
    assert out.dtype == np.uint8 and out.ndim == 2
    assert out[5, 5] == 255                         # speckle gone
    assert (out[20:100, 20:100] == 0).any()         # block survives
    assert (out == 255).any() and (out == 0).any()  # black-on-white


def test_line_weight_thickens():
    img = np.full((60, 60), 255, np.uint8)
    img[30, 10:50] = 0                              # 1-px horizontal line
    thin = clean(img, line_weight=1, despeckle=0)
    thick = clean(img, line_weight=5, despeckle=0)
    assert (thick == 0).sum() > (thin == 0).sum()   # more ink after thickening
