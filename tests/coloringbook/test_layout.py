import numpy as np
from photo2coloringbook.layout import fit_canvas, make_cover


def test_fit_canvas_sizes_and_centers():
    art = np.zeros((100, 200), np.uint8)            # 2:1 black art
    page = fit_canvas(art, page_w=1000, page_h=1000, margin=50)
    assert page.shape == (1000, 1000)
    assert page[0, 0] == 255 and page[-1, -1] == 255  # corners are white margin
    assert (page == 0).any()                          # art placed


def test_make_cover_draws_title():
    cover = make_cover("Hello", page_w=800, page_h=1000)
    assert cover.shape == (1000, 800)
    assert (cover < 128).any()                        # some dark title pixels
