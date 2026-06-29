import numpy as np
import pytest
from pypdf import PdfReader
from photo2coloringbook.book import build_pdf


def test_build_pdf_one_page_per_raster(tmp_path):
    rasters = [np.full((330, 255), 255, np.uint8) for _ in range(3)]
    out = str(tmp_path / "book.pdf")
    build_pdf(rasters, out)
    assert len(PdfReader(out).pages) == 3


def test_build_pdf_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        build_pdf([], str(tmp_path / "x.pdf"))
