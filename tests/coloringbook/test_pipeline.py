import os
import cv2
from pypdf import PdfReader
from photo2cricut.testimage import make_portrait
from photo2coloringbook import BookOptions, convert_book


def _make_dir(d, n):
    for i in range(n):
        cv2.imwrite(os.path.join(d, f"p{i}.jpg"), make_portrait(seed=i),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])


def test_convert_book_produces_cover_plus_pages(tmp_path):
    src = tmp_path / "photos"; src.mkdir()
    _make_dir(str(src), 2)
    out = str(tmp_path / "book.pdf")
    stats = convert_book(str(src), out, BookOptions(title="Fam", backend="cv", dpi=100))
    assert stats == {"pages": 3, "backend": "cv", "output": out}
    assert os.path.exists(out)
    assert len(PdfReader(out).pages) == 3
