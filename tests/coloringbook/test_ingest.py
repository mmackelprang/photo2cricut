import os
import cv2
import pytest
from photo2cricut.testimage import make_portrait
from photo2coloringbook.ingest import load_images, LoadedImage


def _write_imgs(d, n):
    for i in range(n):
        cv2.imwrite(os.path.join(d, f"p{i}.jpg"), make_portrait(seed=i),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])


def test_loads_all_images_sorted(tmp_path):
    _write_imgs(str(tmp_path), 3)
    (tmp_path / "notes.txt").write_text("ignore me")
    imgs = load_images(str(tmp_path))
    assert len(imgs) == 3
    assert all(isinstance(x, LoadedImage) for x in imgs)
    assert imgs[0].bgr.ndim == 3 and imgs[0].bgr.shape[2] == 3
    assert [os.path.basename(x.path) for x in imgs] == ["p0.jpg", "p1.jpg", "p2.jpg"]


def test_empty_dir_raises(tmp_path):
    with pytest.raises(RuntimeError):
        load_images(str(tmp_path))


def test_not_a_directory_raises(tmp_path):
    f = tmp_path / "x.jpg"
    f.write_bytes(b"not an image")
    with pytest.raises(NotADirectoryError):
        load_images(str(f))
