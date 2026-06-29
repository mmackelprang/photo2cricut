import os
import cv2
from photo2cricut.testimage import make_portrait
from photo2coloringbook.cli import main


def test_cli_generates_pdf(tmp_path):
    src = tmp_path / "photos"; src.mkdir()
    cv2.imwrite(str(src / "a.jpg"), make_portrait(seed=0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    out = str(tmp_path / "out.pdf")
    rc = main([str(src), out, "--title", "Test", "--backend", "cv"])
    assert rc == 0
    assert os.path.exists(out)


def test_cli_bad_dir_returns_1(tmp_path, capsys):
    rc = main([str(tmp_path / "nope"), str(tmp_path / "o.pdf")])
    assert rc == 1
    assert "error:" in capsys.readouterr().err
