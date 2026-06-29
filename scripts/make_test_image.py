"""Thin shim so `python scripts/make_test_image.py out.jpg` works from a checkout.
The real implementation lives in photo2cricut/testimage.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from photo2cricut.testimage import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
