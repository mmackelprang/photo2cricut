"""photo2coloringbook -- photos to a print-ready PDF coloring book."""

from .pipeline import BookOptions, convert_book

__version__ = "0.1.0"
__all__ = ["BookOptions", "convert_book", "__version__"]
