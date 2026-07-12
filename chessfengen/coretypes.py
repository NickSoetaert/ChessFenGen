"""Shared type aliases and data containers used across the pipeline.

Heavy third party types (PIL, numpy) are only referenced for annotations here,
so they are imported under TYPE_CHECKING to keep the pure FEN core importable
without those dependencies installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image

# An 8x8 board grid. Outer index is a row, row 0 is the top of the board
# (rank 8), matching the reading order of a FEN placement field. Each cell is
# either a FEN piece character or the EMPTY sentinel from chessfengen.pieces.
Grid = tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class RenderedBoard:
    """A rendered board plus the metadata needed to build training targets.

    corners holds the four board corner pixel coordinates in the order
    top-left, top-right, bottom-right, bottom-left, shape (4, 2), dtype float32.
    Every geometric augmentation must transform these corners with the same
    matrix it applies to the image so Stage 1 labels stay exact.
    """

    image: "Image.Image"
    corners: "np.ndarray"
    placement: str
