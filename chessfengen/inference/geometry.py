"""Geometric operations for the inference pipeline (cv2 and numpy only).

Kept free of torch so the perspective dewarp can be unit tested without the
model dependencies. Corner ordering is top-left, top-right, bottom-right,
bottom-left throughout, matching the labels produced by the renderer and
consumed by Stage 1.
"""

from __future__ import annotations

import cv2
import numpy as np

NUM_CORNERS: int = 4

# Canonical destination corners as fractions of the output size, in the
# top-left, top-right, bottom-right, bottom-left order used everywhere.
_UNIT_CORNERS: np.ndarray = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
)


def denormalize_corners(corners: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale corners in [0, 1] to pixel coordinates, shape (4, 2) float32.

    Accepts either (8,) flat or (4, 2) point inputs.
    """
    points: np.ndarray = np.asarray(corners, dtype=np.float32).reshape(NUM_CORNERS, 2)
    scaled: np.ndarray = points.copy()
    scaled[:, 0] *= width
    scaled[:, 1] *= height
    return scaled


def dewarp_board(image: np.ndarray, corners_px: np.ndarray, size: int) -> np.ndarray:
    """Perspective warp the quadrilateral at corners_px to a size x size board.

    image is an HxWx3 array; corners_px is (4, 2) pixel coordinates in the
    top-left, top-right, bottom-right, bottom-left order. Returns the canonical
    axis aligned board image.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    source: np.ndarray = np.asarray(corners_px, dtype=np.float32).reshape(NUM_CORNERS, 2)
    destination: np.ndarray = _UNIT_CORNERS * float(size)
    matrix: np.ndarray = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, matrix, (size, size))
