"""Tests for the inference geometry helpers (no torch required)."""

from __future__ import annotations

import numpy as np
import pytest

from chessfengen.inference.geometry import denormalize_corners, dewarp_board


def test_denormalize_corners_scales_to_pixels() -> None:
    normalized = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)
    pixels = denormalize_corners(normalized, width=200, height=100)
    assert pixels.shape == (4, 2)
    expected = np.array([[0, 0], [200, 0], [200, 100], [0, 100]], dtype=np.float32)
    assert np.allclose(pixels, expected)


def test_dewarp_full_frame_is_identity() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(60, 60, 3), dtype=np.uint8)
    corners = np.array([[0, 0], [60, 0], [60, 60], [0, 60]], dtype=np.float32)
    out = dewarp_board(image, corners, size=60)
    assert out.shape == (60, 60, 3)
    # A full frame to full frame warp of the same size is the identity.
    assert np.array_equal(out, image)


def test_dewarp_extracts_subrectangle() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Paint the inner 40x40 region a solid colour and dewarp exactly that quad.
    image[30:70, 30:70] = (10, 200, 50)
    corners = np.array([[30, 30], [70, 30], [70, 70], [30, 70]], dtype=np.float32)
    out = dewarp_board(image, corners, size=40)
    assert out.shape == (40, 40, 3)
    # The extracted region should be dominated by the painted colour.
    assert np.allclose(out[20, 20], (10, 200, 50), atol=5)
    assert float((out == 0).mean()) < 0.05


def test_dewarp_rejects_nonpositive_size() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    corners = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    with pytest.raises(ValueError):
        dewarp_board(image, corners, size=0)
