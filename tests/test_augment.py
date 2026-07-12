"""Tests for domain augmentation shape, dtype, and corner tracking."""

from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from chessfengen.augment.domains import Domain, augment
from chessfengen.augment.noise import NoiseLevel

IMAGE_SIZE: int = 64
# Larger size for the rendered debug boards so pieces are legible.
BOARD_RENDER_SIZE: int = 256

# Where the augmentation debug images are written. Under scratch/ (gitignored).
_DEBUG_DIR: Path = Path(
    os.environ.get("CHESSFENGEN_AUGMENT_DEBUG_DIR", "scratch/augment_debug")
)
# cv2 colors are BGR: input corners green, output corners red.
_INPUT_CORNER_COLOR: tuple[int, int, int] = (0, 255, 0)
_OUTPUT_CORNER_COLOR: tuple[int, int, int] = (0, 0, 255)
_UPSCALE: int = 2


def _synthetic_board() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    corners = np.array(
        [[0, 0], [IMAGE_SIZE, 0], [IMAGE_SIZE, IMAGE_SIZE], [0, IMAGE_SIZE]],
        dtype=np.float32,
    )
    return image, corners


def _rendered_board(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Render a real training board (BGR uint8) plus its frame corners.

    This mirrors the actual data pipeline: a random placement rendered with a
    random piece set and theme. Skips when the render dependencies are absent.
    """
    pytest.importorskip("cairosvg")
    pytest.importorskip("chess")
    from chessfengen.fen.generator import generate_random_placement
    from chessfengen.fen.placement import grid_to_placement_field
    from chessfengen.render.board import random_theme, render_board
    from chessfengen.render.piece_sets import default_piece_sets

    grid = generate_random_placement(rng)
    rendered = render_board(
        placement=grid_to_placement_field(grid),
        piece_set=rng.choice(default_piece_sets()),
        theme=random_theme(rng),
        size=BOARD_RENDER_SIZE,
        rng=rng,
    )
    # PIL RGB to BGR numpy, the convention augment (cv2) expects.
    image_bgr: np.ndarray = np.asarray(rendered.image.convert("RGB"))[:, :, ::-1].copy()
    return image_bgr, rendered.corners


def _draw_corners(
    image: np.ndarray, corners: np.ndarray, color: tuple[int, int, int]
) -> np.ndarray:
    """Return a copy of image with the corner quadrilateral and points drawn."""
    canvas: np.ndarray = np.ascontiguousarray(image.copy())
    points: np.ndarray = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [points], isClosed=True, color=color, thickness=1)
    for x, y in corners.astype(np.int32):
        cv2.circle(canvas, (int(x), int(y)), radius=2, color=color, thickness=-1)
    return canvas


def _save_augment_debug(
    domain: Domain,
    level: NoiseLevel,
    image: np.ndarray,
    corners: np.ndarray,
    out_image: np.ndarray,
    out_corners: np.ndarray,
) -> None:
    """Write a labeled input vs output side by side PNG for visual inspection.

    The input (green corners) and augmented output (red corners) are placed side
    by side, upscaled for legibility, and captioned with the domain and level.
    Images are BGR uint8, which is what cv2.imwrite expects.
    """
    _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    left: np.ndarray = _draw_corners(image, corners, _INPUT_CORNER_COLOR)
    right: np.ndarray = _draw_corners(out_image, out_corners, _OUTPUT_CORNER_COLOR)
    separator: np.ndarray = np.full((image.shape[0], 2, 3), 255, dtype=np.uint8)
    combined: np.ndarray = np.hstack([left, separator, right])
    combined = cv2.resize(
        combined, None, fx=_UPSCALE, fy=_UPSCALE, interpolation=cv2.INTER_NEAREST
    )
    header: np.ndarray = np.zeros((28, combined.shape[1], 3), dtype=np.uint8)
    label: str = f"{domain.value} / {level.value}  (green=input, red=output corners)"
    cv2.putText(
        header, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
    )
    combined = np.vstack([header, combined])
    cv2.imwrite(str(_DEBUG_DIR / f"{domain.value}_{level.value}.png"), combined)


@pytest.mark.parametrize("domain", list(Domain))
@pytest.mark.parametrize("level", list(NoiseLevel))
def test_augment_preserves_shape_and_dtype(domain: Domain, level: NoiseLevel) -> None:
    # A distinct but reproducible board per combo, so the nine debug images show
    # different positions, piece sets, and themes across the augmentation grid.
    rng: random.Random = random.Random(f"{domain.value}-{level.value}")
    image, corners = _rendered_board(rng)
    out_image, out_corners = augment(image, corners, domain, level, rng)
    _save_augment_debug(domain, level, image, corners, out_image, out_corners)
    assert out_image.shape == image.shape
    assert out_image.dtype == np.uint8
    assert out_corners.shape == (4, 2)
    assert out_corners.dtype == np.float32


def test_no_noise_screenshot_is_identity_geometry() -> None:
    image, corners = _synthetic_board()
    _, out_corners = augment(image, corners, Domain.SCREENSHOT, NoiseLevel.NONE, random.Random(1))
    assert np.allclose(out_corners, corners)


def test_photo_warp_moves_corners() -> None:
    image, corners = _synthetic_board()
    _, out_corners = augment(image, corners, Domain.PHOTO, NoiseLevel.HEAVY, random.Random(3))
    assert not np.allclose(out_corners, corners)
