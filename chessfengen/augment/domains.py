"""Domain augmentations that turn a clean render into a realistic input.

Three domains are modelled, matching the "mix of all" target:

  SCREENSHOT  clean digital capture: light blur, resampling, JPEG artifacts.
  SCAN        photocopied book diagram: grayscale, contrast shift, speckle,
              small in-plane rotation.
  PHOTO       photo of a screen or page: perspective warp, lighting gradient,
              blur, JPEG artifacts.

Every geometric transform (rotation, perspective) is also applied to the board
corner coordinates so Stage 1 targets stay exact. Augmentation magnitudes are
scaled by NoiseLevel.intensity.

All functions take and return an image as an HxWx3 uint8 BGR numpy array (the
OpenCV convention) plus a (4, 2) float32 corners array.
"""

from __future__ import annotations

import enum
import random

import cv2
import numpy as np

from chessfengen.augment.noise import NoiseLevel


class Domain(enum.Enum):
    """The capture domain an augmented sample imitates."""

    SCREENSHOT = "screenshot"
    SCAN = "scan"
    PHOTO = "photo"


def augment(
    image: np.ndarray,
    corners: np.ndarray,
    domain: Domain,
    level: NoiseLevel,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the augmentation pipeline for one domain at one noise level."""
    if domain is Domain.SCREENSHOT:
        return _augment_screenshot(image, corners, level, rng)
    if domain is Domain.SCAN:
        return _augment_scan(image, corners, level, rng)
    if domain is Domain.PHOTO:
        return _augment_photo(image, corners, level, rng)
    raise ValueError(f"unhandled domain: {domain!r}")


def _augment_screenshot(
    image: np.ndarray, corners: np.ndarray, level: NoiseLevel, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    strength: float = level.intensity
    out: np.ndarray = _maybe_blur(image, strength, rng, max_sigma=1.2)
    out = _jpeg_recompress(out, strength, rng, min_quality=40)
    return out, corners.copy()


def _augment_scan(
    image: np.ndarray, corners: np.ndarray, level: NoiseLevel, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    strength: float = level.intensity
    gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    out: np.ndarray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out = _apply_contrast_brightness(out, strength, rng)
    out, corners = _rotate(out, corners, strength, rng, max_degrees=4.0)
    out = _add_speckle(out, strength, rng)
    out = _jpeg_recompress(out, strength, rng, min_quality=30)
    return out, corners


def _augment_photo(
    image: np.ndarray, corners: np.ndarray, level: NoiseLevel, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    strength: float = level.intensity
    out, corners = _perspective_warp(image, corners, strength, rng)
    out = _apply_lighting_gradient(out, strength, rng)
    out = _maybe_blur(out, strength, rng, max_sigma=2.0)
    out = _jpeg_recompress(out, strength, rng, min_quality=35)
    return out, corners


def _maybe_blur(
    image: np.ndarray, strength: float, rng: random.Random, max_sigma: float
) -> np.ndarray:
    sigma: float = rng.uniform(0.0, max_sigma * strength)
    if sigma < 1e-3:
        return image
    return cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigma)


def _jpeg_recompress(
    image: np.ndarray, strength: float, rng: random.Random, min_quality: int
) -> np.ndarray:
    if strength <= 0.0:
        return image
    quality: int = int(round(_lerp(95, min_quality, strength * rng.uniform(0.5, 1.0))))
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed during JPEG recompression")
    decoded: np.ndarray = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return decoded


def _apply_contrast_brightness(
    image: np.ndarray, strength: float, rng: random.Random
) -> np.ndarray:
    alpha: float = 1.0 + rng.uniform(-0.4, 0.4) * strength
    beta: float = rng.uniform(-40.0, 40.0) * strength
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def _add_speckle(image: np.ndarray, strength: float, rng: random.Random) -> np.ndarray:
    if strength <= 0.0:
        return image
    sigma: float = 30.0 * strength
    noise: np.ndarray = np.random.default_rng(rng.randrange(2**32)).normal(
        0.0, sigma, image.shape
    )
    noisy: np.ndarray = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _apply_lighting_gradient(
    image: np.ndarray, strength: float, rng: random.Random
) -> np.ndarray:
    if strength <= 0.0:
        return image
    height, width = image.shape[:2]
    axis_x: np.ndarray = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    axis_y: np.ndarray = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis_x, axis_y)
    direction_x: float = rng.uniform(-1.0, 1.0)
    direction_y: float = rng.uniform(-1.0, 1.0)
    gradient: np.ndarray = grid_x * direction_x + grid_y * direction_y
    scale: np.ndarray = 1.0 + gradient * (0.35 * strength)
    lit: np.ndarray = image.astype(np.float32) * scale[:, :, np.newaxis]
    return np.clip(lit, 0, 255).astype(np.uint8)


def _rotate(
    image: np.ndarray,
    corners: np.ndarray,
    strength: float,
    rng: random.Random,
    max_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    angle: float = rng.uniform(-max_degrees, max_degrees) * strength
    if abs(angle) < 1e-3:
        return image, corners.copy()
    height, width = image.shape[:2]
    center: tuple[float, float] = (width / 2.0, height / 2.0)
    matrix: np.ndarray = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated: np.ndarray = cv2.warpAffine(
        image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, _transform_affine(corners, matrix)


def _perspective_warp(
    image: np.ndarray, corners: np.ndarray, strength: float, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    if strength <= 0.0:
        return image, corners.copy()
    height, width = image.shape[:2]
    max_shift: float = 0.12 * strength * min(width, height)
    source: np.ndarray = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    jitter: np.ndarray = np.array(
        [[rng.uniform(-max_shift, max_shift) for _ in range(2)] for _ in range(4)],
        dtype=np.float32,
    )
    destination: np.ndarray = source + jitter
    matrix: np.ndarray = cv2.getPerspectiveTransform(source, destination)
    warped: np.ndarray = cv2.warpPerspective(
        image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
    )
    return warped, _transform_perspective(corners, matrix)


def _transform_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous: np.ndarray = np.hstack([points, np.ones((len(points), 1), dtype=np.float32)])
    transformed: np.ndarray = homogeneous @ matrix.T
    return transformed.astype(np.float32)


def _transform_perspective(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous: np.ndarray = np.hstack([points, np.ones((len(points), 1), dtype=np.float32)])
    transformed: np.ndarray = homogeneous @ matrix.T
    transformed = transformed[:, :2] / transformed[:, 2:3]
    return transformed.astype(np.float32)


def _lerp(start: float, end: float, fraction: float) -> float:
    return start + (end - start) * fraction
