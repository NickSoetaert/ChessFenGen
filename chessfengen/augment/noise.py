"""Noise intensity levels applied to rendered boards.

The three levels model the degradation continuum from a pristine digital render
to a heavily photocopied book diagram. Each level exposes a scalar in [0, 1]
that domain augmentations multiply their effect magnitudes by.
"""

from __future__ import annotations

import enum


class NoiseLevel(enum.Enum):
    """Discrete noise budget for an augmentation pass."""

    NONE = "none"
    SOME = "some"
    HEAVY = "heavy"

    @property
    def intensity(self) -> float:
        """Scalar magnitude multiplier for this level."""
        return _INTENSITY[self]


_INTENSITY: dict[NoiseLevel, float] = {
    NoiseLevel.NONE: 0.0,
    NoiseLevel.SOME: 0.4,
    NoiseLevel.HEAVY: 1.0,
}
