"""Tests for random placement generation."""

from __future__ import annotations

import random

import pytest

from chessfengen.fen.generator import generate_random_placement
from chessfengen.pieces import EMPTY


def _piece_count(grid) -> int:
    return sum(1 for row in grid for cell in row if cell != EMPTY)


def test_piece_count_within_bounds() -> None:
    rng = random.Random(0)
    for _ in range(200):
        grid = generate_random_placement(rng, min_pieces=5, max_pieces=20)
        assert 5 <= _piece_count(grid) <= 20


def test_two_kings_present_when_forced() -> None:
    rng = random.Random(0)
    for _ in range(200):
        grid = generate_random_placement(rng, force_two_kings=True)
        flat = [cell for row in grid for cell in row]
        assert flat.count("K") >= 1
        assert flat.count("k") >= 1


def test_deterministic_with_seed() -> None:
    grid_a = generate_random_placement(random.Random(42))
    grid_b = generate_random_placement(random.Random(42))
    assert grid_a == grid_b


def test_invalid_bounds_rejected() -> None:
    with pytest.raises(ValueError):
        generate_random_placement(random.Random(0), min_pieces=10, max_pieces=5)


def test_force_kings_requires_min_two() -> None:
    with pytest.raises(ValueError):
        generate_random_placement(random.Random(0), min_pieces=1, force_two_kings=True)
