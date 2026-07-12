"""Tests for grid, placement field, and class index conversions."""

from __future__ import annotations

import random

import pytest

from chessfengen.fen.generator import generate_random_placement
from chessfengen.fen.placement import (
    class_indices_to_grid,
    grid_to_class_indices,
    grid_to_placement_field,
    placement_field_to_grid,
)
from chessfengen.pieces import NUM_CLASSES

STARTING_PLACEMENT: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def test_starting_position_round_trip() -> None:
    grid = placement_field_to_grid(STARTING_PLACEMENT)
    assert grid_to_placement_field(grid) == STARTING_PLACEMENT


def test_empty_board_round_trip() -> None:
    empty_field: str = "/".join(["8"] * 8)
    grid = placement_field_to_grid(empty_field)
    assert grid_to_placement_field(grid) == empty_field


def test_class_index_round_trip() -> None:
    grid = placement_field_to_grid(STARTING_PLACEMENT)
    indices = grid_to_class_indices(grid)
    assert len(indices) == 64
    assert all(0 <= index < NUM_CLASSES for index in indices)
    assert class_indices_to_grid(indices) == grid


def test_generated_placements_round_trip() -> None:
    rng = random.Random(1234)
    for _ in range(500):
        grid = generate_random_placement(rng)
        field = grid_to_placement_field(grid)
        assert placement_field_to_grid(field) == grid


def test_invalid_row_count_rejected() -> None:
    with pytest.raises(ValueError):
        placement_field_to_grid("8/8/8")


def test_invalid_character_rejected() -> None:
    with pytest.raises(ValueError):
        placement_field_to_grid("8/8/8/8/8/8/8/7x")


def test_row_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        placement_field_to_grid("9/8/8/8/8/8/8/8")
