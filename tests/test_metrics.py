"""Tests for the framework independent evaluation metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from chessfengen.eval.metrics import (
    ConfusionMatrix,
    board_exact_match,
    corner_l2_distances,
    mean_corner_error,
    square_accuracy,
)
from chessfengen.fen.placement import grid_to_class_indices, placement_field_to_grid

STARTING_PLACEMENT: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def _starting_indices() -> list[int]:
    return list(grid_to_class_indices(placement_field_to_grid(STARTING_PLACEMENT)))


def test_square_accuracy_perfect() -> None:
    truth = _starting_indices()
    assert square_accuracy(truth, truth) == 1.0


def test_square_accuracy_partial() -> None:
    truth = _starting_indices()
    pred = list(truth)
    pred[0] = (pred[0] + 1) % 13
    pred[1] = (pred[1] + 1) % 13
    assert square_accuracy(pred, truth) == pytest.approx(62 / 64)


def test_board_exact_match() -> None:
    truth = _starting_indices()
    assert board_exact_match(truth, truth) is True
    wrong = list(truth)
    wrong[10] = (wrong[10] + 1) % 13
    assert board_exact_match(wrong, truth) is False


def test_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        square_accuracy([1, 2, 3], [1, 2])


def test_empty_sequence_rejected() -> None:
    with pytest.raises(ValueError):
        square_accuracy([], [])


def test_confusion_matrix_accuracy_and_recall() -> None:
    confusion = ConfusionMatrix()
    truth = _starting_indices()
    perfect = ConfusionMatrix()
    perfect.update(truth, truth)
    assert perfect.overall_accuracy() == 1.0

    # Two boards: one perfect, one with two empty squares misread.
    confusion.update(truth, truth)
    corrupted = list(truth)
    # squares 16 and 17 are empty (rank 6) in the starting position; flip them.
    corrupted[16] = 1
    corrupted[17] = 1
    confusion.update(corrupted, truth)
    assert confusion.total() == 128
    assert confusion.overall_accuracy() == pytest.approx(126 / 128)
    recall = confusion.per_class_recall()
    # 32 empty squares per board, 64 total, 2 of the second board's got misread.
    assert recall["empty"] == pytest.approx(62 / 64)


def test_confusion_matrix_requires_data() -> None:
    with pytest.raises(ValueError):
        ConfusionMatrix().overall_accuracy()


def test_per_class_recall_nan_without_support() -> None:
    confusion = ConfusionMatrix()
    confusion.update([1], [1])
    recall = confusion.per_class_recall()
    assert recall["P"] == 1.0
    assert math.isnan(recall["empty"])


def test_corner_distances_zero_when_identical() -> None:
    corners = np.array([[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]], dtype=np.float32)
    assert mean_corner_error(corners, corners) == 0.0


def test_corner_distances_known_offset() -> None:
    truth = np.array([[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]], dtype=np.float32)
    pred = truth.copy()
    pred[0, 0] = 0.3  # move top-left corner x by 0.3
    distances = corner_l2_distances(pred, truth)
    assert distances.shape == (1, 4)
    assert distances[0, 0] == pytest.approx(0.3)
    assert mean_corner_error(pred, truth) == pytest.approx(0.3 / 4)


def test_corner_shape_accepts_point_array() -> None:
    flat = np.array([[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]], dtype=np.float32)
    points = flat.reshape(1, 4, 2)
    assert np.allclose(corner_l2_distances(flat, points), 0.0)


def test_corner_bad_shape_rejected() -> None:
    with pytest.raises(ValueError):
        corner_l2_distances(np.zeros((1, 5)), np.zeros((1, 5)))
