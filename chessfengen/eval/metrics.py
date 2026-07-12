"""Evaluation metrics for board recognition, independent of any ML framework.

Square level metrics operate on sequences of class indices (see
grid_to_class_indices), so they can be unit tested without torch. Corner
metrics use numpy. Keeping the metrics framework free lets the same functions
grade a torch model, a future ONNX export, or a hand labelled sanity set.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from chessfengen.fen.placement import BOARD_SIZE
from chessfengen.pieces import INDEX_TO_CLASS, NUM_CLASSES

NUM_SQUARES: int = BOARD_SIZE * BOARD_SIZE
NUM_CORNERS: int = 4
EMPTY_LABEL: str = "empty"


def square_accuracy(pred: Sequence[int], true: Sequence[int]) -> float:
    """Fraction of the squares whose predicted class matches the truth."""
    _check_same_nonempty_length(pred, true)
    correct: int = sum(1 for p, t in zip(pred, true) if p == t)
    return correct / len(true)


def board_exact_match(pred: Sequence[int], true: Sequence[int]) -> bool:
    """True only when every square on the board is predicted correctly."""
    _check_same_nonempty_length(pred, true)
    return all(p == t for p, t in zip(pred, true))


class ConfusionMatrix:
    """Accumulates a class by class confusion matrix over many boards.

    Rows are truth, columns are prediction. This surfaces the failure modes
    that matter for chess: for example empty squares misread as pieces, or one
    piece type systematically confused with another.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        self._num_classes: int = num_classes
        self._counts: list[list[int]] = [[0] * num_classes for _ in range(num_classes)]

    def update(self, pred: Sequence[int], true: Sequence[int]) -> None:
        """Add one board (or any equal length index sequences) to the tally."""
        _check_same_nonempty_length(pred, true)
        for predicted, actual in zip(pred, true):
            self._counts[actual][predicted] += 1

    def total(self) -> int:
        """Total number of squares counted so far."""
        return sum(sum(row) for row in self._counts)

    def overall_accuracy(self) -> float:
        """Fraction of all counted squares on the diagonal."""
        total: int = self.total()
        if total == 0:
            raise ValueError("no squares have been counted")
        correct: int = sum(self._counts[index][index] for index in range(self._num_classes))
        return correct / total

    def per_class_recall(self) -> dict[str, float]:
        """Recall for each class label; NaN for classes with no support."""
        recall: dict[str, float] = {}
        for actual in range(self._num_classes):
            support: int = sum(self._counts[actual])
            label: str = _label_for(actual)
            recall[label] = (self._counts[actual][actual] / support) if support else float("nan")
        return recall

    def support(self) -> dict[str, int]:
        """Number of ground truth squares seen for each class label."""
        return {_label_for(actual): sum(self._counts[actual]) for actual in range(self._num_classes)}


def corner_l2_distances(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per corner Euclidean distance, shape (N, 4).

    Accepts corners as (N, 8) flat coordinates or (N, 4, 2) point arrays. Units
    are whatever the inputs use; pass normalised [0, 1] coordinates to get an
    error as a fraction of the image size.
    """
    pred_points: np.ndarray = _as_corner_points(pred)
    true_points: np.ndarray = _as_corner_points(true)
    if pred_points.shape != true_points.shape:
        raise ValueError(f"shape mismatch: {pred_points.shape} vs {true_points.shape}")
    return np.linalg.norm(pred_points - true_points, axis=2)


def mean_corner_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean per corner Euclidean distance across every corner and board."""
    return float(corner_l2_distances(pred, true).mean())


def _as_corner_points(array: np.ndarray) -> np.ndarray:
    points: np.ndarray = np.asarray(array, dtype=np.float64)
    if points.ndim == 2 and points.shape[1] == NUM_CORNERS * 2:
        return points.reshape(-1, NUM_CORNERS, 2)
    if points.ndim == 3 and points.shape[1:] == (NUM_CORNERS, 2):
        return points
    raise ValueError(f"expected (N, 8) or (N, 4, 2) corners, got shape {points.shape}")


def _label_for(index: int) -> str:
    label: str = INDEX_TO_CLASS[index]
    return label if label else EMPTY_LABEL


def _check_same_nonempty_length(pred: Sequence[int], true: Sequence[int]) -> None:
    if len(pred) != len(true):
        raise ValueError(f"length mismatch: {len(pred)} vs {len(true)}")
    if len(true) == 0:
        raise ValueError("cannot score empty sequences")
