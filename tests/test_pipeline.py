"""Tests for the end to end BoardRecognizer pipeline.

Two kinds of check: a plumbing test with real (untrained) models that confirms
the pipeline always produces a valid, well formed result, and a determinism
test with stub models that confirms the corner to dewarp to assemble wiring is
correct independent of any learned weights.

Skipped automatically when torch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
from torch import nn

from chessfengen.fen.placement import grid_to_class_indices, placement_field_to_grid
from chessfengen.inference.pipeline import BoardPrediction, BoardRecognizer
from chessfengen.models.stage1_detector import CornerDetector
from chessfengen.models.stage2_classifier import SquareClassifier
from chessfengen.pieces import NUM_CLASSES

BOARD_SIZE: int = 128
KNOWN_PLACEMENT: str = "r1bk3r/p2pBpNp/n4n2/1p1NP2P/6P1/3P4/P1P1K3/q5b1"


def _synthetic_image(width: int = 300, height: int = 220) -> Image.Image:
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def test_predict_produces_valid_result() -> None:
    recognizer = BoardRecognizer(
        CornerDetector(), SquareClassifier(), device="cpu", board_size=BOARD_SIZE
    )
    prediction = recognizer.predict(_synthetic_image())
    assert isinstance(prediction, BoardPrediction)
    # placement must be a well formed field that parses to an 8x8 grid.
    grid = placement_field_to_grid(prediction.placement)
    assert len(grid) == 8 and all(len(row) == 8 for row in grid)
    assert prediction.corners.shape == (4, 2)
    assert prediction.square_confidence.shape == (64,)
    assert float(prediction.square_confidence.min()) >= 0.0
    assert float(prediction.square_confidence.max()) <= 1.0
    assert prediction.dewarped.shape == (BOARD_SIZE, BOARD_SIZE, 3)


def test_predict_placement_matches_predict() -> None:
    recognizer = BoardRecognizer(
        CornerDetector(), SquareClassifier(), device="cpu", board_size=BOARD_SIZE
    )
    image = _synthetic_image()
    assert recognizer.predict_placement(image) == recognizer.predict(image).placement


class _StubDetector(nn.Module):
    """Always reports the board fills the whole frame."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        corners = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        return corners.unsqueeze(0).expand(batch, -1)


class _StubClassifier(nn.Module):
    """Always predicts a fixed placement via one hot logits."""

    def __init__(self, indices: tuple[int, ...]) -> None:
        super().__init__()
        logits = torch.zeros(len(indices), NUM_CLASSES)
        for square, class_index in enumerate(indices):
            logits[square, class_index] = 10.0
        self._logits = logits

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        return self._logits.unsqueeze(0).expand(batch, -1, -1)


def test_stub_models_recover_known_board() -> None:
    indices = grid_to_class_indices(placement_field_to_grid(KNOWN_PLACEMENT))
    recognizer = BoardRecognizer(
        _StubDetector(), _StubClassifier(indices), device="cpu", board_size=BOARD_SIZE
    )
    prediction = recognizer.predict(_synthetic_image(width=256, height=256))
    # Deterministic wiring: exact placement is recovered, corners span the frame.
    assert prediction.placement == KNOWN_PLACEMENT
    assert np.allclose(prediction.corners, [[0, 0], [256, 0], [256, 256], [0, 256]])
