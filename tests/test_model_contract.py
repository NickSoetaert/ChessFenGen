"""Contract tests for the model definitions.

These do not need a trained checkpoint. They pin the input and output shapes
and confirm that a Stage 2 prediction can always be assembled into a valid FEN
placement field, which is the interface the rest of the pipeline relies on.

Skipped automatically when torch is not installed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from chessfengen.eval.metrics import NUM_SQUARES
from chessfengen.fen.placement import (
    class_indices_to_grid,
    grid_to_placement_field,
    placement_field_to_grid,
)
from chessfengen.models.stage1_detector import CornerDetector
from chessfengen.models.stage2_classifier import SquareClassifier
from chessfengen.pieces import NUM_CLASSES

BATCH: int = 2
IMAGE_SIZE: int = 128


def test_square_classifier_output_shape() -> None:
    model = SquareClassifier().eval()
    images = torch.rand(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (BATCH, NUM_SQUARES, NUM_CLASSES)


def test_square_classifier_accepts_varied_resolution() -> None:
    model = SquareClassifier().eval()
    with torch.no_grad():
        logits = model(torch.rand(1, 3, 200, 200))
    assert logits.shape == (1, NUM_SQUARES, NUM_CLASSES)


def test_square_prediction_assembles_to_valid_placement() -> None:
    model = SquareClassifier().eval()
    with torch.no_grad():
        logits = model(torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    indices = tuple(logits.argmax(dim=-1)[0].tolist())
    placement = grid_to_placement_field(class_indices_to_grid(indices))
    # A well formed placement field must round trip through the parser.
    assert placement_field_to_grid(placement) == class_indices_to_grid(indices)


def test_corner_detector_output_shape_and_range() -> None:
    model = CornerDetector().eval()
    with torch.no_grad():
        corners = model(torch.rand(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert corners.shape == (BATCH, 8)
    # The sigmoid head must keep every coordinate in [0, 1].
    assert bool((corners >= 0).all()) and bool((corners <= 1).all())
