"""Quality gate tests for trained checkpoints.

These are the "validate the model once trained" checks. Each test loads a
checkpoint, evaluates it on a freshly generated held out synthetic split, and
asserts the quality clears a threshold. They skip cleanly when torch or the
render dependencies are missing, or when the checkpoint file does not exist, so
the suite stays green before any training has happened.

Thresholds and checkpoint paths are environment overridable so the same tests
serve as a CI gate you tune as the model improves:

  CHESSFENGEN_STAGE2_CKPT      stage 2 checkpoint path       (default stage2.pt)
  CHESSFENGEN_STAGE1_CKPT      stage 1 checkpoint path       (default stage1.pt)
  CHESSFENGEN_MIN_SQUARE_ACC   min stage 2 square accuracy   (default 0.95)
  CHESSFENGEN_MIN_EXACT_MATCH  min stage 2 exact board rate  (default 0.30)
  CHESSFENGEN_MAX_CORNER_ERR   max stage 1 normalised error  (default 0.03)
  CHESSFENGEN_EVAL_SIZE        eval boards per test          (default 512)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cairosvg")
pytest.importorskip("chess")

from torch.utils.data import DataLoader

from chessfengen.data.dataset import Stage1CornerDataset, Stage2SquareDataset
from chessfengen.eval.evaluate import (
    EVAL_BASE_SEED,
    evaluate_stage1,
    evaluate_stage2,
)
from chessfengen.models.stage1_detector import CornerDetector
from chessfengen.models.stage2_classifier import SquareClassifier

IMAGE_SIZE: int = 256
DEVICE: str = "cpu"


def _eval_size() -> int:
    return int(os.environ.get("CHESSFENGEN_EVAL_SIZE", "512"))


def _checkpoint_or_skip(env_var: str, default: str) -> Path:
    path = Path(os.environ.get(env_var, default))
    if not path.is_file():
        pytest.skip(f"no checkpoint at {path}; train the model or set {env_var}")
    return path


def test_stage2_square_and_board_accuracy() -> None:
    checkpoint = _checkpoint_or_skip("CHESSFENGEN_STAGE2_CKPT", "stage2.pt")
    model = SquareClassifier()
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    dataset = Stage2SquareDataset(
        epoch_size=_eval_size(), image_size=IMAGE_SIZE, base_seed=EVAL_BASE_SEED
    )
    loader = DataLoader(dataset, batch_size=32, num_workers=0)
    report = evaluate_stage2(model.to(DEVICE), loader, DEVICE)

    min_square = float(os.environ.get("CHESSFENGEN_MIN_SQUARE_ACC", "0.95"))
    min_exact = float(os.environ.get("CHESSFENGEN_MIN_EXACT_MATCH", "0.30"))
    assert report.square_accuracy >= min_square, (
        f"square accuracy {report.square_accuracy:.4f} below gate {min_square}"
    )
    assert report.exact_match_rate >= min_exact, (
        f"exact board match {report.exact_match_rate:.4f} below gate {min_exact}"
    )


def test_stage2_no_class_collapses() -> None:
    """Every piece type must be recognised at least sometimes.

    A recall of zero for a class means the model never predicts it, a failure
    that a single overall accuracy number can hide when that class is rare.
    """
    checkpoint = _checkpoint_or_skip("CHESSFENGEN_STAGE2_CKPT", "stage2.pt")
    model = SquareClassifier()
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    dataset = Stage2SquareDataset(
        epoch_size=_eval_size(), image_size=IMAGE_SIZE, base_seed=EVAL_BASE_SEED
    )
    loader = DataLoader(dataset, batch_size=32, num_workers=0)
    report = evaluate_stage2(model.to(DEVICE), loader, DEVICE)

    collapsed = [label for label, recall in report.per_class_recall.items() if recall == 0.0]
    assert not collapsed, f"model never predicts these classes: {collapsed}"


def test_stage1_corner_localisation_error() -> None:
    checkpoint = _checkpoint_or_skip("CHESSFENGEN_STAGE1_CKPT", "stage1.pt")
    model = CornerDetector()
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    dataset = Stage1CornerDataset(
        epoch_size=_eval_size(), image_size=IMAGE_SIZE, base_seed=EVAL_BASE_SEED
    )
    loader = DataLoader(dataset, batch_size=32, num_workers=0)
    report = evaluate_stage1(model.to(DEVICE), loader, DEVICE, IMAGE_SIZE)

    max_error = float(os.environ.get("CHESSFENGEN_MAX_CORNER_ERR", "0.03"))
    assert report.mean_corner_error_normalized <= max_error, (
        f"corner error {report.mean_corner_error_normalized:.4f} above gate {max_error}"
    )
