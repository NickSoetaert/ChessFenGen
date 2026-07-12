"""Evaluate a trained checkpoint over a held out synthetic split.

The evaluation split reuses the on the fly datasets with a non zero base_seed.
Because each sample is seeded by base_seed * prime + index, an eval base_seed
of 1 produces boards disjoint from training (base_seed 0) as long as the epoch
size stays below the seed stride. Point --real-dir at labelled real book scans
once available to grade the true target domain.

Run as:
  python -m chessfengen.eval.evaluate --stage 2 --checkpoint stage2.pt
  python -m chessfengen.eval.evaluate --stage 1 --checkpoint stage1.pt

Requires torch plus the render dependencies (see requirements.txt).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chessfengen.data.dataset import Stage1CornerDataset, Stage2SquareDataset
from chessfengen.eval.metrics import ConfusionMatrix, board_exact_match, corner_l2_distances
from chessfengen.models.stage1_detector import CornerDetector
from chessfengen.models.stage2_classifier import SquareClassifier
from chessfengen.pieces import NUM_CLASSES

EVAL_BASE_SEED: int = 1


@dataclass(frozen=True)
class Stage2Report:
    """Square classifier quality over an evaluation split."""

    board_count: int
    square_accuracy: float
    exact_match_rate: float
    per_class_recall: dict[str, float]


@dataclass(frozen=True)
class Stage1Report:
    """Corner detector localisation error over an evaluation split."""

    board_count: int
    mean_corner_error_normalized: float
    mean_corner_error_pixels: float


@torch.no_grad()
def evaluate_stage2(
    model: SquareClassifier, loader: DataLoader, device: str
) -> Stage2Report:
    """Grade a square classifier: per square accuracy and full board match."""
    model.eval()
    confusion: ConfusionMatrix = ConfusionMatrix(NUM_CLASSES)
    boards: int = 0
    exact: int = 0
    for images, targets in loader:
        logits: torch.Tensor = model(images.to(device))
        predictions: torch.Tensor = logits.argmax(dim=-1).cpu()
        for row in range(predictions.shape[0]):
            predicted: list[int] = predictions[row].tolist()
            actual: list[int] = targets[row].tolist()
            confusion.update(predicted, actual)
            if board_exact_match(predicted, actual):
                exact += 1
            boards += 1
    if boards == 0:
        raise ValueError("evaluation loader produced no boards")
    return Stage2Report(
        board_count=boards,
        square_accuracy=confusion.overall_accuracy(),
        exact_match_rate=exact / boards,
        per_class_recall=confusion.per_class_recall(),
    )


@torch.no_grad()
def evaluate_stage1(
    model: CornerDetector, loader: DataLoader, device: str, image_size: int
) -> Stage1Report:
    """Grade a corner detector: mean per corner localisation error."""
    model.eval()
    error_sum: float = 0.0
    corner_count: int = 0
    boards: int = 0
    for images, targets in loader:
        predictions: torch.Tensor = model(images.to(device)).cpu()
        distances = corner_l2_distances(predictions.numpy(), targets.numpy())
        error_sum += float(distances.sum())
        corner_count += int(distances.size)
        boards += int(predictions.shape[0])
    if corner_count == 0:
        raise ValueError("evaluation loader produced no boards")
    mean_normalized: float = error_sum / corner_count
    return Stage1Report(
        board_count=boards,
        mean_corner_error_normalized=mean_normalized,
        mean_corner_error_pixels=mean_normalized * image_size,
    )


def _load_state(model: torch.nn.Module, checkpoint: Path, device: str) -> torch.nn.Module:
    state: dict[str, torch.Tensor] = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    return model.to(device)


def _run_stage2(args: argparse.Namespace) -> None:
    dataset: Stage2SquareDataset = Stage2SquareDataset(
        epoch_size=args.epoch_size, image_size=args.image_size, base_seed=EVAL_BASE_SEED
    )
    loader: DataLoader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    model: SquareClassifier = _load_state(SquareClassifier(), args.checkpoint, args.device)
    report: Stage2Report = evaluate_stage2(model, loader, args.device)
    print(f"boards={report.board_count}")
    print(f"square_accuracy={report.square_accuracy:.4f}")
    print(f"exact_match_rate={report.exact_match_rate:.4f}")
    print("per_class_recall:")
    for label, recall in report.per_class_recall.items():
        print(f"  {label:6} {recall:.4f}")


def _run_stage1(args: argparse.Namespace) -> None:
    dataset: Stage1CornerDataset = Stage1CornerDataset(
        epoch_size=args.epoch_size, image_size=args.image_size, base_seed=EVAL_BASE_SEED
    )
    loader: DataLoader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    model: CornerDetector = _load_state(CornerDetector(), args.checkpoint, args.device)
    report: Stage1Report = evaluate_stage1(model, loader, args.device, args.image_size)
    print(f"boards={report.board_count}")
    print(f"mean_corner_error_normalized={report.mean_corner_error_normalized:.4f}")
    print(f"mean_corner_error_pixels={report.mean_corner_error_pixels:.2f}")


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epoch-size", type=int, default=4000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args: argparse.Namespace = parser.parse_args()
    if args.stage == 2:
        _run_stage2(args)
    else:
        _run_stage1(args)


if __name__ == "__main__":
    main()
