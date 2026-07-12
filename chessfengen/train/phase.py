"""Full streaming training phase for either stage.

Training is step based, not epoch based, because the data is effectively
infinite: every step pulls a fresh batch of never seen positions from an
IterableDataset, learns from it, and discards it. Nothing but checkpoints
reaches disk, so a run can consume billions of images on a small HDD.

Validation uses a fixed, reproducible split (the deterministic map dataset with
a reserved seed) so the metric is comparable across checkpoints. Two files are
written under the checkpoint directory:

  stage{N}_last.pt   full training state (model, optimizer, step) for --resume
  stage{N}_best.pt   bare model state_dict at the best validation metric, ready
                     for chessfengen.eval.evaluate and the inference pipeline

Run as:
  python -m chessfengen.train.phase --stage 2 --total-steps 20000
  python -m chessfengen.train.stage2 --total-steps 20000   (equivalent)

Requires torch plus the render dependencies (see requirements.txt).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from chessfengen.data.dataset import (
    Stage1CornerDataset,
    Stage1SampleStream,
    Stage2SampleStream,
    Stage2SquareDataset,
)
from chessfengen.eval.evaluate import EVAL_BASE_SEED, evaluate_stage1, evaluate_stage2
from chessfengen.models.stage1_detector import CornerDetector
from chessfengen.models.stage2_classifier import SquareClassifier
from chessfengen.pieces import NUM_CLASSES


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for one streaming training phase."""

    stage: int
    total_steps: int
    batch_size: int
    image_size: int
    learning_rate: float
    num_workers: int
    device: str
    val_size: int
    val_base_seed: int
    eval_every: int
    archive_every: int
    log_every: int
    checkpoint_dir: Path
    seed: int
    resume: bool


def run_training_phase(config: TrainConfig) -> None:
    """Train one stage over an infinite fresh data stream."""
    if config.stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {config.stage}")
    device: str = config.device
    higher_is_better: bool = config.stage == 2

    model, train_loader, val_loader = _build(config)
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path: Path = config.checkpoint_dir / f"stage{config.stage}_last.pt"
    best_path: Path = config.checkpoint_dir / f"stage{config.stage}_best.pt"

    start_step: int = 0
    best_metric: float | None = None
    if config.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"--resume given but no checkpoint at {last_path}")
        start_step, best_metric = _load_resume(model, optimizer, last_path, device)
        print(f"resumed from {last_path} at step {start_step} (best_metric={best_metric})")

    model.train()
    step: int = start_step
    running_loss: float = 0.0
    images_since_log: int = 0
    window_start: float = time.time()

    for images, targets in train_loader:
        if step >= config.total_steps:
            break
        images = images.to(device)
        targets = targets.to(device)
        loss: torch.Tensor = _stage_loss(config.stage, model, images, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        running_loss += loss.item()
        images_since_log += images.shape[0]

        if step % config.log_every == 0:
            elapsed: float = time.time() - window_start
            rate: float = images_since_log / elapsed if elapsed > 0 else 0.0
            print(
                f"step {step}/{config.total_steps} "
                f"loss={running_loss / config.log_every:.4f} images/s={rate:.1f}"
            )
            running_loss = 0.0
            images_since_log = 0
            window_start = time.time()

        if step % config.eval_every == 0 or step >= config.total_steps:
            metric, summary = _evaluate(config.stage, model, val_loader, device, config.image_size)
            improved: bool = best_metric is None or (
                metric > best_metric if higher_is_better else metric < best_metric
            )
            print(f"  [eval @ step {step}] {summary}" + ("  *best*" if improved else ""))
            if improved:
                best_metric = metric
            _save_resume(model, optimizer, last_path, step, best_metric)
            if improved:
                torch.save(model.state_dict(), best_path)
            model.train()
            window_start = time.time()

        if config.archive_every > 0 and step % config.archive_every == 0:
            archive_path: Path = config.checkpoint_dir / f"stage{config.stage}_step{step}.pt"
            torch.save(model.state_dict(), archive_path)
            print(f"  archived {archive_path}")

    metric_name: str = "square_accuracy" if higher_is_better else "corner_error"
    print(f"done at step {step}. best {metric_name}={best_metric:.4f}. best checkpoint: {best_path}")


def _build(
    config: TrainConfig,
) -> tuple[torch.nn.Module, DataLoader, DataLoader]:
    if config.stage == 2:
        model: torch.nn.Module = SquareClassifier().to(config.device)
        train_stream = Stage2SampleStream(image_size=config.image_size, base_seed=config.seed)
        val_dataset = Stage2SquareDataset(
            epoch_size=config.val_size, image_size=config.image_size, base_seed=config.val_base_seed
        )
    else:
        model = CornerDetector().to(config.device)
        train_stream = Stage1SampleStream(image_size=config.image_size, base_seed=config.seed)
        val_dataset = Stage1CornerDataset(
            epoch_size=config.val_size, image_size=config.image_size, base_seed=config.val_base_seed
        )
    train_loader: DataLoader = DataLoader(
        train_stream,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.device == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    val_loader: DataLoader = DataLoader(
        val_dataset, batch_size=config.batch_size, num_workers=config.num_workers
    )
    return model, train_loader, val_loader


def _stage_loss(
    stage: int, model: torch.nn.Module, images: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    if stage == 2:
        logits: torch.Tensor = model(images)
        return functional.cross_entropy(logits.reshape(-1, NUM_CLASSES), targets.reshape(-1))
    predictions: torch.Tensor = model(images)
    return functional.mse_loss(predictions, targets)


def _evaluate(
    stage: int, model: torch.nn.Module, val_loader: DataLoader, device: str, image_size: int
) -> tuple[float, str]:
    if stage == 2:
        report = evaluate_stage2(model, val_loader, device)
        summary: str = (
            f"square_acc={report.square_accuracy:.4f} exact_match={report.exact_match_rate:.4f}"
        )
        return report.square_accuracy, summary
    stage1_report = evaluate_stage1(model, val_loader, device, image_size)
    summary = (
        f"corner_err_norm={stage1_report.mean_corner_error_normalized:.4f} "
        f"px={stage1_report.mean_corner_error_pixels:.2f}"
    )
    return stage1_report.mean_corner_error_normalized, summary


def _save_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
    step: int,
    best_metric: float | None,
) -> None:
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_metric": best_metric,
        },
        path,
    )


def _load_resume(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path, device: str
) -> tuple[int, float | None]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"]), checkpoint["best_metric"]


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--val-base-seed", type=int, default=EVAL_BASE_SEED)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument(
        "--archive-every",
        type=int,
        default=0,
        help="also save a step numbered checkpoint every N steps (0 disables)",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")


def _config_from_args(args: argparse.Namespace, stage: int) -> TrainConfig:
    return TrainConfig(
        stage=stage,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        device=args.device,
        val_size=args.val_size,
        val_base_seed=args.val_base_seed,
        eval_every=args.eval_every,
        archive_every=args.archive_every,
        log_every=args.log_every,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        resume=args.resume,
    )


def train_stage_cli(stage: int) -> None:
    """Entry point for the per stage wrappers (train.stage1 / train.stage2)."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    _add_common_args(parser)
    args: argparse.Namespace = parser.parse_args()
    run_training_phase(_config_from_args(args, stage))


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    _add_common_args(parser)
    args: argparse.Namespace = parser.parse_args()
    run_training_phase(_config_from_args(args, args.stage))


if __name__ == "__main__":
    main()
