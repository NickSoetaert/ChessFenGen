"""On the fly synthetic data for both training stages.

Images are generated per sample and never written to disk; only checkpoints
touch the HDD. This is what lets training run over effectively unlimited data:
generate a batch, learn from it, discard it, generate more.

Two access patterns share one sample generator:

  make_stage2_sample / make_stage1_sample
      the pure generators: given a seeded Random, produce one sample.
  Stage2SquareDataset / Stage1CornerDataset  (map style, finite, deterministic)
      a fixed reproducible set seeded by index. Used for a stable validation
      split whose boards do not change between checkpoints.
  Stage2SampleStream / Stage1SampleStream    (iterable, infinite)
      an endless stream of fresh, unique positions for training. Each worker
      seeds every sample from (base_seed, worker id, counter), so no board is
      ever repeated within a run.

Requires torch plus the render dependencies (see requirements.txt).
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from chessfengen.augment.domains import Domain, augment
from chessfengen.augment.noise import NoiseLevel
from chessfengen.fen.generator import generate_random_placement
from chessfengen.fen.placement import grid_to_class_indices, grid_to_placement_field
from chessfengen.render.board import random_theme, render_board
from chessfengen.render.piece_sets import default_piece_sets
from chessfengen.coretypes import Grid, RenderedBoard

# Domains a dewarped board can still exhibit. Full perspective warp is handled
# by Stage 1, so Stage 2 only sees residual screenshot and scan noise.
_STAGE2_DOMAINS: tuple[Domain, ...] = (Domain.SCREENSHOT, Domain.SCAN)
_ALL_DOMAINS: tuple[Domain, ...] = (Domain.SCREENSHOT, Domain.SCAN, Domain.PHOTO)
_ALL_LEVELS: tuple[NoiseLevel, ...] = (NoiseLevel.NONE, NoiseLevel.SOME, NoiseLevel.HEAVY)

# Deterministic map-dataset seed strides, distinct per stage so the two fixed
# validation sets never coincide.
_STAGE2_STRIDE: int = 1_000_003
_STAGE1_STRIDE: int = 2_000_003

SampleFn = Callable[[random.Random, int, tuple[str, ...]], tuple[torch.Tensor, torch.Tensor]]


def make_stage2_sample(
    rng: random.Random, image_size: int, piece_sets: tuple[str, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one Stage 2 sample: a canonical board and its 64 class targets."""
    grid: Grid = generate_random_placement(rng)
    placement: str = grid_to_placement_field(grid)
    rendered: RenderedBoard = render_board(
        placement=placement,
        piece_set=rng.choice(piece_sets),
        theme=random_theme(rng),
        size=image_size,
        rng=rng,
    )
    image_bgr: np.ndarray = _pil_to_bgr(rendered.image)
    domain: Domain = rng.choice(_STAGE2_DOMAINS)
    level: NoiseLevel = rng.choice(_ALL_LEVELS)
    augmented, _ = augment(image_bgr, rendered.corners, domain, level, rng)
    image_tensor: torch.Tensor = _bgr_to_tensor(augmented)
    targets: torch.Tensor = torch.tensor(grid_to_class_indices(grid), dtype=torch.long)
    return image_tensor, targets


def make_stage1_sample(
    rng: random.Random, image_size: int, piece_sets: tuple[str, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one Stage 1 sample: a scene image and normalised board corners."""
    grid: Grid = generate_random_placement(rng)
    placement: str = grid_to_placement_field(grid)
    rendered: RenderedBoard = render_board(
        placement=placement,
        piece_set=rng.choice(piece_sets),
        theme=random_theme(rng),
        size=image_size,
        rng=rng,
    )
    image_bgr: np.ndarray = _pil_to_bgr(rendered.image)
    domain: Domain = rng.choice(_ALL_DOMAINS)
    level: NoiseLevel = rng.choice(_ALL_LEVELS)
    augmented, corners = augment(image_bgr, rendered.corners, domain, level, rng)
    image_tensor: torch.Tensor = _bgr_to_tensor(augmented)
    corner_target: torch.Tensor = torch.tensor(
        corners.reshape(-1) / float(image_size), dtype=torch.float32
    )
    return image_tensor, corner_target


class _DeterministicDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A fixed, reproducible sample set seeded by index (for validation)."""

    def __init__(
        self,
        sample_fn: SampleFn,
        stride: int,
        epoch_size: int,
        image_size: int,
        piece_sets: tuple[str, ...] | None,
        base_seed: int,
    ) -> None:
        self._sample_fn: SampleFn = sample_fn
        self._stride: int = stride
        self._epoch_size: int = epoch_size
        self._image_size: int = image_size
        self._piece_sets: tuple[str, ...] = piece_sets or default_piece_sets()
        self._base_seed: int = base_seed

    def __len__(self) -> int:
        return self._epoch_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng: random.Random = random.Random(self._base_seed * self._stride + index)
        return self._sample_fn(rng, self._image_size, self._piece_sets)


class Stage2SquareDataset(_DeterministicDataset):
    """Fixed Stage 2 set: canonical board image plus 64 square class targets."""

    def __init__(
        self,
        epoch_size: int,
        image_size: int = 256,
        piece_sets: tuple[str, ...] | None = None,
        base_seed: int = 0,
    ) -> None:
        super().__init__(
            make_stage2_sample, _STAGE2_STRIDE, epoch_size, image_size, piece_sets, base_seed
        )


class Stage1CornerDataset(_DeterministicDataset):
    """Fixed Stage 1 set: scene image plus the four board corners in [0, 1].

    The board currently fills most of the frame. Compositing the board onto a
    full book page is the next step and is tracked in the README roadmap.
    """

    def __init__(
        self,
        epoch_size: int,
        image_size: int = 256,
        piece_sets: tuple[str, ...] | None = None,
        base_seed: int = 0,
    ) -> None:
        super().__init__(
            make_stage1_sample, _STAGE1_STRIDE, epoch_size, image_size, piece_sets, base_seed
        )


class _SampleStream(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """An endless stream of fresh samples, one unique seed per sample.

    Each DataLoader worker walks its own counter, and the seed mixes in the
    worker id, so N workers produce N disjoint non repeating streams. Seeds are
    drawn from a 63 bit space by a hash, so repeats across a run are negligible
    and are at worst a wasted sample, never a correctness issue.
    """

    def __init__(
        self,
        sample_fn: SampleFn,
        salt: int,
        image_size: int,
        piece_sets: tuple[str, ...] | None,
        base_seed: int,
    ) -> None:
        self._sample_fn: SampleFn = sample_fn
        self._salt: int = salt
        self._image_size: int = image_size
        self._piece_sets: tuple[str, ...] = piece_sets or default_piece_sets()
        self._base_seed: int = base_seed

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        worker_id: int = worker.id if worker is not None else 0
        counter: int = 0
        while True:
            seed: int = _stream_seed(self._base_seed, self._salt, worker_id, counter)
            yield self._sample_fn(random.Random(seed), self._image_size, self._piece_sets)
            counter += 1


class Stage2SampleStream(_SampleStream):
    """Infinite stream of fresh Stage 2 training samples."""

    def __init__(
        self,
        image_size: int = 256,
        piece_sets: tuple[str, ...] | None = None,
        base_seed: int = 0,
    ) -> None:
        super().__init__(make_stage2_sample, salt=2, image_size=image_size, piece_sets=piece_sets, base_seed=base_seed)


class Stage1SampleStream(_SampleStream):
    """Infinite stream of fresh Stage 1 training samples."""

    def __init__(
        self,
        image_size: int = 256,
        piece_sets: tuple[str, ...] | None = None,
        base_seed: int = 0,
    ) -> None:
        super().__init__(make_stage1_sample, salt=1, image_size=image_size, piece_sets=piece_sets, base_seed=base_seed)


def _stream_seed(base_seed: int, salt: int, worker_id: int, counter: int) -> int:
    """Mix an identifying tuple into a 63 bit seed (splitmix64 style)."""
    mask: int = 0xFFFFFFFFFFFFFFFF
    value: int = (
        base_seed * 0x9E3779B97F4A7C15
        + salt * 0xBF58476D1CE4E5B9
        + worker_id * 0x94D049BB133111EB
        + counter * 0xD6E8FEB86659FD93
    ) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value = value ^ (value >> 31)
    return value & 0x7FFFFFFFFFFFFFFF


def _pil_to_bgr(image) -> np.ndarray:
    rgb: np.ndarray = np.asarray(image.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def _bgr_to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    rgb: np.ndarray = image_bgr[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
