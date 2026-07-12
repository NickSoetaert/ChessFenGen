"""Tests for the data streams and the deterministic validation datasets.

The core contract: the training stream yields fresh, non repeating samples,
while the map dataset is byte for byte reproducible so a validation split is
stable across checkpoints. Skipped when torch or the render deps are missing.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cairosvg")
pytest.importorskip("chess")

from itertools import islice

from chessfengen.data.dataset import (
    Stage2SampleStream,
    Stage2SquareDataset,
    _stream_seed,
)

IMAGE_SIZE: int = 64


def test_stream_samples_are_fresh() -> None:
    stream = Stage2SampleStream(image_size=IMAGE_SIZE, base_seed=0)
    batch = [image for image, _ in islice(iter(stream), 5)]
    # No two consecutive stream samples should be identical images.
    for i in range(len(batch)):
        for j in range(i + 1, len(batch)):
            assert not torch.equal(batch[i], batch[j]), f"stream repeated samples {i} and {j}"


def test_map_dataset_is_reproducible() -> None:
    a = Stage2SquareDataset(epoch_size=4, image_size=IMAGE_SIZE, base_seed=1)
    b = Stage2SquareDataset(epoch_size=4, image_size=IMAGE_SIZE, base_seed=1)
    image_a, target_a = a[2]
    image_b, target_b = b[2]
    assert torch.equal(image_a, image_b)
    assert torch.equal(target_a, target_b)


def test_map_dataset_distinct_indices_differ() -> None:
    dataset = Stage2SquareDataset(epoch_size=8, image_size=IMAGE_SIZE, base_seed=1)
    assert not torch.equal(dataset[0][0], dataset[1][0])


def test_stream_seed_distinct_per_worker_and_counter() -> None:
    seeds = {
        _stream_seed(base_seed=0, salt=2, worker_id=worker, counter=counter)
        for worker in range(4)
        for counter in range(1000)
    }
    # 4 workers * 1000 counters should give 4000 distinct seeds (no collisions).
    assert len(seeds) == 4000
    assert all(0 <= seed < (1 << 63) for seed in seeds)
