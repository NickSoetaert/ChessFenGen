"""Dependency free smoke demo: generate a random placement and print it.

Run as: python -m chessfengen.demo

This exercises the pure FEN core (generation, placement serialization, class
index round trip) without needing the render or training dependencies, so it is
the fastest way to confirm the package is wired up.
"""

from __future__ import annotations

import random

from chessfengen.fen.generator import generate_random_placement
from chessfengen.fen.placement import (
    class_indices_to_grid,
    grid_to_class_indices,
    grid_to_placement_field,
)
from chessfengen.pieces import EMPTY
from chessfengen.coretypes import Grid


def ascii_board(grid: Grid) -> str:
    """Render a grid as a simple ASCII board for terminal inspection."""
    lines: list[str] = []
    for row in grid:
        cells: list[str] = [(cell if cell != EMPTY else ".") for cell in row]
        lines.append(" ".join(cells))
    return "\n".join(lines)


def main() -> None:
    rng: random.Random = random.Random(7)
    grid: Grid = generate_random_placement(rng)
    placement: str = grid_to_placement_field(grid)

    # Confirm the class index round trip reconstructs the grid exactly.
    indices: tuple[int, ...] = grid_to_class_indices(grid)
    assert class_indices_to_grid(indices) == grid

    print("placement field:")
    print(placement)
    print()
    print("ascii board:")
    print(ascii_board(grid))


if __name__ == "__main__":
    main()
