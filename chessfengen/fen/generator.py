"""Fast random piece placement generation for synthetic training data.

Positions are deliberately not required to be legal or reachable. Random
placement gives near uniform coverage of every piece type on every square,
which is exactly what the Stage 2 square classifier wants to learn from. The
generator only produces the piece placement grid, never a full FEN, because
the other FEN fields are not visible in a board image.
"""

from __future__ import annotations

import random

from chessfengen.pieces import EMPTY, PIECE_CHARS
from chessfengen.coretypes import Grid

BOARD_SIZE: int = 8
NUM_SQUARES: int = BOARD_SIZE * BOARD_SIZE


def generate_random_placement(
    rng: random.Random,
    min_pieces: int = 2,
    max_pieces: int = 32,
    force_two_kings: bool = True,
) -> Grid:
    """Generate one random 8x8 placement grid.

    rng is passed in explicitly so datasets can seed reproducible streams.
    When force_two_kings is set, exactly one white and one black king are
    placed first, then the remaining squares are filled with uniformly random
    piece characters. min_pieces and max_pieces bound the total piece count.
    """
    if not 0 <= min_pieces <= max_pieces <= NUM_SQUARES:
        raise ValueError(
            f"require 0 <= min_pieces <= max_pieces <= {NUM_SQUARES}, "
            f"got min_pieces={min_pieces}, max_pieces={max_pieces}"
        )
    if force_two_kings and min_pieces < 2:
        raise ValueError("force_two_kings requires min_pieces >= 2")

    piece_count: int = rng.randint(min_pieces, max_pieces)
    squares: list[int] = list(range(NUM_SQUARES))
    rng.shuffle(squares)

    board: list[str] = [EMPTY] * NUM_SQUARES
    cursor: int = 0
    if force_two_kings:
        board[squares[cursor]] = "K"
        cursor += 1
        board[squares[cursor]] = "k"
        cursor += 1
    while cursor < piece_count:
        board[squares[cursor]] = rng.choice(PIECE_CHARS)
        cursor += 1

    return tuple(
        tuple(board[row * BOARD_SIZE + col] for col in range(BOARD_SIZE))
        for row in range(BOARD_SIZE)
    )
