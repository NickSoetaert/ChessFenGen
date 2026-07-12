"""Piece vocabulary and the 13 way class space used by the square classifier.

FEN piece placement encodes each piece as a single character. Uppercase is a
white piece, lowercase is a black piece. An empty square is represented inside
this codebase by the EMPTY sentinel (the empty string).
"""

from __future__ import annotations

# White and black piece characters, in the conventional FEN ordering.
WHITE_PIECES: tuple[str, ...] = ("P", "N", "B", "R", "Q", "K")
BLACK_PIECES: tuple[str, ...] = ("p", "n", "b", "r", "q", "k")
PIECE_CHARS: tuple[str, ...] = WHITE_PIECES + BLACK_PIECES

# Sentinel for an empty square inside an 8x8 grid.
EMPTY: str = ""

# Stable ordering of the 13 classes for the Stage 2 square classifier.
# Index 0 is the empty square, followed by the 12 piece types. This ordering is
# a training contract: changing it invalidates every previously trained model.
CLASS_LABELS: tuple[str, ...] = (EMPTY,) + PIECE_CHARS
CLASS_TO_INDEX: dict[str, int] = {label: index for index, label in enumerate(CLASS_LABELS)}
INDEX_TO_CLASS: tuple[str, ...] = CLASS_LABELS
NUM_CLASSES: int = len(CLASS_LABELS)


def is_piece_char(value: str) -> bool:
    """Return True when value is one of the 12 FEN piece characters."""
    return value in PIECE_CHARS
