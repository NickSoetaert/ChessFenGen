"""Conversions between an 8x8 grid, the FEN placement field, and class indices.

Only the piece placement field of a FEN is handled here. The other five FEN
fields (side to move, castling, en passant, halfmove clock, fullmove number)
are not recoverable from a static board image and are intentionally out of
scope for this project.
"""

from __future__ import annotations

from chessfengen.pieces import EMPTY, INDEX_TO_CLASS, PIECE_CHARS, CLASS_TO_INDEX
from chessfengen.coretypes import Grid

BOARD_SIZE: int = 8


def grid_to_placement_field(grid: Grid) -> str:
    """Serialize an 8x8 grid to the FEN piece placement field."""
    if len(grid) != BOARD_SIZE:
        raise ValueError(f"grid must have {BOARD_SIZE} rows, got {len(grid)}")
    rows: list[str] = []
    for row in grid:
        if len(row) != BOARD_SIZE:
            raise ValueError(f"grid row must have {BOARD_SIZE} cells, got {len(row)}")
        chars: list[str] = []
        empty_run: int = 0
        for cell in row:
            if cell == EMPTY:
                empty_run += 1
                continue
            if cell not in PIECE_CHARS:
                raise ValueError(f"invalid cell value: {cell!r}")
            if empty_run:
                chars.append(str(empty_run))
                empty_run = 0
            chars.append(cell)
        if empty_run:
            chars.append(str(empty_run))
        rows.append("".join(chars))
    return "/".join(rows)


def placement_field_to_grid(field: str) -> Grid:
    """Parse a FEN piece placement field into an 8x8 grid."""
    row_specs: list[str] = field.split("/")
    if len(row_specs) != BOARD_SIZE:
        raise ValueError(f"placement must have {BOARD_SIZE} rows, got {len(row_specs)}")
    grid: list[tuple[str, ...]] = []
    for row_spec in row_specs:
        row: list[str] = []
        for char in row_spec:
            if char.isdigit():
                run: int = int(char)
                if run < 1 or run > BOARD_SIZE:
                    raise ValueError(f"invalid empty run {run} in row {row_spec!r}")
                row.extend([EMPTY] * run)
            elif char in PIECE_CHARS:
                row.append(char)
            else:
                raise ValueError(f"invalid placement character: {char!r}")
        if len(row) != BOARD_SIZE:
            raise ValueError(f"row {row_spec!r} expands to {len(row)} cells, expected {BOARD_SIZE}")
        grid.append(tuple(row))
    return tuple(grid)


def grid_to_class_indices(grid: Grid) -> tuple[int, ...]:
    """Flatten a grid to 64 class indices in row-major order (rank 8 first)."""
    indices: list[int] = []
    for row in grid:
        for cell in row:
            indices.append(CLASS_TO_INDEX[cell])
    return tuple(indices)


def class_indices_to_grid(indices: tuple[int, ...]) -> Grid:
    """Rebuild an 8x8 grid from 64 row-major class indices."""
    expected: int = BOARD_SIZE * BOARD_SIZE
    if len(indices) != expected:
        raise ValueError(f"expected {expected} indices, got {len(indices)}")
    grid: list[tuple[str, ...]] = []
    for row_start in range(0, expected, BOARD_SIZE):
        row_indices: tuple[int, ...] = indices[row_start : row_start + BOARD_SIZE]
        grid.append(tuple(INDEX_TO_CLASS[index] for index in row_indices))
    return tuple(grid)
