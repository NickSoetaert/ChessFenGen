"""Render a placement grid to a canonical, axis aligned board image.

The board fills the whole output image (no coordinate margin), so the four
board corners are exactly the image corners. Geometric augmentations downstream
move those corners; keeping the render itself margin free makes the Stage 1
corner labels trivially exact at generation time.

Asset sets render through a glyph cache: each of a set's 12 piece SVGs is
rasterized once at a given cell size and kept in memory as an RGBA array, and
each board is then assembled by alpha compositing those cached glyphs onto a
cached board background. This turns roughly eighteen SVG rasterizations per
board into a handful of numpy blends, which is about seventy times faster and
pixel identical to rasterizing every board from scratch. The caches are process
local, so every DataLoader worker builds its own once and then streams fast.

The builtin chess.svg set is still supported when named explicitly, but it
cannot use the glyph cache and is not part of the default mix.

This module requires python-chess, cairosvg and Pillow (see requirements.txt).
"""

from __future__ import annotations

import io
import random

import cairosvg
import chess
import chess.svg
import numpy as np
from PIL import Image

from chessfengen.fen.placement import placement_field_to_grid
from chessfengen.pieces import EMPTY, PIECE_CHARS
from chessfengen.render.piece_sets import (
    BUILTIN_SET,
    BOARD_THEMES,
    BoardTheme,
    asset_set_dir,
    piece_svg_filename,
)
from chessfengen.coretypes import Grid, RenderedBoard

BOARD_SIZE: int = 8

# Straight alpha RGBA glyphs keyed by (piece_set, cell): char -> (rgb, alpha).
_GlyphSet = dict[str, tuple[np.ndarray, np.ndarray]]
_GLYPH_CACHE: dict[tuple[str, int], _GlyphSet] = {}
# Board backgrounds keyed by (theme name, size): float32 HxWx3.
_BACKGROUND_CACHE: dict[tuple[str, int], np.ndarray] = {}


def render_board(
    placement: str,
    piece_set: str,
    theme: BoardTheme,
    size: int = 512,
    rng: random.Random | None = None,
) -> RenderedBoard:
    """Render one placement field to a RenderedBoard at size x size pixels."""
    if size % BOARD_SIZE != 0:
        raise ValueError(f"size {size} must be divisible by {BOARD_SIZE}")
    if piece_set == BUILTIN_SET:
        image: Image.Image = _render_builtin(placement, theme, size)
    else:
        grid: Grid = placement_field_to_grid(placement)
        image = _render_cached(grid, piece_set, theme, size)
    corners: np.ndarray = np.array(
        [[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32
    )
    return RenderedBoard(image=image, corners=corners, placement=placement)


def _render_cached(grid: Grid, piece_set: str, theme: BoardTheme, size: int) -> Image.Image:
    cell: int = size // BOARD_SIZE
    glyphs: _GlyphSet = _get_glyphs(piece_set, cell)
    board: np.ndarray = _get_background(theme, size).copy()
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            cell_value: str = grid[row][col]
            if cell_value == EMPTY:
                continue
            rgb, alpha = glyphs[cell_value]
            top: int = row * cell
            left: int = col * cell
            region: np.ndarray = board[top : top + cell, left : left + cell]
            board[top : top + cell, left : left + cell] = alpha * rgb + (1.0 - alpha) * region
    return Image.fromarray(np.clip(board, 0, 255).astype(np.uint8), mode="RGB")


def _get_glyphs(piece_set: str, cell: int) -> _GlyphSet:
    key: tuple[str, int] = (piece_set, cell)
    cached: _GlyphSet | None = _GLYPH_CACHE.get(key)
    if cached is not None:
        return cached
    directory = asset_set_dir(piece_set)
    glyphs: _GlyphSet = {}
    for piece_char in PIECE_CHARS:
        svg_path = directory / piece_svg_filename(piece_char)
        if not svg_path.is_file():
            raise FileNotFoundError(f"missing piece glyph: {svg_path}")
        png_bytes: bytes = cairosvg.svg2png(
            url=str(svg_path), output_width=cell, output_height=cell
        )
        rgba: np.ndarray = np.asarray(
            Image.open(io.BytesIO(png_bytes)).convert("RGBA"), dtype=np.float32
        )
        glyphs[piece_char] = (rgba[:, :, :3], rgba[:, :, 3:4] / 255.0)
    _GLYPH_CACHE[key] = glyphs
    return glyphs


def _get_background(theme: BoardTheme, size: int) -> np.ndarray:
    key: tuple[str, int] = (theme.name, size)
    cached: np.ndarray | None = _BACKGROUND_CACHE.get(key)
    if cached is not None:
        return cached
    cell: int = size // BOARD_SIZE
    light: np.ndarray = _hex_to_rgb(theme.light)
    dark: np.ndarray = _hex_to_rgb(theme.dark)
    background: np.ndarray = np.empty((size, size, 3), dtype=np.float32)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            colour: np.ndarray = light if (row + col) % 2 == 0 else dark
            background[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell] = colour
    _BACKGROUND_CACHE[key] = background
    return background


def _hex_to_rgb(value: str) -> np.ndarray:
    text: str = value.lstrip("#")
    return np.array([int(text[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def _render_builtin(placement: str, theme: BoardTheme, size: int) -> Image.Image:
    base_board: chess.BaseBoard = chess.BaseBoard.empty()
    base_board.set_board_fen(placement)
    svg: str = chess.svg.board(
        base_board,
        coordinates=False,
        size=size,
        colors={"square light": theme.light, "square dark": theme.dark},
    )
    png_bytes: bytes = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"), output_width=size, output_height=size
    )
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def random_theme(rng: random.Random) -> BoardTheme:
    """Pick a random board color theme."""
    return rng.choice(BOARD_THEMES)
