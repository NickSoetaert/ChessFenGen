"""Board colour themes and the piece set registry.

Two rendering paths are supported:

  BUILTIN_SET   rendered directly by python-chess (its embedded cburnett set).
                Always available, no assets required.
  asset sets    a directory of per-piece SVG files, composited onto a drawn
                board. Add open licensed sets (merida, alpha, pirouetti, ...)
                by dropping their SVGs under assets/piece_sets/<name>/.

Piece SVG files follow the lila naming convention: colour letter (w or b) plus
the uppercase piece letter, for example wP.svg, bN.svg.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BUILTIN_SET: str = "cburnett-builtin"

# Root under which asset piece sets live, one subdirectory per set.
ASSETS_ROOT: Path = Path(__file__).resolve().parent.parent.parent / "assets" / "piece_sets"


@dataclass(frozen=True)
class BoardTheme:
    """Light and dark square colours as CSS hex strings."""

    name: str
    light: str
    dark: str


BOARD_THEMES: tuple[BoardTheme, ...] = (
    BoardTheme("brown", "#f0d9b5", "#b58863"),
    BoardTheme("blue", "#dee3e6", "#8ca2ad"),
    BoardTheme("green", "#ffffdd", "#86a666"),
    BoardTheme("gray", "#dcdcdc", "#8f8f8f"),
    BoardTheme("newspaper", "#ffffff", "#c8c8c8"),
)


def piece_svg_filename(piece_char: str) -> str:
    """Map a FEN piece character to its lila style SVG filename."""
    colour: str = "w" if piece_char.isupper() else "b"
    return f"{colour}{piece_char.upper()}.svg"


def available_asset_sets() -> tuple[str, ...]:
    """Return the names of asset piece sets found on disk."""
    if not ASSETS_ROOT.is_dir():
        return ()
    return tuple(sorted(child.name for child in ASSETS_ROOT.iterdir() if child.is_dir()))


def default_piece_sets() -> tuple[str, ...]:
    """Return every asset set found on disk.

    Only asset sets are used by default because they render through the fast
    cached compositing path (see chessfengen.render.board). The builtin
    chess.svg set is still supported when requested explicitly, but it cannot
    use the glyph cache and is left out of the default mix. Raises when no asset
    sets are present rather than silently falling back to the slow builtin set.
    """
    sets: tuple[str, ...] = available_asset_sets()
    if not sets:
        raise RuntimeError(
            f"no piece sets found under {ASSETS_ROOT}. Add open licensed sets "
            "(see assets/piece_sets/README.md) before generating data."
        )
    return sets


def asset_set_dir(name: str) -> Path:
    """Return the directory for an asset piece set, or raise if missing.

    Fails loudly rather than falling back to the builtin set so a typo or a
    missing download surfaces immediately instead of silently degrading data.
    """
    directory: Path = ASSETS_ROOT / name
    if not directory.is_dir():
        raise FileNotFoundError(
            f"piece set {name!r} not found under {ASSETS_ROOT}. "
            f"Available asset sets: {available_asset_sets()}"
        )
    return directory
