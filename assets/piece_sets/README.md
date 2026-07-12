# Piece set assets

Open licensed piece sets, one directory per set. Filenames follow the lila
convention: colour letter (`w` or `b`) plus the uppercase piece letter, for
example `wP.svg`, `bN.svg`. The renderer (`chessfengen/render/board.py`)
composites these onto a drawn board, and the registry
(`chessfengen/render/piece_sets.py`) discovers every subdirectory here.

See `ATTRIBUTION.md` for the author and license of each set. Those files keep
their original licenses and are not covered by the repository MIT license.

## Currently included (13 FOSS sets from lichess lila)

```
cburnett  merida    chessnut  letter    pirouetti  pixel   mpchess
fantasy   spatial   celtic    shapes    rhosgfx    kiwen-suwi
```

Two of these are stylised rather than standard figurines: `letter` uses piece
letters and `shapes` uses abstract symbols. They add robustness but do not
resemble printed book diagrams, so exclude them per training run when needed by
passing an explicit `piece_sets` tuple to the datasets.

## Adding more sets

Drop a new directory here with the 12 SVG files named as above. Only add sets
under free / open licenses, and record the author and license in
`ATTRIBUTION.md`. Do not add proprietary or non-commercial sets (for example
chess.com sets, or lila's `CC BY-NC-SA` and non-free sets).
