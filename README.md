# ChessFenGen

Convert a 2D image of a chess board into a FEN piece placement string.

**Target use case:** a user uploads a PDF of a chess book, and the model reads
any shown position into FEN, so a diagram can be dropped straight into an
engine or database.

## What this model predicts (and what it cannot)

A full FEN has six fields:

```
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
|__ piece placement __|                     |  |     | |  |
                                      side _|  |     | |  |_ fullmove
                                  castling ____|     | |____ halfmove
                                          en passant_|
```

Only the **piece placement field** is visible in a picture of a board. Side to
move, castling rights, en passant, and the move clocks are not, so this project
predicts the placement field only and never tries to hallucinate the rest.

## Architecture: two stages

Real inputs are a mix of clean screenshots, photocopied scans, and photos of a
page (with perspective warp and glare). Rather than force one network to handle
all of that, the hard geometric part is split out:

```
  input image
      |
      v
+-----------------+     four board corners
|  Stage 1        |------------------------+
|  CornerDetector |                        |
+-----------------+                        v
                                   perspective dewarp
                                           |
                                           v
                                 canonical 8x8 board image
                                           |
                                           v
                                  +--------------------+
                                  |  Stage 2           |   64 x 13 logits
                                  |  SquareClassifier  |------------------+
                                  +--------------------+                  |
                                                                          v
                                                            assemble placement field
                                                                          |
                                                                          v
                                                              rnbqkbnr/pppppppp/...
```

- **Stage 1 (`models/stage1_detector.py`)** regresses the four board corners.
  A perspective transform then dewarps the board into a canonical, axis aligned
  image. This absorbs all the "photo" difficulty, so a warped phone photo and a
  clean screenshot look the same to Stage 2.
- **Stage 2 (`models/stage2_classifier.py`)** classifies each of the 64 squares
  into one of 13 classes (empty plus the 12 piece types). The predictions are
  assembled into the placement field.

This decomposition matches the input mix (screenshots mostly skip Stage 1;
photos lean on it), gives near uniform per class coverage for the classifier,
and is interpretable: a wrong result points at a specific square and a specific
stage.

## Synthetic data pipeline

Training data is generated on the fly and never persisted. Each sample:

1. **Generate a random placement** (`fen/generator.py`). Positions need not be
   legal; random placement gives excellent per class coverage. Only the piece
   grid is produced, never a full FEN.
2. **Render to a canonical board** (`render/board.py`). Uses python-chess plus
   colour themes and pluggable piece sets. The board fills the image, so the
   four corner labels are exact by construction.
3. **Augment into a domain** (`augment/domains.py`) at one of three noise levels
   (`augment/noise.py`): `SCREENSHOT`, `SCAN`, or `PHOTO`. Geometric transforms
   also move the corner labels with the same matrix, keeping Stage 1 targets
   exact.
4. **Feed to training** (`data/dataset.py`). Stage 2 sees residual screenshot
   and scan noise on a canonical board; Stage 1 sees the full warped scene and
   the corner targets.

Training draws from an infinite `IterableDataset` stream: every step generates
a fresh batch of never seen positions, learns from it, and discards it, so a
run can consume billions of images while only checkpoints touch the HDD (see
"Training" below). Validation instead uses a fixed, reproducible map dataset
with a reserved seed, so the metric is comparable across checkpoints.

A frozen evaluation split of **real** book scans and photos is intentionally
not generated here. Synthetic only training hides its domain gap unless the
eval set is real; that split is the early warning system.

## Package layout

```
chessfengen/
  pieces.py              piece vocabulary and the 13 class space
  coretypes.py           Grid alias and RenderedBoard container
  demo.py                dependency free smoke demo
  fen/
    generator.py         random placement generation
    placement.py         grid <-> placement field <-> class indices
  render/
    piece_sets.py        board themes and piece set registry
    board.py             placement -> canonical board image + corners
  augment/
    noise.py             NoiseLevel: none / some / heavy
    domains.py           screenshot / scan / photo augmentation
  data/
    dataset.py           infinite training streams and fixed validation sets
  models/
    stage1_detector.py   CornerDetector (corner regression)
    stage2_classifier.py SquareClassifier (64 square classification)
  train/
    phase.py             streaming step based training phase (both stages)
    stage1.py            Stage 1 training entry point (wraps phase)
    stage2.py            Stage 2 training entry point (wraps phase)
  eval/
    metrics.py           framework free accuracy and corner metrics
    evaluate.py          grade a checkpoint over a held out split
  inference/
    geometry.py          perspective dewarp (cv2, framework free)
    pipeline.py          BoardRecognizer: image to FEN placement
assets/piece_sets/       13 FOSS lichess piece sets (see ATTRIBUTION.md)
tests/                   core, generator, augmentation, metric, and model tests
```

## Getting started

The pure FEN core (generation, placement, class indices) needs only the
standard library:

```
python -m chessfengen.demo
```

Everything else needs the pipeline dependencies:

```
pip install -r requirements.txt
```

## Training

Training is step based over an infinite stream of freshly generated positions
(`chessfengen.train.phase`). Each step learns from a new batch and throws it
away, so the only thing written to disk is checkpoints.

```
python -m chessfengen.train.stage2 --total-steps 20000 --checkpoint-dir checkpoints
python -m chessfengen.train.stage1 --total-steps 20000 --checkpoint-dir checkpoints
```

Every `--eval-every` steps the model is scored on a fixed validation split and
two checkpoints are written under `--checkpoint-dir`:

- `stage{N}_best.pt` the bare model weights at the best validation metric, ready
  for `chessfengen.eval.evaluate` and the inference pipeline.
- `stage{N}_last.pt` the full training state (model, optimizer, step) for
  `--resume`, which continues the run from where it stopped.

Useful flags: `--total-steps`, `--batch-size`, `--image-size`, `--learning-rate`,
`--num-workers` (data generation is CPU bound, so more workers means more
images per second), `--eval-every`, `--val-size`, `--seed`, `--resume`. The
equivalent single command is `python -m chessfengen.train.phase --stage 2 ...`.

## Tests

Run the tests:

```
pytest
```

## End to end inference

`BoardRecognizer` ties the two stages together: Stage 1 detects the board
corners, a perspective transform dewarps the board to a canonical image, Stage 2
classifies the 64 squares, and the result is assembled into a FEN placement
field. Corner detection runs on a resized copy but the corners are normalised,
so the dewarp samples the original full resolution image.

```
python -m chessfengen.inference.pipeline \
    --image board.png --stage1 stage1.pt --stage2 stage2.pt
```

Or from Python:

```python
from pathlib import Path
from PIL import Image
from chessfengen.inference.pipeline import BoardRecognizer

recognizer = BoardRecognizer.from_checkpoints(Path("stage1.pt"), Path("stage2.pt"))
prediction = recognizer.predict(Image.open("board.png"))
print(prediction.placement)          # FEN piece placement field
print(prediction.square_confidence)  # per square softmax confidence, shape (64,)
```

`predict` returns a `BoardPrediction` with the placement string, detected
corners, a per square confidence (to flag uncertain squares), and the dewarped
board image for debugging.

## Validating a trained model

Grade a checkpoint over a fresh held out synthetic split (seed disjoint from
training):

```
python -m chessfengen.eval.evaluate --stage 2 --checkpoint stage2.pt
python -m chessfengen.eval.evaluate --stage 1 --checkpoint stage1.pt
```

Stage 2 reports per square accuracy, full board exact match rate, and per class
recall; Stage 1 reports mean corner localisation error (normalised and in
pixels). The same harness backs the quality gate tests in
`tests/test_trained_model.py`, which load a checkpoint and assert the metrics
clear a threshold. Those tests skip when no checkpoint exists, so the suite
stays green before training. Once a model is trained, point them at it and tune
the gates via environment variables:

```
CHESSFENGEN_STAGE2_CKPT      stage 2 checkpoint path       (default stage2.pt)
CHESSFENGEN_STAGE1_CKPT      stage 1 checkpoint path       (default stage1.pt)
CHESSFENGEN_MIN_SQUARE_ACC   min stage 2 square accuracy   (default 0.95)
CHESSFENGEN_MIN_EXACT_MATCH  min stage 2 exact board rate  (default 0.30)
CHESSFENGEN_MAX_CORNER_ERR   max stage 1 normalised error  (default 0.03)
CHESSFENGEN_EVAL_SIZE        eval boards per test          (default 512)
```

The test layers are: pure metric tests and model contract tests (output shapes,
that a prediction always assembles into a valid FEN placement) run with no
training; the quality gates above run once a checkpoint exists. Note that exact
board match compounds per square error over 64 squares, so even a strong 99%
square accuracy yields only about a 53% exact board rate; gate the two
independently.
