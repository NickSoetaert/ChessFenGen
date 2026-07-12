"""End to end image to FEN inference: locate, dewarp, classify, assemble.

The flow mirrors the two stage architecture:

  image -> Stage 1 corners -> perspective dewarp -> Stage 2 squares -> placement

Corner detection runs on a fixed size copy of the input, but the returned
corners are normalised, so the dewarp samples the original full resolution
image for the sharpest board. The classifier output is turned into a FEN
placement field plus a per square confidence, which flags squares the model is
unsure about.

Run as:
  python -m chessfengen.inference.pipeline --image board.png \
      --stage1 stage1.pt --stage2 stage2.pt

Requires torch plus Pillow, numpy, and opencv (see requirements.txt).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from chessfengen.fen.placement import class_indices_to_grid, grid_to_placement_field
from chessfengen.inference.geometry import denormalize_corners, dewarp_board
from chessfengen.models.stage1_detector import CornerDetector
from chessfengen.models.stage2_classifier import SquareClassifier

DEFAULT_STAGE1_INPUT_SIZE: int = 256
DEFAULT_BOARD_SIZE: int = 256


@dataclass(frozen=True)
class BoardPrediction:
    """The result of recognising one board image.

    placement is the FEN piece placement field. corners are the detected board
    corners in input image pixels, shape (4, 2). square_confidence is the
    softmax probability of the chosen class for each of the 64 squares in row
    major order, shape (64,). dewarped is the canonical board image (RGB uint8)
    that Stage 2 actually classified, useful for debugging.
    """

    placement: str
    corners: np.ndarray
    square_confidence: np.ndarray
    dewarped: np.ndarray


class BoardRecognizer:
    """Runs the corner detector and square classifier as one pipeline."""

    def __init__(
        self,
        corner_detector: CornerDetector,
        square_classifier: SquareClassifier,
        device: str = "cpu",
        stage1_input_size: int = DEFAULT_STAGE1_INPUT_SIZE,
        board_size: int = DEFAULT_BOARD_SIZE,
    ) -> None:
        self._corner_detector: CornerDetector = corner_detector.to(device).eval()
        self._square_classifier: SquareClassifier = square_classifier.to(device).eval()
        self._device: str = device
        self._stage1_input_size: int = stage1_input_size
        self._board_size: int = board_size

    @classmethod
    def from_checkpoints(
        cls,
        stage1_checkpoint: Path,
        stage2_checkpoint: Path,
        device: str = "cpu",
        stage1_input_size: int = DEFAULT_STAGE1_INPUT_SIZE,
        board_size: int = DEFAULT_BOARD_SIZE,
    ) -> "BoardRecognizer":
        """Build a recognizer from two saved state_dict checkpoints."""
        detector: CornerDetector = CornerDetector()
        detector.load_state_dict(torch.load(stage1_checkpoint, map_location=device))
        classifier: SquareClassifier = SquareClassifier()
        classifier.load_state_dict(torch.load(stage2_checkpoint, map_location=device))
        return cls(detector, classifier, device, stage1_input_size, board_size)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> BoardPrediction:
        """Recognize the board in a PIL image and return its FEN placement."""
        rgb: np.ndarray = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]

        detect_input: Image.Image = image.convert("RGB").resize(
            (self._stage1_input_size, self._stage1_input_size)
        )
        detect_tensor: torch.Tensor = _image_to_tensor(np.asarray(detect_input)).to(self._device)
        normalized_corners: np.ndarray = (
            self._corner_detector(detect_tensor.unsqueeze(0))[0].cpu().numpy()
        )
        corners_px: np.ndarray = denormalize_corners(normalized_corners, width, height)

        dewarped: np.ndarray = dewarp_board(rgb, corners_px, self._board_size)
        classify_tensor: torch.Tensor = _image_to_tensor(dewarped).to(self._device)
        logits: torch.Tensor = self._square_classifier(classify_tensor.unsqueeze(0))[0].cpu()

        probabilities: torch.Tensor = functional.softmax(logits, dim=-1)
        confidence, indices = probabilities.max(dim=-1)
        placement: str = grid_to_placement_field(class_indices_to_grid(tuple(indices.tolist())))
        return BoardPrediction(
            placement=placement,
            corners=corners_px,
            square_confidence=confidence.numpy(),
            dewarped=dewarped,
        )

    def predict_placement(self, image: Image.Image) -> str:
        """Convenience wrapper returning only the FEN placement field."""
        return self.predict(image).placement


def load_image(path: Path) -> Image.Image:
    """Load an image file as a PIL image."""
    return Image.open(path)


def _image_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """Convert an HxWx3 RGB uint8 array to a CHW float tensor in [0, 1]."""
    scaled: np.ndarray = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(scaled.transpose(2, 0, 1)))


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--board-size", type=int, default=DEFAULT_BOARD_SIZE)
    parser.add_argument("--stage1-size", type=int, default=DEFAULT_STAGE1_INPUT_SIZE)
    parser.add_argument(
        "--low-confidence",
        type=float,
        default=0.5,
        help="report squares whose confidence falls below this value",
    )
    args: argparse.Namespace = parser.parse_args()

    recognizer: BoardRecognizer = BoardRecognizer.from_checkpoints(
        args.stage1, args.stage2, args.device, args.stage1_size, args.board_size
    )
    prediction: BoardPrediction = recognizer.predict(load_image(args.image))
    print(prediction.placement)

    low: list[int] = [
        index
        for index, value in enumerate(prediction.square_confidence)
        if value < args.low_confidence
    ]
    if low:
        print(f"low confidence squares (row major index): {low}")


if __name__ == "__main__":
    main()
