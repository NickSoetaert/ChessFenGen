"""Stage 2: per square classifier over a dewarped canonical board.

The network maps a board image to an 8x8 grid of 13 way logits. An adaptive
pool to 8x8 makes the head independent of the exact input resolution, and a
1x1 convolution turns each grid cell into class logits. The output is reshaped
to (batch, 64, num_classes) so it lines up with the row major targets produced
by grid_to_class_indices.

Requires torch (see requirements.txt).
"""

from __future__ import annotations

import torch
from torch import nn

from chessfengen.pieces import NUM_CLASSES

BOARD_SIZE: int = 8


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SquareClassifier(nn.Module):
    """CNN that predicts a piece class for each of the 64 squares."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.backbone: nn.Sequential = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        self.pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((BOARD_SIZE, BOARD_SIZE))
        self.head: nn.Conv2d = nn.Conv2d(256, num_classes, kernel_size=1)
        self._num_classes: int = num_classes

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Map a (batch, 3, H, W) image to (batch, 64, num_classes) logits."""
        features: torch.Tensor = self.backbone(image)
        pooled: torch.Tensor = self.pool(features)
        logits: torch.Tensor = self.head(pooled)
        batch: int = logits.shape[0]
        logits = logits.reshape(batch, self._num_classes, BOARD_SIZE * BOARD_SIZE)
        return logits.permute(0, 2, 1).contiguous()
