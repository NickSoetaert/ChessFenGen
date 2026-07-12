"""Stage 1: board corner detector.

The network regresses the four board corner coordinates (top-left, top-right,
bottom-right, bottom-left) normalized to [0, 1] of the image dimensions. Those
corners drive a perspective transform that dewarps the board into the canonical
image consumed by Stage 2.

Requires torch (see requirements.txt).
"""

from __future__ import annotations

import torch
from torch import nn

NUM_CORNER_COORDS: int = 8


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class CornerDetector(nn.Module):
    """CNN that regresses four normalised board corner coordinates."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone: nn.Sequential = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        self.pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.head: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, NUM_CORNER_COORDS),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Map a (batch, 3, H, W) image to (batch, 8) corner coordinates."""
        features: torch.Tensor = self.backbone(image)
        pooled: torch.Tensor = self.pool(features)
        return self.head(pooled)
