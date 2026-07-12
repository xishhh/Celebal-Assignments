"""Simple CNN for land-use classification (no pretrained weights)."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn, Tensor


class ConvBlock(nn.Module):
    """One convolutional block: Conv2d → BatchNorm → ReLU → MaxPool.

    Args:
        in_channels:  Number of input feature maps.
        out_channels: Number of output feature maps.
        kernel_size:  Convolution kernel size (default 3).
        pool_size:    Max-pool kernel size (default 2).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class SimpleCNN(nn.Module):
    """Baseline CNN with 3 convolutional blocks + GAP + FC classifier.

    Expects 224×224 RGB inputs and produces logits over *num_classes*.

    Architecture overview::

        Input (3, 224, 224)
        └── ConvBlock(3 → 64)      → (64, 112, 112)
        └── ConvBlock(64 → 128)     → (128, 56, 56)
        └── ConvBlock(128 → 256)    → (256, 28, 28)
        └── Global Average Pooling   → (256,)
        └── Dropout
        └── Linear(256 → num_classes)

    Args:
        in_channels: Number of input channels (default 3 for RGB).
        num_classes: Number of output logits (default 10 for EuroSAT).
        dropout:     Dropout probability before the FC layer (default 0.3).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal initialisation for all Conv and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

    def describe(self) -> None:
        """Print a summary of layers and total trainable parameters."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(self)
        print(f"\nTotal trainable parameters: {total:,}")


def build_classifier(
    backbone: str = "simple_cnn",
    num_classes: int = 10,
    pretrained: bool = False,
    dropout: float = 0.3,
    in_channels: int = 3,
) -> nn.Module:
    """Model factory that returns a new *SimpleCNN*.

    Args:
        backbone: Ignored for this baseline (always returns ``SimpleCNN``).
        num_classes: Output dimensionality.
        pretrained: Ignored (no pretrained weights available for ``SimpleCNN``).
        dropout: Dropout probability.
        in_channels: Number of input channels.

    Returns:
        A ``SimpleCNN`` instance.
    """
    return SimpleCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
    )
