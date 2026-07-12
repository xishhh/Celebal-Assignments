"""Transfer-learning model using pretrained ResNet-18."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor
from torchvision.models import ResNet18_Weights


class TransferLearningModel(nn.Module):
    """ResNet-18 with a custom classifier head, pretrained on ImageNet.

    The backbone is kept frozen by default; only the classifier head is
    trainable.  Use :meth:`freeze_backbone` / :meth:`unfreeze_last_blocks`
    to control which layers receive gradients.

    Architecture::

        Input (3, H, W)
        └── ResNet-18 backbone (up to & including GAP)  → (512,)
        └── Linear(512 → 512) + ReLU + Dropout
        └── Linear(512 → num_classes)                   → logits

    Args:
        num_classes: Number of output logits (default 10).
        pretrained: If ``True``, load ImageNet-1K weights.
        dropout: Dropout probability between the two Linear layers.
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Remove the original FC layer — backbone now stops at GAP → 512-d vector
        in_features = resnet.fc.in_features  # 512
        resnet.fc = nn.Identity()

        self.backbone = resnet

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        self._init_classifier()

    def _init_classifier(self) -> None:
        """Initialise the custom classifier head with small normal weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Classification forward pass.

        Args:
            x: Input tensor of shape ``(N, 3, H, W)``.

        Returns:
            Logits of shape ``(N, num_classes)``.
        """
        features = self.backbone(x)         # (N, 512)
        return self.classifier(features)    # (N, num_classes)

    def extract_features(self, x: Tensor) -> Tensor:
        """Return the 512-dimensional embedding after GAP, *before* the classifier.

        This is the feature vector used for downstream tasks such as
        cosine-similarity change detection.

        Args:
            x: Input tensor of shape ``(N, 3, H, W)``.

        Returns:
            Feature tensor of shape ``(N, 512)``.
        """
        return self.backbone(x)

    # ── Freezing helpers ──────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze every backbone parameter so only the classifier head trains."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_blocks(self) -> None:
        """Unfreeze ``layer3`` and ``layer4`` of the backbone.

        Earlier layers (conv1, bn1, layer1, layer2) remain frozen.
        Call :meth:`freeze_backbone` first to reset, then this method
        to selectively enable fine-tuning of deeper features.
        """
        for param in self.backbone.layer3.parameters():
            param.requires_grad = True
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

    # ── Parameter counting ────────────────────────────────────────────

    def count_trainable_parameters(self) -> int:
        """Return the number of parameters that currently require gradients."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def count_total_parameters(model: nn.Module) -> int:
        """Return the total number of parameters (trainable or not)."""
        return sum(p.numel() for p in model.parameters())

    def describe(self) -> None:
        """Print a detailed summary of parameter counts."""
        total = self.count_total_parameters(self)
        trainable = self.count_trainable_parameters()
        frozen = total - trainable

        print(self)
        print(f"\n{'=' * 50}")
        print(f"  Total parameters:           {total:>10,}")
        print(f"  Trainable parameters:       {trainable:>10,}")
        print(f"  Frozen parameters:          {frozen:>10,}")
        print(f"  Trainable / Total ratio:    {trainable / total:>9.2%}")
        print(f"{'=' * 50}")
