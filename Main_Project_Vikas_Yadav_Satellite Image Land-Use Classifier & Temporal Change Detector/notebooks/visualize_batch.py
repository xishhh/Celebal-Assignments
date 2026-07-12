#!/usr/bin/env python
"""Visualise one batch of EuroSAT images with ground-truth labels.

Usage:
    python notebooks/visualize_batch.py
"""

import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid

# Ensure the project root is on sys.path so we can import src & config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import EUROSAT_ROOT, CLASS_NAMES, IMAGE_SIZE, BATCH_SIZE, SEED  # noqa: E402
from src.dataloader import build_dataloaders  # noqa: E402
from src.dataset import _discover_paths  # noqa: E402
from src.transforms import get_eval_transforms  # noqa: E402


def denormalize(
    tensor: torch.Tensor,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Reverse ImageNet normalisation for display."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    return tensor.permute(1, 2, 0).numpy()


def visualize_five_per_class(save_dir: Path) -> None:
    """Display 5 random samples per EuroSAT class in a grid."""
    all_paths = _discover_paths(EUROSAT_ROOT)
    rng = random.Random(SEED)

    class_to_paths = {name: [] for name in CLASS_NAMES}
    for p in all_paths:
        class_to_paths[p.parent.name].append(p)

    transform = get_eval_transforms(IMAGE_SIZE)

    fig, axes = plt.subplots(10, 5, figsize=(10, 16))
    fig.suptitle("5 Samples Per EuroSAT Class", fontsize=16, y=0.92)

    for row, cls in enumerate(CLASS_NAMES):
        paths = class_to_paths[cls]
        rng.shuffle(paths)
        samples = paths[:5]
        for col, p in enumerate(samples):
            img = Image.open(p).convert("RGB")
            img_t = transform(img)
            img_disp = denormalize(img_t)
            ax = axes[row, col]
            ax.imshow(img_disp)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cls, fontsize=8, rotation=0, ha="right", va="center",
                              labelpad=10)

    save_path = save_dir / "five_per_class.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"5-per-class visualisation saved to: {save_path}")


def visualize_class_distribution(save_dir: Path) -> None:
    """Plot class distribution bar chart for the full EuroSAT dataset."""
    all_paths = _discover_paths(EUROSAT_ROOT)
    class_counts = Counter(p.parent.name for p in all_paths)
    counts = [class_counts[name] for name in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(CLASS_NAMES, counts, color="steelblue")
    ax.set_title("EuroSAT Class Distribution", fontsize=14)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.tick_params(axis="x", rotation=45)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(count), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    save_path = save_dir / "class_distribution.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Class distribution plot saved to: {save_path}")


def main():
    reports_dir = PROJECT_ROOT / "reports"

    # ── 1. Sample batch visualization ──
    loaders = build_dataloaders(
        root=EUROSAT_ROOT,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
        seed=SEED,
    )

    images, labels = next(iter(loaders["train"]))

    print(f"Batch shape: {images.shape}      # (N, C, H, W)")
    print(f"Labels shape: {labels.shape}")
    print(f"Label values: {labels.tolist()}")
    print(f"Label names:  {[CLASS_NAMES[l] for l in labels.tolist()]}")

    nrow = 8
    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    grid_img = denormalize(grid)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(grid_img)
    ax.axis("off")
    ax.set_title("EuroSAT Training Batch (denormalised)", fontsize=14)

    for i in range(len(labels)):
        row = i // nrow
        col = i % nrow
        x = col * (IMAGE_SIZE[1] + 4) + IMAGE_SIZE[1] // 2
        y = row * (IMAGE_SIZE[0] + 4) + IMAGE_SIZE[0] + 8
        ax.text(x, y, CLASS_NAMES[labels[i]], ha="center", va="top", fontsize=6)

    save_path = reports_dir / "sample_batch.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Batch visualisation saved to: {save_path}")

    # ── 2. 5 samples per class ──
    visualize_five_per_class(reports_dir)

    # ── 3. Class distribution ──
    visualize_class_distribution(reports_dir)


if __name__ == "__main__":
    main()
