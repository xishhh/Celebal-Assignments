"""DataLoader builders for train / val / test."""

from __future__ import annotations

from typing import Dict, Tuple

from torch.utils.data import DataLoader, Dataset

from .dataset import EuroSATDataset, create_eurosat_datasets


def build_dataloaders(
    root,
    class_names,
    image_size: Tuple[int, int],
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """Create train / val / test DataLoaders from EuroSAT folder.

    Args:
        root: Path to the folder containing class subdirectories.
        class_names: Ordered list of class folder names.
        image_size: (height, width) for resizing.
        batch_size: Samples per batch.
        num_workers: Subprocess workers for data loading.
        pin_memory: Whether to pin memory for GPU transfer.
        train_ratio: Fraction for training (default 0.70).
        val_ratio: Fraction for validation (default 0.15).
        seed: Deterministic split seed.

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, ``"test"``.
    """
    datasets: Dict[str, EuroSATDataset] = create_eurosat_datasets(
        root=root,
        class_names=class_names,
        image_size=image_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    loaders: Dict[str, DataLoader] = {}
    for split_name, ds in datasets.items():
        loaders[split_name] = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split_name == "train"),
        )
    return loaders


def print_dataset_stats(datasets: Dict[str, Dataset], class_names):
    """Print number of classes, per-split sizes, and a sample batch shape.

    Args:
        datasets: Dict with keys ``"train"``, ``"val"``, ``"test"``.
        class_names: List of class label strings.
    """
    print(f"Number of classes: {len(class_names)}")
    print(f"Class labels: {class_names}")
    print()

    for split_name, ds in datasets.items():
        print(f"{split_name:>6s}: {len(ds):>5d} images")

    print()
    sample_batch = next(iter(DataLoader(datasets["train"], batch_size=4)))
    images, labels = sample_batch
    print(f"Batch image tensor shape: {images.shape}")
    print(f"Batch label tensor shape: {labels.shape}")
