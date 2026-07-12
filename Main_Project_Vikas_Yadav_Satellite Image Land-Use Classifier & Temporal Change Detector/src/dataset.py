"""PyTorch Dataset for EuroSAT land-use classification."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

from .transforms import get_eval_transforms, get_train_transforms


class EuroSATDataset(Dataset):
    """PyTorch Dataset that walks a EuroSAT folder tree.

    Expects the following layout under *root*::

        root/
            AnnualCrop/
                *.jpg
            Forest/
                *.jpg
            …10 classes total…

    Args:
        root: Path to the folder containing class-named subdirectories.
        class_names: Ordered list of class folder names used to
            build the label-to-index mapping.
        file_paths: Pre-determined list of image paths for this split.
            If *None*, the Dataset is empty.
        transform: Callable transforming a PIL image to a tensor.
    """

    def __init__(
        self,
        root: Path,
        class_names: List[str],
        file_paths: Optional[List[Path]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.class_names = class_names
        self.class_to_idx: Dict[str, int] = {name: i for i, name in enumerate(class_names)}
        self.transform = transform
        self.file_paths: List[Path] = file_paths or []

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Tuple:
        """Return (image_tensor, label_index) for the given index."""
        path = self.file_paths[index]
        image = Image.open(path).convert("RGB")
        label = self.class_to_idx[path.parent.name]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def _discover_paths(root: Path) -> List[Path]:
    """Return all ``*.jpg`` paths sorted recursively under *root*."""
    return sorted([p for p in root.rglob("*.jpg") if p.is_file()])


def _stratified_split(
    paths: List[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Deterministic stratified split based on parent folder name."""
    groups: Dict[str, List[Path]] = defaultdict(list)
    for p in paths:
        groups[p.parent.name].append(p)

    rng = random.Random(seed)

    train_paths: List[Path] = []
    val_paths: List[Path] = []
    test_paths: List[Path] = []

    for group in groups.values():
        rng.shuffle(group)
        n = len(group)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        train_paths.extend(group[:n_train])
        val_paths.extend(group[n_train : n_train + n_val])
        test_paths.extend(group[n_train + n_val :])

    return sorted(train_paths), sorted(val_paths), sorted(test_paths)


def group_images_into_blocks(
    all_paths: List[Path],
    block_size: int,
) -> Dict[str, List[List[Path]]]:
    """Group images into spatial blocks per class.

    Within each class, images are sorted by filename (deterministic) and
    grouped into consecutive blocks of *block_size*.

    Parameters
    ----------
    all_paths : List[Path]
        All EuroSAT image paths.
    block_size : int
        Number of images per spatial block.

    Returns
    -------
    Dict[str, List[List[Path]]]
        Mapping ``class_name → [block1, block2, …]`` where each block
        is a list of image paths.
    """
    groups: Dict[str, List[Path]] = defaultdict(list)
    for p in all_paths:
        groups[p.parent.name].append(p)

    class_blocks: Dict[str, List[List[Path]]] = {}
    for cls in sorted(groups):
        paths = sorted(groups[cls])
        blocks = [paths[i:i + block_size] for i in range(0, len(paths), block_size)]
        class_blocks[cls] = blocks

    return class_blocks


def _block_split(
    all_paths: List[Path],
    block_size: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Deterministic block-based split.

    Within each class, images are grouped into spatial blocks, blocks are
    shuffled, then assigned proportionally to train / val / test.  Images
    in the same block never cross splits.
    """
    class_blocks = group_images_into_blocks(all_paths, block_size)
    rng = random.Random(seed)

    train_paths: List[Path] = []
    val_paths: List[Path] = []
    test_paths: List[Path] = []

    for cls, blocks in class_blocks.items():
        rng.shuffle(blocks)
        n = len(blocks)
        n_train = max(1, round(train_ratio * n))
        n_val = max(1, round(val_ratio * n))
        remaining = n - n_train - n_val
        if remaining <= 0 and n >= 3:
            n_train = n - 2
            n_val = 1

        for block in blocks[:n_train]:
            train_paths.extend(block)
        for block in blocks[n_train:n_train + n_val]:
            val_paths.extend(block)
        for block in blocks[n_train + n_val:]:
            test_paths.extend(block)

    return sorted(train_paths), sorted(val_paths), sorted(test_paths)


def create_spatial_datasets(
    root: Path,
    class_names: List[str],
    image_size: Tuple[int, int],
    block_size: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, EuroSATDataset]:
    """Build train / val / test datasets using a spatial block split.

    Images are grouped into blocks of *block_size* (by sorted filename
    order) and entire blocks are assigned to one split, preventing
    geographic leakage between train and test sets.

    Args:
        root: Path to the folder with class subdirectories.
        class_names: Ordered list of class folder names.
        image_size: (height, width) passed to transforms.
        block_size: Images per spatial block.
        train_ratio: Fraction of blocks for training (default 0.70).
        val_ratio: Fraction of blocks for validation (default 0.15).
        seed: RNG seed for block shuffling.

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, ``"test"``.
    """
    all_paths = _discover_paths(root)
    train_paths, val_paths, test_paths = _block_split(
        all_paths, block_size, train_ratio, val_ratio, seed,
    )

    datasets = {
        "train": EuroSATDataset(
            root=root,
            class_names=class_names,
            file_paths=train_paths,
            transform=get_train_transforms(image_size),
        ),
        "val": EuroSATDataset(
            root=root,
            class_names=class_names,
            file_paths=val_paths,
            transform=get_eval_transforms(image_size),
        ),
        "test": EuroSATDataset(
            root=root,
            class_names=class_names,
            file_paths=test_paths,
            transform=get_eval_transforms(image_size),
        ),
    }
    return datasets


def create_eurosat_datasets_stratified(
    root: Path,
    class_names: List[str],
    image_size: Tuple[int, int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, EuroSATDataset]:
    """Build train / val / test datasets using a random stratified split.

    Images are shuffled per-class to preserve class balance across splits.
    This is the random-split baseline; use ``create_eurosat_datasets``
    for the spatial block split.

    Args:
        root: Path to the folder with class subdirectories.
        class_names: Ordered list of class folder names.
        image_size: (height, width) passed to transforms.
        train_ratio: Fraction of data used for training.
        val_ratio: Fraction of data used for validation.
        seed: RNG seed for reproducible splitting.

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, ``"test"``.
    """
    all_paths = _discover_paths(root)
    train_paths, val_paths, test_paths = _stratified_split(
        all_paths, train_ratio, val_ratio, seed
    )

    datasets = {
        "train": EuroSATDataset(
            root=root,
            class_names=class_names,
            file_paths=train_paths,
            transform=get_train_transforms(image_size),
        ),
        "val": EuroSATDataset(
            root=root,
            class_names=class_names,
            file_paths=val_paths,
            transform=get_eval_transforms(image_size),
        ),
        "test": EuroSATDataset(
            root=root,
            class_names=class_names,
            file_paths=test_paths,
            transform=get_eval_transforms(image_size),
        ),
    }
    return datasets


def create_eurosat_datasets(
    root: Path,
    class_names: List[str],
    image_size: Tuple[int, int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    block_size: int = 100,
) -> Dict[str, EuroSATDataset]:
    """Build train / val / test datasets using a spatial block split.

    This is the primary data pipeline.  Images are grouped into spatial
    blocks (by sorted filename order) and entire blocks are assigned to
    one split, preventing geographic leakage.

    Args:
        root: Path to the folder with class subdirectories.
        class_names: Ordered list of class folder names.
        image_size: (height, width) passed to transforms.
        train_ratio: Fraction of blocks for training (default 0.70).
        val_ratio: Fraction of blocks for validation (default 0.15).
        seed: RNG seed for block shuffling.
        block_size: Images per spatial block (default 100).

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, ``"test"``.
    """
    return create_spatial_datasets(
        root=root,
        class_names=class_names,
        image_size=image_size,
        block_size=block_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
