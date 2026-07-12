"""Embedding extraction, persistence, and PCA visualisation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm

from .transfer_model import TransferLearningModel
from .utils import load_checkpoint, setup_logger

logger = setup_logger(__name__)


class EmbeddingExtractor:
    """Extract 512-d feature embeddings from a trained TransferLearningModel.

    The classifier head is automatically stripped so that only the
    backbone (conv1 … avgpool) produces the embedding.

    Args:
        checkpoint_path: Path to a ``resnet18_final.pt`` checkpoint.
        device: ``"cuda"`` or ``"cpu"``.
        num_classes: Must match the value used during training (default 10).
        dropout: Must match the value used during training (default 0.3).
    """

    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        # Load full model from checkpoint
        model = TransferLearningModel(
            num_classes=num_classes,
            pretrained=False,
            dropout=dropout,
        )
        load_checkpoint(checkpoint_path, model)
        model = model.to(device)
        model.eval()

        # Strip the classifier head — keep only the backbone
        self.backbone = model.backbone
        self.backbone.eval()
        self.device = device

        logger.info(f"EmbeddingExtractor initialised | device={device}")

    # ── Public extraction API ─────────────────────────────────────────

    @torch.no_grad()
    def extract(self, image: torch.Tensor) -> torch.Tensor:
        """Extract a single 512-d embedding from one image.

        Args:
            image: Single image tensor of shape ``(C, H, W)``.

        Returns:
            Embedding of shape ``(512,)``.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        embedding = self.backbone(image.to(self.device, non_blocking=True))
        return embedding.squeeze(0).cpu()

    @torch.no_grad()
    def extract_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Extract embeddings for a batch of images.

        Args:
            images: Batch tensor of shape ``(N, C, H, W)``.

        Returns:
            Embedding tensor of shape ``(N, 512)``.
        """
        return self.backbone(images.to(self.device, non_blocking=True)).cpu()

    @torch.no_grad()
    def extract_dataset(
        self,
        loader: DataLoader,
        desc: str = "Extracting embeddings",
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract embeddings for an entire DataLoader.

        Args:
            loader: DataLoader yielding ``(images, labels)`` or
                ``(images, labels, _, filenames)`` (e.g. UC Merced).

        Returns:
            Tuple of ``(embeddings, labels, filenames)`` where
            ``embeddings`` is ``(N, 512)``, ``labels`` is ``(N,)``,
            and ``filenames`` is a list of strings.
        """
        all_emb: List[np.ndarray] = []
        all_labels: List[int] = []
        all_files: List[str] = []

        pbar = tqdm(loader, desc=desc)
        for batch in pbar:
            images = batch[0].to(self.device, non_blocking=True)
            labels = batch[1]

            emb = self.backbone(images).cpu().numpy()
            all_emb.append(emb)
            all_labels.extend(labels.cpu().numpy().tolist())

            # Try to get filenames from the batch (e.g. UCMercedDataset)
            if len(batch) >= 4:
                all_files.extend(batch[3])
            else:
                all_files.extend([f"sample_{len(all_labels) + i}" for i in range(len(emb))])

        return (
            np.concatenate(all_emb, axis=0),
            np.array(all_labels),
            all_files,
        )

    # ── Persistence ───────────────────────────────────────────────────

    @staticmethod
    def save_embeddings(
        path: Path,
        embeddings: np.ndarray,
        labels: np.ndarray,
        filenames: List[str],
    ) -> None:
        """Save embeddings, labels, and filenames to a ``.npz`` file.

        Args:
            path: Output path (e.g. ``embeddings/eurosat_embeddings.npz``).
            embeddings: Array of shape ``(N, D)``.
            labels: Array of shape ``(N,)``.
            filenames: List of ``N`` filename strings.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=embeddings,
            labels=labels,
            filenames=np.array(filenames, dtype=object),
        )
        logger.info(f"Embeddings saved to {path}  |  shape={embeddings.shape}")

    @staticmethod
    def load_embeddings(path: Path) -> Dict[str, Any]:
        """Load a previously saved ``.npz`` embeddings file.

        Args:
            path: Path to a ``.npz`` file saved by :meth:`save_embeddings`.

        Returns:
            Dictionary with keys ``embeddings``, ``labels``, ``filenames``.
        """
        data = np.load(path, allow_pickle=True)
        logger.info(f"Embeddings loaded from {path}  |  shape={data['embeddings'].shape}")
        return {
            "embeddings": data["embeddings"],
            "labels": data["labels"],
            "filenames": data["filenames"].tolist(),
        }

    # ── Statistics ────────────────────────────────────────────────────

    @staticmethod
    def print_stats(embeddings: np.ndarray) -> None:
        """Print summary statistics of the embedding array.

        Args:
            embeddings: Array of shape ``(N, D)``.
        """
        print(f"\n{'=' * 50}")
        print(f"  Embedding dimension:      {embeddings.shape[1]}")
        print(f"  Number of samples:        {embeddings.shape[0]:,}")
        print(f"  Mean:                     {embeddings.mean():.6f}")
        print(f"  Std:                      {embeddings.std():.6f}")
        print(f"  Min:                      {embeddings.min():.6f}")
        print(f"  Max:                      {embeddings.max():.6f}")
        print(f"{'=' * 50}")

    # ── PCA visualisation ─────────────────────────────────────────────

    @staticmethod
    def plot_pca(
        embeddings: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        save_path: Path,
    ) -> None:
        """Fit PCA on embeddings and plot the first two components.

        Each point is coloured according to its land-use class.

        Args:
            embeddings: Array of shape ``(N, D)``.
            labels: Array of shape ``(N,)`` with integer label indices.
            class_names: Ordered list of class label strings.
            save_path: Destination for the PNG figure.
        """
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(10, 8))
        unique_labels = np.unique(labels)
        cmap = plt.colormaps["tab10"]
        for idx in unique_labels:
            mask = labels == idx
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=[cmap(idx % 10)],
                label=class_names[int(idx)],
                s=8,
                alpha=0.7,
            )
        ax.legend(loc="best", title="Classes", fontsize=8)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
        ax.set_title("PCA Projection of EuroSAT Embeddings")
        ax.grid(True, alpha=0.3)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"PCA plot saved to {save_path}")
