"""Visual change heatmaps for temporal image pairs.

This module produces pixel-level difference visualisations between
two images of the same scene at different times.  It does **not**
use neural-network feature maps or Grad-CAM — the heatmap reflects
purely visual (pixel) differences.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# PART 1 + 2 — HEATMAP GENERATION & IMAGE PROCESSING
# ═══════════════════════════════════════════════════════════════════════

class ChangeHeatmapGenerator:
    """Generate visual change heatmaps for temporal image pairs.

    The pipeline for each pair:

    1. Load and resize both images to identical dimensions.
    2. Convert to RGB.
    3. Compute absolute pixel difference.
    4. Convert difference to grayscale intensity.
    5. Normalise to ``[0, 1]``.
    6. Apply a perceptually uniform colormap.

    Parameters
    ----------
    target_size : Tuple[int, int]
        ``(width, height)`` to resize images to (default ``(224, 224)``).
    alpha : float
        Opacity for the heatmap overlay in ``[0, 1]`` (default ``0.5``).
    colormap : str
        Matplotlib colormap name (default ``"inferno"``).
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        alpha: float = 0.5,
        colormap: str = "inferno",
    ) -> None:
        self.target_size = target_size
        self.alpha = alpha
        self.cmap = plt.get_cmap(colormap)

    # ── Core image processing ─────────────────────────────────────────

    @staticmethod
    def _load_and_resize(path: Path, size: Tuple[int, int]) -> np.ndarray:
        """Load an image from disk, resize, and return an RGB array.

        Parameters
        ----------
        path : Path
            Image file path.
        size : Tuple[int, int]
            ``(width, height)`` target size.

        Returns
        -------
        np.ndarray
            RGB image of shape ``(H, W, 3)`` with dtype ``uint8``.
        """
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _compute_difference(
        img1: np.ndarray,
        img2: np.ndarray,
    ) -> np.ndarray:
        """Compute the normalised absolute pixel difference.

        Parameters
        ----------
        img1 : np.ndarray
            RGB image ``(H, W, 3)`` dtype ``uint8``.
        img2 : np.ndarray
            RGB image ``(H, W, 3)`` dtype ``uint8``.

        Returns
        -------
        np.ndarray
            Grayscale difference map ``(H, W)`` in ``[0, 1]``.
        """
        diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
        gray = diff.mean(axis=2)                      # (H, W) 0-255
        return gray / 255.0                            # normalise [0, 1]

    @staticmethod
    def _apply_colormap(
        diff_norm: np.ndarray,
        cmap: plt.Colormap,
    ) -> np.ndarray:
        """Map a normalised grayscale array to a colormap.

        Parameters
        ----------
        diff_norm : np.ndarray
            ``(H, W)`` values in ``[0, 1]``.
        cmap : plt.Colormap
            Matplotlib colormap instance.

        Returns
        -------
        np.ndarray
            RGB heatmap ``(H, W, 3)`` dtype ``uint8``.
        """
        colored = cmap(diff_norm)                     # (H, W, 4) RGBA
        return (colored[:, :, :3] * 255).astype(np.uint8)

    # ── Single-pair heatmap ───────────────────────────────────────────

    def generate_heatmap(
        self,
        image_t1: Path,
        image_t2: Path,
        similarity: float,
        threshold: float,
        save_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Generate a 4-panel change heatmap figure for one pair.

        Panels
        ------
        1. Original T1 image
        2. Original T2 image
        3. Absolute-difference heatmap (colormap)
        4. Heatmap overlay on T2 (alpha-blended)

        The similarity score, threshold, and change decision are
        displayed as a title.

        Parameters
        ----------
        image_t1 : Path
            Path to the T1 image.
        image_t2 : Path
            Path to the T2 image.
        similarity : float
            Cosine similarity score for this pair.
        threshold : float
            Operating threshold (below => changed).
        save_path : Optional[Path]
            If provided, the figure is saved to this path.

        Returns
        -------
        Dict[str, Any]
            Keys: ``similarity``, ``threshold``, ``changed``,
            ``image_t1``, ``image_t2``.
        """
        img1 = self._load_and_resize(image_t1, self.target_size)
        img2 = self._load_and_resize(image_t2, self.target_size)

        diff_norm = self._compute_difference(img1, img2)
        heatmap = self._apply_colormap(diff_norm, self.cmap)
        overlay = self._blend_overlay(img2, heatmap)

        changed = similarity < threshold
        status = "CHANGE DETECTED" if changed else "NO CHANGE"
        title = (
            f"Similarity : {similarity:.2f}    "
            f"Threshold : {threshold:.2f}    "
            f"Status : {status}"
        )

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
        titles = ["T1 Image", "T2 Image", "Difference Heatmap", "Overlay on T2"]

        images = [img1, img2, heatmap, overlay]
        for ax, im, ti in zip(axes, images, titles):
            ax.imshow(im)
            ax.set_title(ti, fontsize=11)
            ax.axis("off")

        fig.suptitle(title, fontsize=13, y=1.02)
        fig.tight_layout()

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Heatmap saved to {save_path}")

        plt.close(fig)

        return {
            "similarity": round(similarity, 6),
            "threshold": threshold,
            "changed": changed,
            "image_t1": str(image_t1),
            "image_t2": str(image_t2),
        }

    # ── Overlay helper ────────────────────────────────────────────────

    def _blend_overlay(
        self,
        img: np.ndarray,
        heatmap: np.ndarray,
    ) -> np.ndarray:
        """Alpha-blend the heatmap over the T2 image.

        Parameters
        ----------
        img : np.ndarray
            RGB image ``(H, W, 3)`` dtype ``uint8``.
        heatmap : np.ndarray
            RGB heatmap ``(H, W, 3)`` dtype ``uint8``.

        Returns
        -------
        np.ndarray
            Blended RGB image ``(H, W, 3)`` dtype ``uint8``.
        """
        return (img * (1 - self.alpha) + heatmap * self.alpha).astype(np.uint8)

    # ── Batch processing ──────────────────────────────────────────────

    def generate_batch(
        self,
        pairs: List[Dict[str, Any]],
        output_dir: Path,
        summary_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Generate heatmaps for multiple image pairs.

        Parameters
        ----------
        pairs : List[Dict[str, Any]]
            Each dict must have keys ``image_t1``, ``image_t2``,
            ``similarity``, ``threshold``.
        output_dir : Path
            Directory where heatmap PNGs are saved.
        summary_path : Optional[Path]
            If provided, saves the summary CSV to this path.

        Returns
        -------
        List[Dict[str, Any]]
            Result dicts (one per pair) with heatmap paths included.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: List[Dict[str, Any]] = []
        for i, pair in enumerate(pairs, start=1):
            save_path = output_dir / f"pair_{i:03d}.png"
            result = self.generate_heatmap(
                image_t1=Path(pair["image_t1"]),
                image_t2=Path(pair["image_t2"]),
                similarity=pair["similarity"],
                threshold=pair["threshold"],
                save_path=save_path,
            )
            result["pair_id"] = f"pair_{i:03d}"
            results.append(result)

        if summary_path is not None:
            self.save_summary(results, summary_path)

        return results

    # ── CSV summary ───────────────────────────────────────────────────

    @staticmethod
    def save_summary(
        results: List[Dict[str, Any]],
        path: Path,
    ) -> None:
        """Save a summary CSV of all generated heatmaps.

        Columns: ``pair_id``, ``similarity``, ``threshold``, ``changed``,
        ``image_t1``, ``image_t2``.

        Parameters
        ----------
        results : List[Dict[str, Any]]
            Result dicts from :meth:`generate_heatmap` or
            :meth:`generate_batch`.
        path : Path
            Output CSV path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "pair_id", "similarity", "threshold", "changed",
                "image_t1", "image_t2",
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "pair_id": r.get("pair_id", ""),
                    "similarity": r["similarity"],
                    "threshold": r["threshold"],
                    "changed": r["changed"],
                    "image_t1": r["image_t1"],
                    "image_t2": r["image_t2"],
                })
        print(f"Heatmap summary saved to {path}  |  {len(results)} rows")
