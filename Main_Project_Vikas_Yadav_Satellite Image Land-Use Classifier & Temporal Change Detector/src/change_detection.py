"""Temporal change detection via cosine similarity on embeddings.

Parts
-----
1. RegionPairGenerator  – simulate temporal T1/T2 pairs via offset-based regions
2. ChangeDetector       – cosine similarity, ROC, threshold, prediction
3. print_stats          – standalone summary printer
"""

from __future__ import annotations

import csv
import json
import random
import re as _re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


# ═══════════════════════════════════════════════════════════════════════
# PART 1 — TEMPORAL PAIR GENERATION VIA SPATIAL-OFFSET REGIONS
# ═══════════════════════════════════════════════════════════════════════

_PATCH_INDEX_RE = _re.compile(r"_(\d+)")


def _extract_patch_index(path: Path) -> int:
    """Return the numeric patch index from a EuroSAT filename.

    ``AnnualCrop_42.jpg`` → ``42``

    EuroSAT filenames follow the convention ``<ClassName>_<index>.jpg``
    where *index* is a 1-based raster-scan offset within the Sentinel-2
    tile.  Sorting by this index preserves the spatial layout of patches
    within each land-cover class.
    """
    match = _PATCH_INDEX_RE.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot extract patch index from {path.name}")
    return int(match.group(1))


class RegionPairGenerator:
    """Generate synthetic temporal pairs using spatial-offset regions.

    EuroSAT lacks real multi-temporal imagery, so temporal change is
    simulated by:

    1. **Natural numeric sort** — images within each class are sorted by
       their numeric patch index (``_1, _2, _3, …``), not alphabetically
       (``_1, _10, _100, …``).  This preserves the raster-scan order of
       patches within the original Sentinel-2 tile.

    2. **Spatial-offset regions** — patch indices are partitioned into
       consecutive ranges of ``block_size``.  A **region** is the set of
       *all* patches (across all classes) whose numeric index falls inside
       a given range.  For example, with ``block_size=100``, region ``0``
       contains every satellite patch with index 1-100 regardless of class.

    3. **Pairing within each region:**

       * **Unchanged** (label = 0): two patches from the **same class**
         but *different numeric offsets* within the region.
       * **Changed**   (label = 1): two patches from **different classes**
         but the *same numeric offset* within the region.

    Pairing across the *same offset* approximates "same location, different
    land-cover label" — the closest temporal simulation possible with a
    single-label dataset.  Pairing within the *same class* at different
    offsets represents spatial consistency of a land-cover type.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility (default ``42``).
    block_size : int, optional
        Number of consecutive patch indices per region (default ``100``).
    """

    def __init__(self, seed: int = 42, block_size: int = 100) -> None:
        self._rng = random.Random(seed)
        self.block_size = block_size

    # ── Region partitioning ───────────────────────────────────────────

    def _build_regions(
        self,
        all_paths: List[Path],
        class_to_idx: Dict[str, int],
    ) -> Dict[int, Dict[int, List[Tuple[int, Path]]]]:
        """Partition images by numeric offset into spatial-offset regions.

        Within each class, images are sorted by their numeric patch index.
        A **region** is a range of consecutive indices
        ``[N*block_size + 1, (N+1)*block_size]``.  Every image whose
        index falls in that range is placed in that region, grouped by
        class.

        Returns
        -------
        Dict[int, Dict[int, List[Tuple[int, Path]]]]
            ``{region_id: {class_idx: [(offset, path), …]}}``.
        """
        class_entries: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
        for p in all_paths:
            cls_name = p.parent.name
            offset = _extract_patch_index(p)
            class_entries[cls_name].append((offset, p))

        regions: Dict[int, Dict[int, List[Tuple[int, Path]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for cls_name, entries in class_entries.items():
            cls_idx = class_to_idx[cls_name]
            sorted_entries = sorted(entries, key=lambda x: x[0])
            for offset, path in sorted_entries:
                region_id = (offset - 1) // self.block_size
                regions[region_id][cls_idx].append((offset, path))

        return dict(regions)

    # ── Pair generation ───────────────────────────────────────────────

    def generate(
        self,
        all_paths: List[Path],
        class_to_idx: Dict[str, int],
        class_names: List[str],
        num_unchanged: int = 5000,
        num_changed: int = 5000,
    ) -> Tuple[List[str], List[str], List[int], List[str], List[str], List[int]]:
        """Generate synthetic temporal pairs from spatial-offset regions.

        Parameters
        ----------
        all_paths : List[Path]
            All EuroSAT image file paths.
        class_to_idx : Dict[str, int]
            Mapping ``class_name → class_index``.
        class_names : List[str]
            Ordered class label strings.
        num_unchanged : int
            Number of unchanged (label 0) pairs to generate.
        num_changed : int
            Number of changed (label 1) pairs to generate.

        Returns
        -------
        Tuple[List[str], List[str], List[int], List[str], List[str], List[int]]
            ``(paths_t1, paths_t2, labels, class_t1, class_t2, region_ids)``.
        """
        regions = self._build_regions(all_paths, class_to_idx)
        sorted_region_ids = sorted(regions.keys())

        pairs: List[Tuple[str, str, int, str, str, int]] = []
        seen: set = set()
        n_unchanged = 0
        n_changed = 0

        # ── Phase 1: unchanged pairs ──────────────────────────────────
        # Conditions: same class, same region, DIFFERENT offsets
        for region_id in sorted_region_ids:
            if n_unchanged >= num_unchanged:
                break
            region = regions[region_id]
            for cls_idx, entries in region.items():
                unique_offsets = {e[0] for e in entries}
                if len(unique_offsets) < 2:
                    continue
                if n_unchanged >= num_unchanged:
                    break
                n_possible = len(unique_offsets) * (len(unique_offsets) - 1) // 2
                n_take = min(num_unchanged - n_unchanged, n_possible)
                n_target = n_unchanged + n_take
                attempts = 0
                while n_unchanged < n_target and attempts < n_take * 10:
                    attempts += 1
                    if len(entries) < 2:
                        break
                    (off_a, path_a), (off_b, path_b) = self._rng.sample(entries, 2)
                    if off_a == off_b:
                        continue
                    key = (str(path_a), str(path_b))
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append((
                        str(path_a), str(path_b), 0,
                        class_names[cls_idx], class_names[cls_idx],
                        region_id,
                    ))
                    n_unchanged += 1

        # ── Phase 2: changed pairs ────────────────────────────────────
        # Conditions: same offset within region, DIFFERENT classes
        for region_id in sorted_region_ids:
            if n_changed >= num_changed:
                break
            region = regions[region_id]

            offset_to_classes: Dict[int, Dict[int, Path]] = defaultdict(dict)
            for cls_idx, entries in region.items():
                for offset, path in entries:
                    offset_to_classes[offset][cls_idx] = path

            multi_class_offsets = [
                off for off, cd in offset_to_classes.items() if len(cd) >= 2
            ]
            self._rng.shuffle(multi_class_offsets)

            for offset in multi_class_offsets:
                if n_changed >= num_changed:
                    break
                classes = list(offset_to_classes[offset].items())
                n_possible = len(classes) * (len(classes) - 1)
                attempts = 0
                while n_changed < num_changed and attempts < n_possible * 10:
                    attempts += 1
                    (cls1, path1), (cls2, path2) = self._rng.sample(classes, 2)
                    key = (str(path1), str(path2))
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append((
                        str(path1), str(path2), 1,
                        class_names[cls1], class_names[cls2],
                        region_id,
                    ))
                    n_changed += 1

        self._rng.shuffle(pairs)

        paths_t1 = [p[0] for p in pairs]
        paths_t2 = [p[1] for p in pairs]
        labels = [p[2] for p in pairs]
        cls_t1 = [p[3] for p in pairs]
        cls_t2 = [p[4] for p in pairs]
        region_ids = [p[5] for p in pairs]

        print(
            f"Generated {len(labels)} temporal pairs  "
            f"(unchanged={labels.count(0)}, changed={labels.count(1)})  "
            f"from {len(regions)} offset-based regions"
        )
        return paths_t1, paths_t2, labels, cls_t1, cls_t2, region_ids

    # ── CSV persistence ───────────────────────────────────────────────

    @staticmethod
    def save_metadata(
        path: Path,
        paths_t1: List[str],
        paths_t2: List[str],
        labels: List[int],
        class_t1: List[str],
        class_t2: List[str],
    ) -> None:
        """Save pair metadata to a CSV file.

        Columns: ``image_t1``, ``image_t2``, ``label``, ``class_t1``, ``class_t2``.

        Parameters
        ----------
        path : Path
            Output CSV path.
        paths_t1 : List[str]
            Paths or filenames for the T1 image.
        paths_t2 : List[str]
            Paths or filenames for the T2 image.
        labels : List[int]
            0 = unchanged, 1 = changed.
        class_t1 : List[str]
            Class label of the T1 image.
        class_t2 : List[str]
            Class label of the T2 image.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_t1", "image_t2", "label", "class_t1", "class_t2"])
            writer.writerows(zip(paths_t1, paths_t2, labels, class_t1, class_t2))
        print(f"Change pairs metadata saved to {path}  |  {len(labels)} rows")

    @staticmethod
    def save_similarity_scores(
        path: Path,
        paths_t1: List[str],
        paths_t2: List[str],
        labels: List[int],
        similarities: np.ndarray,
        class_t1: List[str],
        class_t2: List[str],
        region_ids: Optional[List[int]] = None,
    ) -> None:
        """Save similarity scores and metadata to CSV.

        Columns: ``image_t1``, ``image_t2``, ``label``, ``similarity``,
        ``class_t1``, ``class_t2``, ``region_id``.

        Parameters
        ----------
        path : Path
            Output CSV path.
        paths_t1 : List[str]
            T1 image paths.
        paths_t2 : List[str]
            T2 image paths.
        labels : List[int]
            0 = unchanged, 1 = changed.
        similarities : np.ndarray
            Cosine similarity scores, shape ``(N,)``.
        class_t1 : List[str]
            Class label of the T1 image.
        class_t2 : List[str]
            Class label of the T2 image.
        region_ids : List[int], optional
            Region id for each pair.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["image_t1", "image_t2", "label", "similarity", "class_t1", "class_t2"]
        rows = list(zip(paths_t1, paths_t2, labels, similarities.tolist(), class_t1, class_t2))
        if region_ids is not None:
            fieldnames.append("region_id")
            rows = [r + (rid,) for r, rid in zip(rows, region_ids)]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            writer.writerows(rows)
        print(f"Similarity scores saved to {path}  |  {len(labels)} rows")


# ═══════════════════════════════════════════════════════════════════════
# PART 2 + 3 — COSINE SIMILARITY & CHANGE DETECTION
# ═══════════════════════════════════════════════════════════════════════

class ChangeDetector:
    """Detect land-use change between image pairs using cosine similarity.

    All core methods are static (``compute_similarity``,
    ``compute_batch_similarity``, ``predict_change``, ``generate_roc``,
    ``select_threshold``) so they can be used as plain functions or
    composed into larger pipelines.
    """

    # ── Similarity ────────────────────────────────────────────────────

    @staticmethod
    def compute_similarity(
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Cosine similarity between two embedding vectors.

        Parameters
        ----------
        embedding1 : np.ndarray
            1-D vector of shape ``(D,)``.
        embedding2 : np.ndarray
            1-D vector of shape ``(D,)``.

        Returns
        -------
        float
            Cosine similarity in ``[-1, 1]``.
        """
        v1 = embedding1 / (np.linalg.norm(embedding1) + 1e-12)
        v2 = embedding2 / (np.linalg.norm(embedding2) + 1e-12)
        return float(np.dot(v1, v2))

    @staticmethod
    def compute_batch_similarity(
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
    ) -> np.ndarray:
        """Pairwise cosine similarity between two batches of embeddings.

        Parameters
        ----------
        embeddings1 : np.ndarray
            Array of shape ``(N, D)``.
        embeddings2 : np.ndarray
            Array of shape ``(N, D)``.

        Returns
        -------
        np.ndarray
            Array of shape ``(N,)`` with cosine similarities.
        """
        norms1 = np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-12
        norms2 = np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-12
        return (embeddings1 / norms1 * embeddings2 / norms2).sum(axis=1)

    # ── Change prediction ────────────────────────────────────────────

    @staticmethod
    def predict_change(
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        threshold: float,
    ) -> Dict[str, Any]:
        """Predict whether change occurred between two images.

        Parameters
        ----------
        embedding1 : np.ndarray
            Embedding of the T1 image, shape ``(D,)``.
        embedding2 : np.ndarray
            Embedding of the T2 image, shape ``(D,)``.
        threshold : float
            Cosine-similarity threshold.  ``similarity < threshold``
            means *changed*.

        Returns
        -------
        Dict[str, Any]
            ``{"similarity": float, "threshold": float, "changed": bool}``.
        """
        sim = ChangeDetector.compute_similarity(embedding1, embedding2)
        return {
            "similarity": round(sim, 6),
            "threshold": threshold,
            "changed": sim < threshold,
        }

    # ── ROC analysis ─────────────────────────────────────────────────

    @staticmethod
    def generate_roc(
        similarities: np.ndarray,
        ground_truth: np.ndarray,
        save_curve: Optional[Path] = None,
        save_metrics: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Compute ROC curve and AUC.

        Parameters
        ----------
        similarities : np.ndarray
            Cosine similarity scores of shape ``(N,)``.
        ground_truth : np.ndarray
            Binary labels (0 = unchanged, 1 = changed).
        save_curve : Optional[Path]
            If provided, saves the ROC curve figure to this path.
        save_metrics : Optional[Path]
            If provided, saves a JSON with AUC, FPR, TPR, thresholds.

        Returns
        -------
        Dict[str, Any]
            Keys ``fpr``, ``tpr``, ``thresholds``, ``roc_auc``.
        """
        # Lower similarity → more likely changed, so negate
        fpr, tpr, thresholds = roc_curve(ground_truth, -similarities)
        roc_auc = auc(fpr, tpr)

        result: Dict[str, Any] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "roc_auc": round(roc_auc, 4),
        }

        if save_curve is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})", lw=2)
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve — Temporal Change Detection")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            save_curve.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_curve, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"ROC curve saved to {save_curve}")

        if save_metrics is not None:
            save_metrics.parent.mkdir(parents=True, exist_ok=True)
            with open(save_metrics, "w") as f:
                json.dump(result, f, indent=2)
            print(f"ROC metrics saved to {save_metrics}")

        return result

    # ── Automatic threshold (Youden's J) ──────────────────────────────

    @staticmethod
    def select_threshold(
        fpr: List[float],
        tpr: List[float],
        thresholds: List[float],
        roc_auc: float,
        save_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Select the optimal threshold using Youden's J statistic.

        ``J = TPR - FPR`` is maximised.

        Parameters
        ----------
        fpr : List[float]
            False positive rates.
        tpr : List[float]
            True positive rates.
        thresholds : List[float]
            Threshold values (same length as ``fpr`` / ``tpr``).
        roc_auc : float
            AUC value to include in the output.
        save_path : Optional[Path]
            If provided, saves the result as JSON.

        Returns
        -------
        Dict[str, Any]
            Keys ``selected_threshold``, ``roc_auc``, ``optimal_tpr``,
            ``optimal_fpr``.
        """
        fpr_arr = np.array(fpr)
        tpr_arr = np.array(tpr)
        thr_arr = np.array(thresholds)

        j_stat = tpr_arr - fpr_arr
        best_idx = int(np.argmax(j_stat))

        result = {
            "selected_threshold": round(-float(thr_arr[best_idx]), 6),
            "roc_auc": roc_auc,
            "optimal_tpr": round(float(tpr_arr[best_idx]), 4),
            "optimal_fpr": round(float(fpr_arr[best_idx]), 4),
        }

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Change threshold saved to {save_path}")

        return result


# ═══════════════════════════════════════════════════════════════════════
# PART 6 — STATISTICS
# ═══════════════════════════════════════════════════════════════════════

def print_stats(
    ground_truth: np.ndarray,
    similarities: np.ndarray,
    threshold: float,
    roc_auc: float,
) -> None:
    """Print a summary of change detection statistics.

    Parameters
    ----------
    ground_truth : np.ndarray
        Binary labels (0 = unchanged, 1 = changed).
    similarities : np.ndarray
        Cosine similarity scores.
    threshold : float
        Operating threshold.
    roc_auc : float
        Area under the ROC curve.
    """
    n_total = len(ground_truth)
    n_changed = int(ground_truth.sum())
    n_unchanged = n_total - n_changed

    mean_sim_changed = float(similarities[ground_truth == 1].mean())
    mean_sim_unchanged = float(similarities[ground_truth == 0].mean())

    print("")
    print("=" * 55)
    print("  CHANGE DETECTION SUMMARY")
    print("=" * 55)
    print(f"  Total pairs:                {n_total}")
    print(f"  Changed pairs:              {n_changed}")
    print(f"  Unchanged pairs:            {n_unchanged}")
    print(f"  Average similarity (changed):   {mean_sim_changed:.4f}")
    print(f"  Average similarity (unchanged): {mean_sim_unchanged:.4f}")
    print(f"  Selected threshold:         {threshold:.4f}")
    print(f"  ROC AUC:                    {roc_auc:.4f}")
    print("=" * 55)
