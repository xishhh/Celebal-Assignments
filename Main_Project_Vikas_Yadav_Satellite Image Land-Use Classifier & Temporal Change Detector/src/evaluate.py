"""Evaluation: metrics, confusion matrix, classification report for a trained model."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    classification_report as sk_classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .utils import setup_logger

logger = setup_logger(__name__)


class Evaluator:
    """Evaluate a trained classifier and persist results.

    Args:
        model: A trained PyTorch model.
        device: ``"cuda"`` or ``"cpu"``.
        class_names: Ordered list of class label strings matching the model's
            output indices.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: List[str],
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference over a DataLoader.

        Args:
            loader: A DataLoader yielding ``(images, labels)``.

        Returns:
            Tuple of ``(predictions, targets, probabilities)`` as NumPy arrays.
        """
        self.model.eval()

        all_preds: List[int] = []
        all_targets: List[int] = []
        all_probs: List[np.ndarray] = []

        pbar = tqdm(loader, desc=" Evaluating")
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)

            logits = self.model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())

        return (
            np.array(all_preds),
            np.array(all_targets),
            np.array(all_probs),
        )

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
    ) -> Dict:
        """Compute accuracy, per-class and macro precision/recall/F1.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            class_names: Class label strings.

        Returns:
            Nested dictionary with keys ``accuracy``, ``per_class``, and
            ``macro_avg``.
        """
        overall_acc = accuracy_score(y_true, y_pred) * 100.0

        n_classes = len(class_names)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0
        )

        per_class: List[Dict] = []
        for i, name in enumerate(class_names):
            per_class.append(
                {
                    "class": name,
                    "precision": round(precision[i] * 100, 2),
                    "recall": round(recall[i] * 100, 2),
                    "f1_score": round(f1[i] * 100, 2),
                    "support": int(support[i]),
                }
            )

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0
        )

        return {
            "accuracy": round(overall_acc, 2),
            "per_class": per_class,
            "macro_avg": {
                "precision": round(macro_p * 100, 2),
                "recall": round(macro_r * 100, 2),
                "f1_score": round(macro_f1 * 100, 2),
            },
        }

    def evaluate(self, loader: DataLoader) -> Dict:
        """Convenience: ``predict`` + ``compute_metrics`` in one call.

        Args:
            loader: DataLoader for the evaluation set.

        Returns:
            Metrics dictionary (see ``compute_metrics``).
        """
        preds, targets, probs = self.predict(loader)
        metrics = self.compute_metrics(targets, preds, self.class_names)
        metrics["num_samples"] = len(targets)
        return metrics

    def save_results(
        self,
        metrics: Dict,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_dir: Path,
    ) -> None:
        """Save metrics JSON, classification report CSV, and confusion matrix PNG.

        Args:
            metrics: Dictionary returned by ``compute_metrics``.
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            output_dir: Directory where results will be written.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── JSON ──────────────────────────────────────────────────────
        metrics_path = output_dir / "evaluation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")

        # ── CSV classification report ─────────────────────────────────
        report = sk_classification_report(
            y_true,
            y_pred,
            labels=list(range(len(self.class_names))),
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )
        csv_path = output_dir / "classification_report.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "precision", "recall", "f1_score", "support"])
            for name in self.class_names:
                row = report[name]
                writer.writerow(
                    [
                        name,
                        f"{row['precision']*100:.2f}",
                        f"{row['recall']*100:.2f}",
                        f"{row['f1-score']*100:.2f}",
                        int(row["support"]),
                    ]
                )
            writer.writerow([])
            for key in ("macro avg", "weighted avg"):
                row = report[key]
                writer.writerow(
                    [
                        key,
                        f"{row['precision']*100:.2f}",
                        f"{row['recall']*100:.2f}",
                        f"{row['f1-score']*100:.2f}",
                        int(row["support"]),
                    ]
                )
            writer.writerow([])
            writer.writerow(["accuracy", "", "", f"{report['accuracy']*100:.2f}", ""])
        logger.info(f"Classification report saved to {csv_path}")

        # ── Confusion matrix plot ─────────────────────────────────────
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(self.class_names))))
        self._plot_confusion_matrix(cm, self.class_names, output_dir / "confusion_matrix.png")

    # ── Private helpers ───────────────────────────────────────────────

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        save_path: Path,
    ) -> None:
        """Plot a clean confusion matrix with matplotlib."""
        n = len(class_names)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046)
        cbar.set_label("Count")

        ax.set(
            xticks=np.arange(n),
            yticks=np.arange(n),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted Label",
            ylabel="True Label",
            title="Confusion Matrix",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Confusion matrix saved to {save_path}")

    # ── Summary ───────────────────────────────────────────────────────

    @staticmethod
    def print_summary(metrics: Dict) -> None:
        """Print a concise evaluation summary to the console.

        Args:
            metrics: Dictionary returned by ``compute_metrics``.
        """
        print("\n" + "=" * 55)
        print("  EVALUATION SUMMARY")
        print("=" * 55)
        print(f"  Overall Accuracy        {metrics['accuracy']:.2f}%")
        print(f"  Macro Avg Precision     {metrics['macro_avg']['precision']:.2f}%")
        print(f"  Macro Avg Recall        {metrics['macro_avg']['recall']:.2f}%")
        print(f"  Macro Avg F1 Score      {metrics['macro_avg']['f1_score']:.2f}%")
        print("-" * 55)
        print(f"  {'Class':<25s}  {'Prec':>7s}  {'Rec':>7s}  {'F1':>7s}")
        print("-" * 55)
        for entry in metrics["per_class"]:
            print(
                f"  {entry['class']:<25s}  {entry['precision']:>6.2f}%  "
                f"{entry['recall']:>6.2f}%  {entry['f1_score']:>6.2f}%"
            )
        print("=" * 55)
