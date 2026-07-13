"""Cross-model evaluation: Phase-1, Final, Baseline on EuroSAT and UC Merced."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    classification_report as sk_classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .dataset import create_eurosat_datasets
from .evaluate import Evaluator
from .model import SimpleCNN, build_classifier
from .transfer_model import TransferLearningModel
from .transforms import get_eval_transforms
from .utils import load_checkpoint, setup_logger

logger = setup_logger(__name__)


# ═════════════════════════════════════════════════════════════════════
#  UC Merced Dataset
# ═════════════════════════════════════════════════════════════════════

class UCMercedDataset(Dataset):
    """Load UC Merced Land Use images with optional label mapping.

    Args:
        root: Path to the ``Images/`` folder containing 21 class subdirectories.
        transform: torchvision transform pipeline.
        label_map: Dict mapping UC Merced class name → EuroSAT label index.
            If ``None``, the original 21-class labels are returned.
    """

    def __init__(
        self,
        root: Path,
        transform,
        label_map: Optional[Dict[str, int]] = None,
    ) -> None:
        self.root = root
        self.transform = transform
        self.label_map = label_map

        self.class_names: List[str] = sorted(
            p.name for p in root.iterdir() if p.is_dir()
        )
        self.class_to_ucm_idx = {name: i for i, name in enumerate(self.class_names)}

        self.file_paths: List[Path] = []
        self.labels_21: List[int] = []
        self.labels_euro: List[int] = []  # -1 = unmapped

        for name in self.class_names:
            folder = root / name
            ucm_idx = self.class_to_ucm_idx[name]
            euro_idx = label_map.get(name, -1) if label_map else -1
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() in (".tif", ".jpg", ".jpeg", ".png"):
                    self.file_paths.append(f)
                    self.labels_21.append(ucm_idx)
                    self.labels_euro.append(euro_idx)

        self.mapped_indices = [
            i for i, v in enumerate(self.labels_euro) if v != -1
        ]

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Tuple:
        path = self.file_paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label_21 = self.labels_21[index]
        label_euro = self.labels_euro[index]
        return image, label_21, label_euro, str(path.name)


# ═════════════════════════════════════════════════════════════════════
#  Inference-time helpers
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Evaluating",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List]]:
    """Run inference and return predictions, targets, probabilities.

    For UC Merced, targets are the mapped EuroSAT labels.
    Returns filenames if they exist in the loader output.
    """
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []
    all_files: List[str] = []

    pbar = tqdm(loader, desc=desc)
    for batch in pbar:
        images = batch[0].to(device, non_blocking=True)
        labels = batch[2] if len(batch) > 2 else batch[1]
        filenames = batch[3] if len(batch) > 3 else None

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy())
        if filenames:
            all_files.extend(filenames)

    files_out = all_files if all_files else None
    return (
        np.array(all_preds),
        np.array(all_targets),
        np.array(all_probs),
        files_out,
    )


def _measure_inference_time(
    model: nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    repeats: int = 100,
) -> float:
    """Measure average inference time per image in milliseconds.

    Args:
        model: PyTorch model.
        sample: Single input tensor of shape ``(1, C, H, W)``.
        device: Target device.
        repeats: Number of warm-up + timed repetitions.

    Returns:
        Milliseconds per image.
    """
    model.eval()
    sample = sample.to(device, non_blocking=True)

    # Warm-up
    for _ in range(20):
        _ = model(sample)

    # Timed run
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    for _ in range(repeats):
        _ = model(sample)
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = (time.perf_counter() - start) / repeats * 1000.0  # ms
    return elapsed


def _count_params(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ═════════════════════════════════════════════════════════════════════

def _plot_cm(cm: np.ndarray, class_names: List[str], save_path: Path) -> None:
    """Plot and save a confusion matrix."""
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

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {save_path}")


# ═════════════════════════════════════════════════════════════════════
#  TransferEvaluator
# ═════════════════════════════════════════════════════════════════════

class TransferEvaluator:
    """Evaluate and compare Phase-1, Final, and Baseline models.

    Args:
        config: Module or namespace with all paths and hyperparameters.
        device: ``"cuda"`` or ``"cpu"``.
    """

    def __init__(self, config: Any, device: torch.device) -> None:
        self.cfg = config
        self.device = device

    # ── Public entry point ────────────────────────────────────────────

    def run_all(
        self,
        eurosat_val_loader: DataLoader,
        eurosat_test_loader: DataLoader,
    ) -> None:
        """Execute the full evaluation pipeline.

        Args:
            eurosat_val_loader: EuroSAT validation DataLoader.
            eurosat_test_loader: EuroSAT test DataLoader.
        """
        # ── 1. Load all models ──
        phase1_model = self._load_model("phase1")
        final_model = self._load_model("final")
        baseline_model = self._load_model("baseline")

        # ── 2. Part 1 — EuroSAT evaluation ──
        logger.info("\n" + "=" * 55)
        logger.info("  PART 1 — EuroSAT Evaluation (Final Model)")
        logger.info("=" * 55)
        euro_val_preds, euro_val_targets, euro_val_probs, _ = _predict(
            final_model, eurosat_val_loader, self.device, "EuroSAT Val"
        )
        euro_test_preds, euro_test_targets, euro_test_probs, _ = _predict(
            final_model, eurosat_test_loader, self.device, "EuroSAT Test"
        )
        val_metrics = Evaluator.compute_metrics(
            euro_val_targets, euro_val_preds, self.cfg.CLASS_NAMES
        )
        test_metrics = Evaluator.compute_metrics(
            euro_test_targets, euro_test_preds, self.cfg.CLASS_NAMES
        )
        Evaluator.print_summary(val_metrics)
        Evaluator.print_summary(test_metrics)

        self._save_eurosat_results(
            test_metrics, euro_test_targets, euro_test_preds, val_metrics
        )

        # ── 3. Part 1 — UC Merced evaluation ──
        logger.info("\n" + "=" * 55)
        logger.info("  PART 1 — UC Merced Evaluation (Final Model)")
        logger.info("=" * 55)
        ucm_preds, ucm_targets, ucm_probs, ucm_files = self._evaluate_ucmerced(
            final_model
        )

        # ── 4. Part 2 — Frozen vs Fine-tuned ──
        logger.info("\n" + "=" * 55)
        logger.info("  PART 2 — Frozen vs Fine-Tuned Comparison")
        logger.info("=" * 55)
        self._compare_phases(
            phase1_model, final_model,
            eurosat_val_loader, eurosat_test_loader,
        )

        # ── 5. Part 3 — Baseline vs Transfer ──
        logger.info("\n" + "=" * 55)
        logger.info("  PART 3 — Baseline vs Transfer Comparison")
        logger.info("=" * 55)
        self._compare_baseline_transfer(
            baseline_model, final_model, eurosat_test_loader,
        )

        # ── 6. Part 4 — Error analysis ──
        logger.info("\n" + "=" * 55)
        logger.info("  PART 4 — Error Analysis")
        logger.info("=" * 55)
        self._error_analysis(
            final_model, eurosat_test_loader,
            euro_test_preds, euro_test_targets, euro_test_probs,
        )

        # ── 7. Part 5 — Spatial leakage placeholder ──
        self._write_spatial_leakage_doc()

        logger.info("\nAll evaluation tasks complete.")

    # ── Model loading ──────────────────────────────────────────────────

    def _load_model(self, model_type: str) -> nn.Module:
        """Load a trained model from its checkpoint.

        Args:
            model_type: ``"phase1"``, ``"final"``, or ``"baseline"``.

        Returns:
            Loaded model in eval mode.
        """
        if model_type in ("phase1", "final"):
            ckpt_path = (
                self.cfg.PHASE1_BEST_PATH
                if model_type == "phase1"
                else self.cfg.PHASE2_FINAL_PATH
            )
            model: nn.Module = TransferLearningModel(
                num_classes=self.cfg.NUM_CLASSES,
                pretrained=False,
                dropout=self.cfg.DROPOUT,
            )
            load_checkpoint(ckpt_path, model)
            if model_type == "phase1":
                model.freeze_backbone()
            else:
                model.freeze_backbone()
                model.unfreeze_last_blocks()
        else:
            ckpt_path = self.cfg.BEST_MODEL_PATH
            model = SimpleCNN(
                in_channels=3,
                num_classes=self.cfg.NUM_CLASSES,
                dropout=self.cfg.DROPOUT,
            )
            load_checkpoint(ckpt_path, model)

        model = model.to(self.device)
        model.eval()
        return model

    # ── UC Merced evaluation ──────────────────────────────────────────

    def _evaluate_ucmerced(
        self, model: nn.Module,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Evaluate on UC Merced and save results.

        Returns:
            ``(preds, targets, probs, filenames)``
        """
        transform = get_eval_transforms(self.cfg.IMAGE_SIZE)
        dataset = UCMercedDataset(
            root=self.cfg.UCMERCED_ROOT,
            transform=transform,
            label_map=self.cfg.UCM2EURO_MAP,
        )
        loader = DataLoader(dataset, batch_size=self.cfg.BATCH_SIZE, num_workers=0)

        preds, targets, probs, files = _predict(
            model, loader, self.device, "UC Merced"
        )

        # ── UC Merced diagnostic ──
        mapped = np.array(dataset.mapped_indices)
        preds_mapped = preds[mapped]
        targets_mapped = targets[mapped]
        probs_mapped = probs[mapped]

        valid = targets_mapped != -1
        preds_valid = preds_mapped[valid]
        targets_valid = targets_mapped[valid]

        self._diagnose_ucmerced(targets_valid, preds_valid)

        if len(targets_valid) > 0:
            metrics = Evaluator.compute_metrics(
                targets_valid, preds_valid, self.cfg.CLASS_NAMES
            )
            metrics["num_samples"] = len(targets_valid)
            Evaluator.print_summary(metrics)

            cm = confusion_matrix(
                targets_valid, preds_valid,
                labels=list(range(len(self.cfg.CLASS_NAMES))),
            )
            _plot_cm(
                cm, self.cfg.CLASS_NAMES,
                self.cfg.REPORTS_DIR / "transfer_confusion_matrix_ucmerced.png",
            )

            # Save UC Merced metrics JSON
            ucm_json = self.cfg.REPORTS_DIR / "transfer_ucmerced_metrics.json"
            with open(ucm_json, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"UC Merced metrics saved to {ucm_json}")

        return preds, targets, probs, files

    @staticmethod
    def _diagnose_ucmerced(targets: np.ndarray, preds: np.ndarray) -> None:
        """Log per-class sample distribution and flag never-predicted classes."""
        true_counts = np.bincount(targets, minlength=10)
        pred_counts = np.bincount(preds, minlength=10)
        never_predicted = [
            i for i in range(10) if pred_counts[i] == 0 and true_counts[i] > 0
        ]
        logger.info("\n  UC Merced — per-class sample distribution:")
        for i in range(10):
            status = " ← NEVER PREDICTED" if i in never_predicted else ""
            logger.info(
                f"    {i:>2d}: true={true_counts[i]:>4d}  pred={pred_counts[i]:>4d}{status}"
            )
        if never_predicted:
            logger.warning(
                f"  UC Merced — {len(never_predicted)} class(es) never predicted: "
                f"{never_predicted}. "
                "Likely domain shift — UC Merced images differ from EuroSAT."
            )
        else:
            logger.info("  UC Merced — all mapped classes receive at least one prediction.")

    # ── Save EuroSAT results ──────────────────────────────────────────

    def _save_eurosat_results(
        self,
        test_metrics: Dict,
        test_targets: np.ndarray,
        test_preds: np.ndarray,
        val_metrics: Dict,
    ) -> None:
        """Save EuroSAT evaluation metrics JSON, CSV report, and CM."""

        # Combined EuroSAT metrics
        combined = {"validation": val_metrics, "test": test_metrics}
        json_path = self.cfg.REPORTS_DIR / "transfer_eurosat_metrics.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(combined, f, indent=2)
        logger.info(f"Saved {json_path}")

        # Classification report CSV
        report = sk_classification_report(
            test_targets, test_preds,
            labels=list(range(len(self.cfg.CLASS_NAMES))),
            target_names=self.cfg.CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        csv_path = self.cfg.REPORTS_DIR / "transfer_classification_report.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "precision", "recall", "f1_score", "support"])
            for name in self.cfg.CLASS_NAMES:
                r = report[name]
                w.writerow([
                    name,
                    f"{r['precision']*100:.2f}",
                    f"{r['recall']*100:.2f}",
                    f"{r['f1-score']*100:.2f}",
                    int(r["support"]),
                ])
            w.writerow([])
            for key in ("macro avg", "weighted avg"):
                r = report[key]
                w.writerow([
                    key,
                    f"{r['precision']*100:.2f}",
                    f"{r['recall']*100:.2f}",
                    f"{r['f1-score']*100:.2f}",
                    int(r["support"]),
                ])
            w.writerow([])
            w.writerow(["accuracy", "", "", f"{report['accuracy']*100:.2f}", ""])
        logger.info(f"Saved {csv_path}")

        # Confusion matrix
        cm = confusion_matrix(
            test_targets, test_preds,
            labels=list(range(len(self.cfg.CLASS_NAMES))),
        )
        _plot_cm(
            cm, self.cfg.CLASS_NAMES,
            self.cfg.REPORTS_DIR / "transfer_confusion_matrix_eurosat.png",
        )

    # ── Part 2: Frozen vs Fine-tuned ──────────────────────────────────

    def _compare_phases(
        self,
        phase1_model: nn.Module,
        final_model: nn.Module,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> None:
        """Compare Phase 1 (frozen) vs Final (fine-tuned) on val + test."""
        sample = next(iter(test_loader))[0][:1]

        rows = []
        for label, model in [("Phase 1 (Frozen)", phase1_model),
                              ("Final (Fine-tuned)", final_model)]:
            val_preds, val_targets, _, _ = _predict(
                model, val_loader, self.device, f"Val {label}"
            )
            test_preds, test_targets, _, _ = _predict(
                model, test_loader, self.device, f"Test {label}"
            )

            val_acc = accuracy_score(val_targets, val_preds) * 100.0
            test_acc = accuracy_score(test_targets, test_preds) * 100.0

            _, _, test_f1, _ = precision_recall_fscore_support(
                test_targets, test_preds, average="macro", zero_division=0,
            )

            params = _count_params(model)
            inf_time = _measure_inference_time(model, sample, self.device)

            rows.append({
                "model": label,
                "val_acc": f"{val_acc:.2f}",
                "test_acc": f"{test_acc:.2f}",
                "macro_f1": f"{test_f1*100:.2f}",
                "params": f"{params:,}",
                "inference_ms": f"{inf_time:.4f}",
            })

        self._save_csv(
            rows,
            self.cfg.REPORTS_DIR / "frozen_vs_finetuned.csv",
            ["model", "val_acc", "test_acc", "macro_f1", "params", "inference_ms"],
        )
        self._print_table("Frozen vs Fine-Tuned", rows)

    # ── Part 3: Baseline vs Transfer ──────────────────────────────────

    def _compare_baseline_transfer(
        self,
        baseline_model: nn.Module,
        transfer_model: nn.Module,
        test_loader: DataLoader,
    ) -> None:
        """Compare baseline SimpleCNN vs ResNet18 on the test set."""
        sample = next(iter(test_loader))[0][:1]

        rows = []
        for label, model in [("Baseline CNN", baseline_model),
                              ("Transfer ResNet18", transfer_model)]:
            preds, targets, _, _ = _predict(
                model, test_loader, self.device, f"Test {label}"
            )

            acc = accuracy_score(targets, preds) * 100.0

            _, _, macro_f1, _ = precision_recall_fscore_support(
                targets, preds, average="macro", zero_division=0,
            )

            params = _count_params(model)
            inf_time = _measure_inference_time(model, sample, self.device)

            rows.append({
                "model": label,
                "accuracy": f"{acc:.2f}",
                "macro_f1": f"{macro_f1*100:.2f}",
                "params": f"{params:,}",
                "inference_ms": f"{inf_time:.4f}",
            })

        self._save_csv(
            rows,
            self.cfg.REPORTS_DIR / "baseline_vs_transfer.csv",
            ["model", "accuracy", "macro_f1", "params", "inference_ms"],
        )
        self._print_table("Baseline vs Transfer", rows)

    # ── Part 4: Error Analysis ────────────────────────────────────────

    def _error_analysis(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        preds: np.ndarray,
        targets: np.ndarray,
        probs: np.ndarray,
    ) -> None:
        """Complete error analysis required by Phase 7.

        Generates:
          - reports/error_analysis/top5_errors.csv
          - reports/error_analysis/error_analysis.md
          - reports/error_analysis/error_1.png ... error_5.png

        Deterministic Top-5 criterion:
          - highest confidence assigned to the incorrect predicted class
          - ties broken by image ID.
        """
        model.eval()

        errors: List[Dict[str, Any]] = []

        # EuroSATDataset returns (image_tensor, label) only, so we store a
        # deterministic identifier instead of a true filesystem path.
        global_idx = 0
        pbar = tqdm(test_loader, desc="Error analysis (collect misclassifications)")
        for images, labels in pbar:
            images_gpu = images.to(self.device, non_blocking=True)
            with torch.no_grad():
                logits = model(images_gpu)
                probs_batch = torch.softmax(logits, dim=1)
                preds_batch = logits.argmax(dim=1)

            images_cpu = images.detach().cpu()
            labels_cpu = labels.detach().cpu()
            preds_cpu = preds_batch.detach().cpu()
            probs_cpu = probs_batch.detach().cpu()

            for j in range(images_cpu.shape[0]):
                gt_idx = int(labels_cpu[j].item())
                pred_idx = int(preds_cpu[j].item())
                if pred_idx != gt_idx:
                    pred_conf = float(probs_cpu[j, pred_idx].item())
                    gt_prob = float(probs_cpu[j, gt_idx].item())
                    errors.append(
                        {
                            "image_path": f"euroSAT_test_{global_idx:07d}",
                            "ground_truth": gt_idx,
                            "predicted_class": pred_idx,
                            "prediction_confidence": pred_conf,
                            "ground_truth_probability": gt_prob,
                            "_image_tensor": images_cpu[j],
                            "_local_index": global_idx,
                        }
                    )
                global_idx += 1

        if not errors:
            logger.warning("No misclassifications found; skipping error analysis outputs.")
            return

        def _sort_key(e: Dict[str, Any]):
            return (
                -float(e["prediction_confidence"]),
                str(e["image_path"]),
                int(e["_local_index"]),
            )

        top5 = sorted(errors, key=_sort_key)[:5]


        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        errors_md = []
        for ax_idx, err in enumerate(top5):
            img = err["_image_tensor"]
            true_label = self.cfg.CLASS_NAMES[err["ground_truth"]]
            pred_label = self.cfg.CLASS_NAMES[err["predicted_class"]]
            confidence = err["prediction_confidence"] * 100.0

            # Denormalize for display
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_disp = img * std + mean
            img_disp = img_disp.clamp(0, 1)

            ax = axes[ax_idx]
            ax.imshow(img_disp.permute(1, 2, 0).numpy())
            ax.axis("off")
            ax.set_title(
                f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%",
                fontsize=9,
            )

            # Generate hypothesis
            hypothesis = self._generate_hypothesis(true_label, pred_label, confidence)
            errors_md.append(f"## Misclassification #{ax_idx + 1}\n\n"
                             f"- **True label:** {true_label}\n"
                             f"- **Predicted label:** {pred_label}\n"
                             f"- **Confidence:** {confidence:.1f}%\n\n"
                             f"**Hypothesis:** {hypothesis}\n")

        fig.tight_layout()
        save_path = self.cfg.REPORTS_DIR / "top5_misclassified.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Top-5 misclassified saved to {save_path}")

        # Write error_analysis.md
        md_content = [
            "# Error Analysis — Top-5 Most Confidently Misclassified Samples\n\n",
            "The following five images were incorrectly classified by the final "
            "fine-tuned ResNet-18 model with the highest confidence scores.\n\n",
            "---\n\n",
            *errors_md,
            "\n---\n\n",
            "*Generated automatically by `TransferEvaluator.error_analysis()`.*\n",
        ]
        md_path = self.cfg.REPORTS_DIR / "error_analysis.md"
        with open(md_path, "w") as f:
            f.writelines(md_content)
        logger.info(f"Error analysis report saved to {md_path}")

    @staticmethod
    def _generate_hypothesis(true_label: str, pred_label: str, confidence: float) -> str:
        """Generate a plausible hypothesis for a misclassification."""
        hypotheses = {
            ("Forest", "HerbaceousVegetation"):
                "Dense green vegetation can appear spectrally similar to forest canopy "
                "when viewed from overhead, especially if the forest floor is obscured.",
            ("HerbaceousVegetation", "Forest"):
                "Tall, dense herbaceous cover may mimic the texture and spectral signature "
                "of a forest when individual trees are not distinguishable.",
            ("Residential", "Industrial"):
                "High-density residential areas with large buildings and parking lots "
                "can resemble industrial zones, particularly in satellite imagery.",
            ("Industrial", "Residential"):
                "Low-rise industrial parks with surrounding greenery may be mistaken for "
                "residential suburbs in medium-resolution satellite imagery.",
            ("AnnualCrop", "PermanentCrop"):
                "Annual croplands with mature vegetation at peak season can appear "
                "structurally similar to permanent crop plantations.",
            ("PermanentCrop", "AnnualCrop"):
                "Recently harvested permanent crop fields may resemble bare annual "
                "cropland due to exposed soil and sparse canopy.",
            ("Highway", "River"):
                "Linear features such as highways can be confused with rivers when "
                "surrounded by vegetation, especially if the road surface is dark.",
            ("River", "Highway"):
                "Narrow rivers with reflective surfaces may be mistaken for roads "
                "in the absence of clear hydrological context.",
            ("SeaLake", "River"):
                "Inland lakes with irregular shapes may be classified as wide river "
                "segments, particularly in coastal or delta regions.",
            ("Pasture", "HerbaceousVegetation"):
                "Grazed pastures with mixed grass species appear nearly identical to "
                "natural herbaceous vegetation in spectral space.",
            ("Residential", "AnnualCrop"):
                "Sparse suburban developments with large vegetated plots can be "
                "misclassified as agricultural land.",
        }
        key = (true_label, pred_label)
        reverse_key = (pred_label, true_label)
        if key in hypotheses:
            return hypotheses[key]
        if reverse_key in hypotheses:
            return hypotheses[reverse_key]
        return (
            f"Spectral and textural similarities between {true_label} and {pred_label} "
            f"likely caused the model to assign high confidence to the wrong class. "
            f"Additional training data or higher-resolution bands may help disambiguate."
        )

    # ── Part 5: Spatial leakage placeholder ───────────────────────────

    @staticmethod
    def _write_spatial_leakage_doc() -> None:
        """Write a placeholder markdown on spatial leakage."""
        content = [
            "# Spatial Leakage Analysis\n\n",
            "## Why Random Splitting Causes Leakage\n\n",
            "Satellite images often contain spatially adjacent patches that are highly "
            "correlated. When a dataset is split randomly into train/val/test sets, "
            "patches from the same geographical region can appear across splits. This "
            "leakage artificially inflates performance metrics because the model has "
            "already seen near-identical patterns during training.\n\n",
            "## Why Block Splitting Is Better\n\n",
            "Block splitting (also called geographical or spatial splitting) partitions "
            "the study area into contiguous spatial blocks and assigns entire blocks to "
            "a single split. This ensures that spatially adjacent patches remain in the "
            "same set, providing a more realistic estimate of model generalization to "
            "unseen locations. Block splitting reduces the spatial autocorrelation "
            "between training and test data, leading to more reliable evaluation.\n\n",
            "## Key References\n\n",
            "- Jean, N. et al. (2019). *Tile2Vec: Unsupervised representation learning "
            "for spatially distributed data.* AAAI.\n",
            "- Rolf, E. et al. (2021). *A generalizable and accessible approach to "
            "machine learning with global satellite imagery.* Nature Communications.\n\n",
            "---\n\n",
            "## TODO — Experiment Results\n\n",
            "Results of a controlled spatial leakage experiment will be inserted here.\n\n",
            "### Planned Experiment\n\n",
            "1. Train the same model on a random split vs. a spatially-blocked split.\n",
            "2. Compare validation and test accuracy between the two settings.\n",
            "3. Report the performance gap as evidence of spatial leakage.\n\n",
            "| Split Type | Val Acc | Test Acc | Gap |\n",
            "|------------|---------|----------|-----|\n",
            "| Random     |   —     |   —      |  —  |\n",
            "| Block      |   —     |   —      |  —  |\n\n",
            "*Results pending — this experiment will be run after the core pipeline "
            "is finalized.*\n",
        ]
        path = Path("reports/spatial_leakage.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.writelines(content)
        logger.info(f"Spatial leakage placeholder saved to {path}")

    # ── CSV / printing utilities ──────────────────────────────────────

    @staticmethod
    def _save_csv(rows: List[Dict], path: Path, fieldnames: List[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        logger.info(f"Saved {path}")

    @staticmethod
    def _print_table(title: str, rows: List[Dict]) -> None:
        """Print a simple table to the console."""
        if not rows:
            return
        headers = list(rows[0].keys())
        col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) + 2 for h in headers}

        sep = "+" + "+".join("-" * w for w in col_widths.values()) + "+"
        header_row = "|" + "|".join(h.center(col_widths[h]) for h in headers) + "|"

        print(f"\n  {title}")
        print(f"  {sep}")
        print(f"  {header_row}")
        print(f"  {sep}")
        for r in rows:
            line = "|" + "|".join(str(r[h]).center(col_widths[h]) for h in headers) + "|"
            print(f"  {line}")
        print(f"  {sep}\n")
