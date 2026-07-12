"""Spatial leakage experiment: block-split vs random-split comparison.

This module simulates a spatially-aware train/val/test split by grouping
images into deterministic blocks (based on sorted filename order) and
ensuring that images from the same block never appear in more than one
split.  This mirrors the performance drop one would expect when
training on one geographic region and testing on another.

Usage
-----
    python -m src.spatial_split
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Ensure project root is on sys.path when run as script
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from src.dataset import (
    EuroSATDataset,
    create_eurosat_datasets_stratified as create_random_datasets,
    create_spatial_datasets,
)
from src.dataloader import print_dataset_stats
from src.evaluate import Evaluator
from src.transfer_model import TransferLearningModel
from src.transfer_trainer import TransferTrainer
from src.utils import load_checkpoint, set_seed, setup_logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = __import__("logging").getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
#  PART 2 — TRAINING
# ═════════════════════════════════════════════════════════════════════

def train_spatial_model() -> TransferLearningModel:
    """Train ``TransferLearningModel`` using the spatial block split.

    Uses the exact same hyperparameters as the original random-split
    experiment (phase 1 frozen backbone, phase 2 fine-tune layer3+4).

    Returns
    -------
    TransferLearningModel
        Trained model on the spatial split.
    """
    set_seed(config.SEED)

    logger.info("Creating spatial block datasets...")
    datasets = create_spatial_datasets(
        root=config.EUROSAT_ROOT,
        class_names=config.CLASS_NAMES,
        image_size=config.IMAGE_SIZE,
        block_size=config.SPATIAL_BLOCK_SIZE,
        train_ratio=config.TRAIN_SPLIT,
        val_ratio=config.VAL_SPLIT,
        seed=config.SPATIAL_SEED,
    )
    print_dataset_stats(datasets, config.CLASS_NAMES)

    train_loader = torch.utils.data.DataLoader(
        datasets["train"],
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    val_loader = torch.utils.data.DataLoader(
        datasets["val"],
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    model = TransferLearningModel(
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        dropout=config.DROPOUT,
    )

    trainer = TransferTrainer(model, DEVICE, config)
    trainer.fit(train_loader, val_loader)
    trainer.save_history(config.SPATIAL_HISTORY_PATH)
    trainer.save_plots(config.SPATIAL_LOSS_PLOT_PATH, config.SPATIAL_ACCURACY_PLOT_PATH)

    logger.info(f"Spatial model saved to {config.SPATIAL_MODEL_PATH}")
    return model


# ═════════════════════════════════════════════════════════════════════
#  PART 3 — EVALUATION & COMPARISON
# ═════════════════════════════════════════════════════════════════════

def evaluate_model(
    model: TransferLearningModel,
    split_name: str,
    datasets: Dict[str, EuroSATDataset],
) -> Dict:
    """Evaluate a model on the test set of *datasets*.

    Parameters
    ----------
    model : TransferLearningModel
        Trained model.
    split_name : str
        Label for the split (e.g. ``"random"`` or ``"spatial"``).
    datasets : Dict[str, EuroSATDataset]
        Must contain a ``"test"`` key.

    Returns
    -------
    Dict
        Metrics dict from ``Evaluator.compute_metrics`` plus
        ``split_name`` and ``num_samples``.
    """
    test_loader = torch.utils.data.DataLoader(
        datasets["test"],
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    evaluator = Evaluator(model, DEVICE, config.CLASS_NAMES)
    preds, targets, _ = evaluator.predict(test_loader)
    metrics = evaluator.compute_metrics(targets, preds, config.CLASS_NAMES)
    metrics["split_name"] = split_name
    metrics["num_samples"] = len(targets)

    evaluator.print_summary(metrics)
    return metrics


def _random_split_datasets() -> Dict[str, EuroSATDataset]:
    """Build the original random-stratified datasets (for comparison)."""
    return create_random_datasets(
        root=config.EUROSAT_ROOT,
        class_names=config.CLASS_NAMES,
        image_size=config.IMAGE_SIZE,
        train_ratio=config.TRAIN_SPLIT,
        val_ratio=config.VAL_SPLIT,
        seed=config.SEED,
    )


def _load_or_train_spatial_model() -> TransferLearningModel:
    """Load the spatial model from disk, or train it if missing."""
    model = TransferLearningModel(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        dropout=config.DROPOUT,
    )
    if config.SPATIAL_MODEL_PATH.exists():
        logger.info(f"Loading spatial model from {config.SPATIAL_MODEL_PATH}")
        load_checkpoint(config.SPATIAL_MODEL_PATH, model)
    else:
        logger.info("Spatial model checkpoint not found — training from scratch.")
        model = train_spatial_model()
    model = model.to(DEVICE)
    model.eval()
    return model


def _load_random_model() -> TransferLearningModel:
    """Load the random-split model from the existing checkpoint."""
    model = TransferLearningModel(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        dropout=config.DROPOUT,
    )
    load_checkpoint(config.PHASE2_FINAL_PATH, model)
    model = model.to(DEVICE)
    model.eval()
    return model


def compare_splits(
    random_metrics: Dict,
    spatial_metrics: Dict,
    save_csv: Path,
) -> List[Dict]:
    """Build a comparison table and save to CSV.

    Parameters
    ----------
    random_metrics : Dict
        Metrics from the random split evaluation.
    spatial_metrics : Dict
        Metrics from the spatial split evaluation.
    save_csv : Path
        Destination for the CSV table.

    Returns
    -------
    List[Dict]
        Rows suitable for the markdown report.
    """
    rows = [
        {
            "metric": "Accuracy (%)",
            "random_split": random_metrics["accuracy"],
            "spatial_split": spatial_metrics["accuracy"],
            "gap": round(random_metrics["accuracy"] - spatial_metrics["accuracy"], 2),
        },
        {
            "metric": "Macro F1 (%)",
            "random_split": random_metrics["macro_avg"]["f1_score"],
            "spatial_split": spatial_metrics["macro_avg"]["f1_score"],
            "gap": round(
                random_metrics["macro_avg"]["f1_score"]
                - spatial_metrics["macro_avg"]["f1_score"],
                2,
            ),
        },
        {
            "metric": "Macro Precision (%)",
            "random_split": random_metrics["macro_avg"]["precision"],
            "spatial_split": spatial_metrics["macro_avg"]["precision"],
            "gap": round(
                random_metrics["macro_avg"]["precision"]
                - spatial_metrics["macro_avg"]["precision"],
                2,
            ),
        },
        {
            "metric": "Macro Recall (%)",
            "random_split": random_metrics["macro_avg"]["recall"],
            "spatial_split": spatial_metrics["macro_avg"]["recall"],
            "gap": round(
                random_metrics["macro_avg"]["recall"]
                - spatial_metrics["macro_avg"]["recall"],
                2,
            ),
        },
        {
            "metric": "Test Samples",
            "random_split": random_metrics["num_samples"],
            "spatial_split": spatial_metrics["num_samples"],
            "gap": "",
        },
    ]

    save_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(save_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "random_split", "spatial_split", "gap"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Spatial leakage results saved to {save_csv}")

    return rows


# ═════════════════════════════════════════════════════════════════════
#  PART 4 — REPORT
# ═════════════════════════════════════════════════════════════════════

def generate_report(
    comparison_rows: List[Dict],
    random_metrics: Dict,
    spatial_metrics: Dict,
    save_path: Path,
) -> None:
    """Write the spatial leakage markdown report.

    Parameters
    ----------
    comparison_rows : List[Dict]
        Rows from ``compare_splits``.
    random_metrics : Dict
        Random-split metrics.
    spatial_metrics : Dict
        Spatial-split metrics.
    save_path : Path
        Destination ``.md`` file.
    """
    # ── Per-class breakdown ──
    def _per_class_table(metrics: Dict) -> str:
        lines = ["| Class | Precision | Recall | F1 | Support |",
                 "|-------|-----------|--------|----|---------|"]
        for entry in metrics["per_class"]:
            lines.append(
                f"| {entry['class']} | {entry['precision']} | "
                f"{entry['recall']} | {entry['f1_score']} | {entry['support']} |"
            )
        return "\n".join(lines)

    random_per_class = _per_class_table(random_metrics)
    spatial_per_class = _per_class_table(spatial_metrics)

    gap_acc = comparison_rows[0]["gap"]
    gap_f1 = comparison_rows[1]["gap"]

    md = f"""# Spatial Leakage Experiment

## Experiment Setup

| Parameter | Value |
|-----------|-------|
| Dataset | EuroSAT (10 classes, 27 000 images) |
| Base model | ResNet-18 pretrained on ImageNet |
| Training | Phase 1: frozen backbone (3 epochs, LR={config.TRANSFER_LR}) |
| | Phase 2: fine-tune layer3+layer4 (5 epochs, LR={config.TRANSFER_LR/10}) |
| Batch size | {config.BATCH_SIZE} |
| Block size | {config.SPATIAL_BLOCK_SIZE} images per spatial block |
| Random seed (split) | {config.SEED} / {config.SPATIAL_SEED} |
| Split ratios | Train {config.TRAIN_SPLIT*100:.0f}% / Val {config.VAL_SPLIT*100:.0f}% / Test {config.TEST_SPLIT*100:.0f}% |

The *random split* shuffles individual images before partitioning, so
spatially adjacent patches from the same Sentinel-2 scene can appear in
both training and test sets.  The *spatial block split* groups consecutive
images (by sorted filename order) into blocks of {config.SPATIAL_BLOCK_SIZE}
and assigns **entire blocks** to one split only, preventing geographic
leakage.

---

## Quantitative Comparison

| Metric | Random Split | Spatial Split | Gap |
|--------|-------------|---------------|-----|
"""

    for row in comparison_rows:
        md += f"| {row['metric']} | {row['random_split']} | {row['spatial_split']} | {row['gap']} |\n"

    md += f"""

### Per-Class Metrics — Random Split

{random_per_class}

### Per-Class Metrics — Spatial Block Split

{spatial_per_class}

---

## Explanation: Why Random Splitting Can Inflate Performance

Random stratified splitting shuffles individual images before
partitioning.  When satellite images are acquired as contiguous raster
scans, neighbouring patches in the filename order often share:

* **Similar spectral signatures** — adjacent 64×64 patches capture
  nearly identical ground cover.
* **Illumination and atmospheric conditions** — the same sun angle,
  cloud cover, and sensor calibration.
* **Tile-level artefacts** — compression, striping, or stitching
  artefacts are correlated within a scene.

If a model sees one patch of a forest during training and a nearby patch
of the same forest during testing, it can memorise tile-specific
textures rather than learning *general* land-cover features.  This
is known as **spatial autocorrelation** or **geographic leakage**.

The block split breaks this shortcut: whole blocks are held out, so the
model never sees any image from a held-out spatial neighbourhood during
training.  The performance drop (gap = {gap_acc} pp accuracy,
{gap_f1} pp macro F1) quantifies how much of the original score was
due to spatial leakage rather than true generalisation.

---

## Discussion: Model Generalisation

The gap between random-split and spatial-split performance is a lower
bound on the **over-optimism** introduced by naive splitting.  In
operational remote sensing the model must generalise to unseen
geographies, so the spatial-split accuracy is a more honest estimate of
real-world performance.

Closing this gap typically requires:

* **Geographically independent test sets** — held-out regions, scenes,
  or tiles.
* **Domain-adaptation techniques** — adversarial alignment of feature
  distributions across regions.
* **Self-supervised pretraining** — learning invariances from large
  amounts of unlabelled satellite imagery.
* **Augmentation strategies** — spectrally aware augmentations that
  break tile-specific shortcuts.

The ResNet-18 backbone, even with spatial blocking, still achieves
{spatial_metrics['accuracy']:.1f}% accuracy — suggesting that ImageNet
pretraining provides useful generic features — but the {gap_acc:.1f} pp
drop relative to the random-split baseline ({random_metrics['accuracy']:.1f}%)
confirms that simple random splits overstate a model's ability to
generalise across space.
"""

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(md, encoding="utf-8")
    logger.info(f"Spatial leakage report saved to {save_path}")


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the full spatial leakage experiment.

    1. Build datasets (random + spatial block).
    2. Evaluate the existing random-split model.
    3. Train and evaluate a model on the spatial block split.
    4. Compare and generate the report.
    """
    setup_logger()

    logger.info("=" * 55)
    logger.info("  SPATIAL LEAKAGE EXPERIMENT")
    logger.info("=" * 55)

    # ── Step 1: Build random-split datasets ──
    logger.info("\n[1/4] Building random-stratified datasets ...")
    random_datasets = _random_split_datasets()

    # ── Step 2: Evaluate random model ──
    logger.info("\n[2/4] Evaluating random-split model ...")
    random_model = _load_random_model()
    random_metrics = evaluate_model(random_model, "random", random_datasets)

    # ── Step 3: Train & evaluate spatial model ──
    logger.info("\n[3/4] Training/evaluating spatial-block model ...")
    spatial_model = _load_or_train_spatial_model()

    spatial_datasets = create_spatial_datasets(
        root=config.EUROSAT_ROOT,
        class_names=config.CLASS_NAMES,
        image_size=config.IMAGE_SIZE,
        block_size=config.SPATIAL_BLOCK_SIZE,
        train_ratio=config.TRAIN_SPLIT,
        val_ratio=config.VAL_SPLIT,
        seed=config.SPATIAL_SEED,
    )
    spatial_metrics = evaluate_model(spatial_model, "spatial", spatial_datasets)

    # ── Step 4: Compare & report ──
    logger.info("\n[4/4] Generating comparison report ...")
    rows = compare_splits(random_metrics, spatial_metrics, config.SPATIAL_RESULTS_PATH)
    generate_report(rows, random_metrics, spatial_metrics, config.SPATIAL_REPORT_PATH)

    logger.info("\nSpatial leakage experiment complete.")
    logger.info(f"Results CSV : {config.SPATIAL_RESULTS_PATH}")
    logger.info(f"Report MD  : {config.SPATIAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
