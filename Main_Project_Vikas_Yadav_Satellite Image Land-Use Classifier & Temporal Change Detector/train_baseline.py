"""Official baseline training pipeline for the scratch CNN.

Usage
-----
    python train_baseline.py
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.dataloader import build_dataloaders
from src.evaluate import Evaluator
from src.model import SimpleCNN
from src.train import Trainer


def main() -> None:
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # ── 1. Dataloaders ───────────────────────────────────────────────
    loaders = build_dataloaders(
        root=config.EUROSAT_ROOT,
        class_names=config.CLASS_NAMES,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        seed=config.SEED,
    )

    # ── 2. Model ──────────────────────────────────────────────────────
    model = SimpleCNN(
        in_channels=3,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
    )

    # ── 3. Train (seed automatically set inside Trainer.__init__) ─────
    trainer = Trainer(model, device, config)

    print(f"\n{'=' * 55}")
    print(f"  BASELINE CNN TRAINING")
    print(f"  Epochs: {config.NUM_EPOCHS}  |  Device: {device}")
    print(f"{'=' * 55}\n")

    trainer.fit(loaders["train"], loaders["val"])

    # ── 4. Save training artifacts ────────────────────────────────────
    trainer.save_history(config.HISTORY_PATH)
    trainer.save_plots(config.LOSS_PLOT_PATH, config.ACCURACY_PLOT_PATH)

    print(f"\n{'=' * 55}")
    print(f"  EVALUATION ON TEST SET")
    print(f"{'=' * 55}\n")

    # ── 5. Reload best checkpoint for evaluation ─────────────────────
    trainer.load_best()

    # ── 6. Full test-set evaluation ───────────────────────────────────
    evaluator = Evaluator(model, device, config.CLASS_NAMES)
    preds, targets, _ = evaluator.predict(loaders["test"])
    metrics = evaluator.compute_metrics(targets, preds, config.CLASS_NAMES)
    metrics["num_samples"] = len(targets)

    evaluator.save_results(metrics, targets, preds, config.REPORTS_DIR)
    evaluator.print_summary(metrics)

    # ── 7. Summary ────────────────────────────────────────────────────
    n_train = len(loaders["train"].dataset)
    n_val = len(loaders["val"].dataset)
    n_test = len(loaders["test"].dataset)

    print()
    print(f"  Train samples  : {n_train}")
    print(f"  Val samples    : {n_val}")
    print(f"  Test samples   : {n_test}")
    print(f"  Test accuracy  : {metrics['accuracy']:.2f}%")
    print(f"  Macro F1       : {metrics['macro_avg']['f1_score']:.2f}%")
    print()
    print(f"  Artifacts saved to:")
    print(f"    - {config.HISTORY_PATH}")
    print(f"    - {config.LOSS_PLOT_PATH}")
    print(f"    - {config.ACCURACY_PLOT_PATH}")
    print(f"    - {config.EVAL_METRICS_PATH}")
    print(f"    - {config.CLASSIFICATION_REPORT_PATH}")
    print(f"    - {config.CONFUSION_MATRIX_PATH}")
    print(f"    - {config.BEST_MODEL_PATH}")
    print(f"    - {config.MODELS_DIR / 'baseline_final.pt'}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
