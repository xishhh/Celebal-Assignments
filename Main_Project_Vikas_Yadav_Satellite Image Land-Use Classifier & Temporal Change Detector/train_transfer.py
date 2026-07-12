"""Official transfer learning training pipeline.

Usage
-----
    python train_transfer.py
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.dataloader import build_dataloaders
from src.transfer_model import TransferLearningModel
from src.transfer_trainer import TransferTrainer
from src.utils import set_seed


def main() -> None:
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    set_seed(config.SEED)

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
    model = TransferLearningModel(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
        dropout=config.DROPOUT,
    )

    # ── 3. Train ──────────────────────────────────────────────────────
    trainer = TransferTrainer(model, device, config)

    print(f"\n{'=' * 55}")
    print(f"  TRANSFER LEARNING TRAINING")
    print(f"  Phase 1: {config.PHASE1_EPOCHS} epochs  |  Freeze backbone")
    print(f"  Phase 2: {config.PHASE2_EPOCHS} epochs  |  Fine-tune layer3+layer4")
    print(f"  Device: {device}")
    print(f"{'=' * 55}\n")

    trainer.fit(loaders["train"], loaders["val"])

    # ── 4. Save artifacts ────────────────────────────────────────────
    trainer.save_history(config.TRANSFER_HISTORY_PATH)
    trainer.save_plots(config.TRANSFER_LOSS_PLOT_PATH, config.TRANSFER_ACCURACY_PLOT_PATH)

    # ── 5. Summary ────────────────────────────────────────────────────
    n_train = len(loaders["train"].dataset)
    n_val = len(loaders["val"].dataset)

    print(f"\n{'=' * 55}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 55}")
    print(f"  Train samples  : {n_train}")
    print(f"  Val samples    : {n_val}")
    print(f"  Phase 1 epochs : {config.PHASE1_EPOCHS}")
    print(f"  Phase 2 epochs : {config.PHASE2_EPOCHS}")
    print(f"  Artifacts:")
    print(f"    - {config.TRANSFER_HISTORY_PATH}")
    print(f"    - {config.TRANSFER_LOSS_PLOT_PATH}")
    print(f"    - {config.TRANSFER_ACCURACY_PLOT_PATH}")
    print(f"    - {config.PHASE1_BEST_PATH}")
    print(f"    - {config.PHASE2_FINAL_PATH}")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
