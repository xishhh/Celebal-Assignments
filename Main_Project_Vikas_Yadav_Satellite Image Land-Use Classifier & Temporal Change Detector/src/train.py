"""Production-quality training loop with checkpointing, logging, and early stopping."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import (
    AverageMeter,
    accuracy,
    load_checkpoint,
    plot_curves,
    save_checkpoint,
    set_seed,
    setup_logger,
)

logger = setup_logger(__name__)


class EarlyStopping:
    """Stop training when a monitored metric has stopped improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        mode: ``"min"`` for loss (lower is better) or ``"max"`` for accuracy.
        min_delta: Minimum change to qualify as improvement.
    """

    def __init__(
        self,
        patience: int = 7,
        mode: str = "min",
        min_delta: float = 1e-4,
    ) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = float("inf") if mode == "min" else -float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, current: float) -> bool:
        """Return *True* if training should stop."""
        if self.mode == "min":
            improved = current < self.best - self.min_delta
        else:
            improved = current > self.best + self.min_delta

        if improved:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class Trainer:
    """End-to-end trainer with validation, checkpointing, LR scheduling, and early stopping.

    Args:
        model: PyTorch model.
        device: ``"cuda"`` or ``"cpu"``.
        config: Module or namespace with training hyperparameters.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Any,
    ) -> None:
        set_seed(config.SEED)
        self.model = model.to(device)
        self.device = device
        self.config = config

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=config.PATIENCE // 2,
        )
        self.early_stopping = EarlyStopping(
            patience=config.PATIENCE,
            mode="min",
        )

        self.history: List[Dict[str, float]] = []
        self.best_val_loss = float("inf")

    def _train_one_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run a single training epoch.

        Args:
            loader: Training DataLoader.

        Returns:
            Dict with ``loss`` and ``acc`` (top-1 accuracy %).
        """
        self.model.train()
        losses = AverageMeter()
        accs = AverageMeter()

        pbar = tqdm(loader, desc="Train", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            acc = accuracy(outputs, labels)[0]
            bs = images.size(0)
            losses.update(loss.item(), bs)
            accs.update(acc, bs)

            pbar.set_postfix(loss=losses.avg, acc=accs.avg)

        return {"loss": losses.avg, "acc": accs.avg}

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        """Run a validation epoch.

        Args:
            loader: Validation DataLoader.

        Returns:
            Dict with ``loss`` and ``acc`` (top-1 accuracy %).
        """
        self.model.eval()
        losses = AverageMeter()
        accs = AverageMeter()

        pbar = tqdm(loader, desc=" Val ", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            acc = accuracy(outputs, labels)[0]
            bs = images.size(0)
            losses.update(loss.item(), bs)
            accs.update(acc, bs)

            pbar.set_postfix(loss=losses.avg, acc=accs.avg)

        return {"loss": losses.avg, "acc": accs.avg}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Run the full training loop.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs: Override number of epochs (defaults to ``config.NUM_EPOCHS``).

        Returns:
            History list of per-epoch metric dicts.
        """
        epochs = epochs or self.config.NUM_EPOCHS
        logger.info(f"Training for {epochs} epochs | Device: {self.device}")

        for epoch in range(1, epochs + 1):
            train_metrics = self._train_one_epoch(train_loader)
            val_metrics = self._validate(val_loader)

            self.scheduler.step(val_metrics["loss"])

            epoch_log = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
            }
            self.history.append(epoch_log)

            logger.info(
                f"Epoch {epoch:>3d}/{epochs}  "
                f"Train Loss: {train_metrics['loss']:.4f}  "
                f"Train Acc: {train_metrics['acc']:.2f}%  "
                f"Val Loss: {val_metrics['loss']:.4f}  "
                f"Val Acc: {val_metrics['acc']:.2f}%"
            )

            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]

            epoch_path = self.config.MODELS_DIR / f"baseline_epoch_{epoch:02d}.pt"
            save_checkpoint(
                path=epoch_path,
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                val_loss=val_metrics["loss"],
                val_acc=val_metrics["acc"],
                is_best=False,
            )
            if is_best:
                save_checkpoint(
                    path=self.config.BEST_MODEL_PATH,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    val_loss=val_metrics["loss"],
                    val_acc=val_metrics["acc"],
                    is_best=False,
                )

            if self.early_stopping(val_metrics["loss"]):
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

        # Save final checkpoint
        if self.history:
            final_epoch = self.history[-1]["epoch"]
            final_path = self.config.MODELS_DIR / "baseline_final.pt"
            save_checkpoint(
                path=final_path,
                model=self.model,
                optimizer=self.optimizer,
                epoch=final_epoch,
                val_loss=self.history[-1]["val_loss"],
                val_acc=self.history[-1]["val_acc"],
                is_best=False,
            )
            logger.info(f"Final checkpoint saved to {final_path}")

        return self.history

    def save_history(self, path: Path) -> None:
        """Save training history to a CSV file.

        Args:
            path: Output CSV path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            writer.writeheader()
            writer.writerows(self.history)
        logger.info(f"Training history saved to {path}")

    def save_plots(self, loss_path: Path, acc_path: Path) -> None:
        """Save loss and accuracy curves.

        Args:
            loss_path: Where to save the loss figure.
            acc_path:  Where to save the accuracy figure.
        """
        plot_curves(self.history, loss_path, acc_path)
        logger.info(f"Loss plot saved to {loss_path}")
        logger.info(f"Accuracy plot saved to {acc_path}")

    def load_best(self, path: Optional[Path] = None) -> None:
        """Load the best model checkpoint.

        Args:
            path: Checkpoint path (defaults to ``config.BEST_MODEL_PATH``).
        """
        ckpt_path = path or self.config.BEST_MODEL_PATH
        load_checkpoint(ckpt_path, self.model)
        logger.info(f"Best model loaded from {ckpt_path}")
