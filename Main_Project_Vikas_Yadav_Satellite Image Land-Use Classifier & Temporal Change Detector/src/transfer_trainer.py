"""Two-phase transfer-learning trainer for ResNet-18.

Phase 1 — classifier head only (3 epochs, ``TRANSFER_LR``).
Phase 2 — fine-tune layer3+layer4 (5 epochs, ``TRANSFER_LR / 10``).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .train import EarlyStopping
from .transfer_model import TransferLearningModel
from .utils import (
    AverageMeter,
    accuracy,
    load_checkpoint,
    plot_curves,
    save_checkpoint,
    setup_logger,
)

logger = setup_logger(__name__)


class TransferTrainer:
    """Two-phase trainer for ``TransferLearningModel``.

    Args:
        model: A ``TransferLearningModel`` instance.
        device: ``"cuda"`` or ``"cpu"``.
        config: Module or namespace with transfer-learning hyperparameters.
    """

    def __init__(
        self,
        model: TransferLearningModel,
        device: torch.device,
        config: Any,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.config = config

        self.criterion = nn.CrossEntropyLoss()
        self.history: List[Dict[str, Any]] = []

    # ── Phase 1 ───────────────────────────────────────────────────────

    def _phase1(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Train only the classifier head with backbone frozen.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
        """
        self.model.freeze_backbone()

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.TRANSFER_LR,
            weight_decay=self.config.WEIGHT_DECAY,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        early_stopping = EarlyStopping(patience=self.config.PATIENCE, mode="min")
        best_val_loss = float("inf")

        num_epochs = self.config.PHASE1_EPOCHS
        logger.info(
            f"\n{'=' * 50}\n  PHASE 1\n"
            f"  Training classifier head\n"
            f"  Frozen backbone\n"
            f"  Epochs: {num_epochs}\n"
            f"{'=' * 50}"
        )

        for epoch in range(1, num_epochs + 1):
            train_metrics = self._run_one_epoch(train_loader, optimizer, phase=1)
            val_metrics = self._validate(val_loader, phase=1)

            scheduler.step(val_metrics["loss"])
            current_lr = optimizer.param_groups[0]["lr"]

            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]

            self._save_phase_checkpoint(
                self.config.PHASE1_BEST_PATH,
                optimizer,
                epoch,
                val_metrics["loss"],
                val_metrics["acc"],
                is_best,
            )

            self._log_epoch(epoch, num_epochs, train_metrics, val_metrics, current_lr, "phase1")

            if early_stopping(val_metrics["loss"]):
                logger.info(f"Phase 1 early stopping triggered after {epoch} epochs.")
                break

    # ── Phase 2 ───────────────────────────────────────────────────────

    def _phase2(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Fine-tune layer3 + layer4 with a 10× lower learning rate.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
        """
        # Load best Phase-1 checkpoint
        load_checkpoint(self.config.PHASE1_BEST_PATH, self.model)

        self.model.unfreeze_last_blocks()

        phase2_lr = self.config.TRANSFER_LR / 10.0

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=phase2_lr,
            weight_decay=self.config.WEIGHT_DECAY,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        early_stopping = EarlyStopping(patience=self.config.PATIENCE, mode="min")
        best_val_loss = float("inf")

        num_epochs = self.config.PHASE2_EPOCHS
        logger.info(
            f"\n{'=' * 50}\n  PHASE 2\n"
            f"  Fine tuning layer3 + layer4\n"
            f"  Learning rate reduced by 10x\n"
            f"  Epochs: {num_epochs}\n"
            f"{'=' * 50}"
        )

        for epoch in range(1, num_epochs + 1):
            train_metrics = self._run_one_epoch(train_loader, optimizer, phase=2)
            val_metrics = self._validate(val_loader, phase=2)

            scheduler.step(val_metrics["loss"])
            current_lr = optimizer.param_groups[0]["lr"]

            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]

            self._save_phase_checkpoint(
                self.config.PHASE2_FINAL_PATH,
                optimizer,
                epoch,
                val_metrics["loss"],
                val_metrics["acc"],
                is_best,
            )

            self._log_epoch(epoch, num_epochs, train_metrics, val_metrics, current_lr, "phase2")

            if early_stopping(val_metrics["loss"]):
                logger.info(f"Phase 2 early stopping triggered after {epoch} epochs.")
                break

    # ── Public entry point ────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> List[Dict[str, Any]]:
        """Run Phase 1 then Phase 2 sequentially.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.

        Returns:
            Combined training history across both phases.
        """
        self.history.clear()

        self._phase1(train_loader, val_loader)
        self._phase2(train_loader, val_loader)

        return self.history

    # ── Internal helpers ──────────────────────────────────────────────

    def _run_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        phase: int,
    ) -> Dict[str, float]:
        """Train for a single epoch.

        Args:
            loader: Training DataLoader.
            optimizer: Optimizer to step.
            phase: 1 or 2.

        Returns:
            Dict with ``loss`` and ``acc``.
        """
        self.model.train()
        losses = AverageMeter()
        accs = AverageMeter()

        pbar = tqdm(loader, desc=f"Train (P{phase})", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acc = accuracy(outputs, labels)[0]
            bs = images.size(0)
            losses.update(loss.item(), bs)
            accs.update(acc, bs)

            pbar.set_postfix(loss=losses.avg, acc=accs.avg)

        return {"loss": losses.avg, "acc": accs.avg}

    @torch.no_grad()
    def _validate(self, loader: DataLoader, phase: int) -> Dict[str, float]:
        """Run a validation epoch.

        Args:
            loader: Validation DataLoader.
            phase: 1 or 2.

        Returns:
            Dict with ``loss`` and ``acc``.
        """
        self.model.eval()
        losses = AverageMeter()
        accs = AverageMeter()

        pbar = tqdm(loader, desc=f" Val  (P{phase})", leave=False)
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

    def _save_phase_checkpoint(
        self,
        path: Path,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
        val_acc: float,
        is_best: bool,
    ) -> None:
        """Save a checkpoint for the current phase.

        Args:
            path: Checkpoint path.
            optimizer: Optimizer to snapshot.
            epoch: Current epoch.
            val_loss: Validation loss.
            val_acc: Validation accuracy.
            is_best: If True, saves a copy with ``.best.pt`` suffix.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        torch.save(state, path)
        if is_best:
            best_path = path.with_name(f"{path.stem}_best{path.suffix}")
            torch.save(state, best_path)

    def _log_epoch(
        self,
        epoch: int,
        total: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
        phase_label: str,
    ) -> None:
        """Record metrics and log to console.

        Args:
            epoch: Current epoch number.
            total: Total epochs in current phase.
            train_metrics: Training loss and acc.
            val_metrics: Validation loss and acc.
            lr: Current learning rate.
            phase_label: ``"phase1"`` or ``"phase2"``.
        """
        record = {
            "epoch": epoch + (0 if phase_label == "phase1" else self.config.PHASE1_EPOCHS),
            "phase": phase_label,
            "train_loss": round(train_metrics["loss"], 4),
            "train_acc": round(train_metrics["acc"], 2),
            "val_loss": round(val_metrics["loss"], 4),
            "val_acc": round(val_metrics["acc"], 2),
            "lr": lr,
        }
        self.history.append(record)

        logger.info(
            f"[{phase_label}] "
            f"Epoch {epoch:>2d}/{total}  "
            f"Train Loss: {train_metrics['loss']:.4f}  "
            f"Train Acc: {train_metrics['acc']:.2f}%  "
            f"Val Loss: {val_metrics['loss']:.4f}  "
            f"Val Acc: {val_metrics['acc']:.2f}%  "
            f"LR: {lr:.2e}"
        )

    # ── Output persistence ────────────────────────────────────────────

    def save_history(self, path: Path) -> None:
        """Save combined training history to a CSV file.

        Args:
            path: Output CSV path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["epoch", "phase", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)
        logger.info(f"Transfer history saved to {path}")

    def save_plots(self, loss_path: Path, acc_path: Path) -> None:
        """Save loss and accuracy curves across both phases.

        Args:
            loss_path: Where to save the loss figure.
            acc_path:  Where to save the accuracy figure.
        """
        plot_curves(self.history, loss_path, acc_path)
        logger.info(f"Transfer loss plot saved to {loss_path}")
        logger.info(f"Transfer accuracy plot saved to {acc_path}")
