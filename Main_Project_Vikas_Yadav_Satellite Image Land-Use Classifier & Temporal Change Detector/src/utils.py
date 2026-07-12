"""Shared utilities: seeding, logging, checkpointing, metrics, plotting."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name: str = "satellite") -> logging.Logger:
    """Configure and return a logger with stdout handler.

    Args:
        name: Logger name.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


class AverageMeter:
    """Tracks running average and current value of a metric."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> List[float]:
    """Compute top-k accuracy.

    Args:
        output: Model logits of shape (N, C).
        target: Ground-truth labels of shape (N,).
        topk: Tuple of k values to compute.

    Returns:
        List of accuracy percentages corresponding to each k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res: List[float] = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True).item()
        res.append(correct_k / batch_size * 100.0)
    return res


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_acc: float,
    is_best: bool = False,
) -> None:
    """Save a training checkpoint to disk.

    Args:
        path: Destination file path.
        model: Model whose state_dict will be saved.
        optimizer: Optimizer state_dict.
        epoch: Current epoch number.
        val_loss: Validation loss value.
        val_acc: Validation accuracy value.
        is_best: If True, saves a separate ``best_model.pt`` copy.
    """
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = path.with_name(f"{path.stem}_best{path.suffix}")
        torch.save(state, best_path)


def load_checkpoint(
    path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    """Load a checkpoint and restore model (and optional optimizer) state.

    Args:
        path: Checkpoint file path.
        model: Model instance to load weights into.
        optimizer: Optional optimizer to restore state.

    Returns:
        The checkpoint dictionary (epoch, metrics, etc.).
    """
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state


def plot_curves(
    history: List[Dict[str, float]],
    loss_path: Path,
    acc_path: Path,
) -> None:
    """Plot and save training/validation loss and accuracy curves.

    Args:
        history: List of dicts with keys ``train_loss``, ``val_loss``,
                 ``train_acc``, ``val_acc``.
        loss_path: Where to save the loss figure.
        acc_path:  Where to save the accuracy figure.
    """
    epochs = range(1, len(history) + 1)
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Train Loss")
    ax.plot(epochs, val_loss, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True)
    loss_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_acc, label="Train Acc")
    ax.plot(epochs, val_acc, label="Val Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training & Validation Accuracy")
    ax.legend()
    ax.grid(True)
    acc_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
