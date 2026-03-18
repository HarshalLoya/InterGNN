"""
Training callbacks: checkpointing, early stopping, explainer monitoring.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping based on validation metric.

    Stops training when monitored metric hasn't improved for `patience` epochs.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def step(self, value: float) -> bool:
        """Returns True if training should stop."""
        improved = (
            (value < self.best_value - self.min_delta)
            if self.mode == "min"
            else (value > self.best_value + self.min_delta)
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


class ModelCheckpoint:
    """
    Save model checkpoints when validation metric improves.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        prefix: str = "model",
        mode: str = "min",
        save_top_k: int = 3,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.prefix = prefix
        self.mode = mode
        self.save_top_k = save_top_k
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.saved_checkpoints: List[str] = []

        os.makedirs(checkpoint_dir, exist_ok=True)

    def step(self, value: float, model: nn.Module, epoch: int) -> Optional[str]:
        """Save checkpoint if metric improved. Returns path or None."""
        improved = (
            (value < self.best_value) if self.mode == "min" else (value > self.best_value)
        )

        if improved:
            self.best_value = value
            path = os.path.join(
                self.checkpoint_dir,
                f"{self.prefix}_epoch{epoch}_val{value:.4f}.pt",
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metric": value,
            }, path)

            self.saved_checkpoints.append(path)
            logger.info(f"Checkpoint saved: {path}")

            # Clean old checkpoints
            while len(self.saved_checkpoints) > self.save_top_k:
                old = self.saved_checkpoints.pop(0)
                if os.path.exists(old):
                    os.remove(old)

            return path
        return None


class ExplainerMonitor:
    """
    Periodically run explainer diagnostics during training to track
    explanation quality over training epochs.
    """

    def __init__(self, eval_interval: int = 5, num_samples: int = 10):
        self.eval_interval = eval_interval
        self.num_samples = num_samples
        self.history: List[dict] = []

    def should_evaluate(self, epoch: int) -> bool:
        return epoch % self.eval_interval == 0

    def evaluate(self, model, data_list, epoch: int) -> dict:
        """Run lightweight explanation diagnostics."""
        model.eval()
        device = next(model.parameters()).device

        importances = []
        for data in data_list[: self.num_samples]:
            data = data.to(device)
            batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
            try:
                imp = model.get_node_importance(data.x, data.edge_index, data.edge_attr, batch)
                importances.append(imp.detach().cpu())
            except Exception:
                continue

        if not importances:
            return {"epoch": epoch, "status": "no_data"}

        all_imp = torch.cat(importances)
        result = {
            "epoch": epoch,
            "mean_importance": float(all_imp.mean()),
            "std_importance": float(all_imp.std()),
            "max_importance": float(all_imp.max()),
            "sparsity": float((all_imp < 0.1).float().mean()),
        }
        self.history.append(result)
        return result


class CallbackManager:
    """Manages a list of callbacks executed during training."""

    def __init__(self):
        self.early_stopping: Optional[EarlyStopping] = None
        self.checkpoint: Optional[ModelCheckpoint] = None
        self.explainer_monitor: Optional[ExplainerMonitor] = None

    def setup(
        self,
        early_stopping_patience: int = 15,
        checkpoint_dir: str = "checkpoints",
        prefix: str = "model",
        monitor_interval: int = 5,
    ):
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        self.checkpoint = ModelCheckpoint(checkpoint_dir, prefix)
        self.explainer_monitor = ExplainerMonitor(eval_interval=monitor_interval)
