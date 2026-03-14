"""Training pipeline: losses, trainer, callbacks, and configuration."""

from inter_gnn.training.losses import TotalLoss
from inter_gnn.training.trainer import InterGNNTrainer
from inter_gnn.training.config import InterGNNConfig

__all__ = [
    "TotalLoss",
    "InterGNNTrainer",
    "InterGNNConfig",
]
