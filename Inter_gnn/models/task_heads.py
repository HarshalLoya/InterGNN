"""
Task-specific prediction heads.

Classification head (binary/multi-label) and regression head for
property prediction and drug-target affinity tasks.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    MLP classification head supporting binary and multi-label tasks.

    Architecture: input → FC → GELU → Dropout → FC → output
    Uses sigmoid for multi-label, softmax for multi-class.

    Args:
        input_dim: Input embedding dimension.
        num_tasks: Number of classification tasks/labels.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate.
        multi_label: If True, use sigmoid (independent labels).
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_tasks: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        multi_label: bool = True,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.multi_label = multi_label

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) graph-level or fused embeddings.

        Returns:
            (B, num_tasks) raw logits (always). Apply sigmoid/softmax
            externally for prediction probabilities.
        """
        return self.head(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return calibrated probabilities (sigmoid for multi-label, softmax otherwise)."""
        logits = self.forward(x)
        if self.multi_label or self.num_tasks == 1:
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1)


class RegressionHead(nn.Module):
    """
    MLP regression head for continuous value prediction (e.g., pKd, KIBA).

    Args:
        input_dim: Input embedding dimension.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate.
        output_dim: Output dimension (typically 1 for single-target regression).
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        output_dim: int = 1,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) embeddings.

        Returns:
            (B, output_dim) continuous predictions.
        """
        return self.head(x)


def TaskHead(
    task_type: str = "classification",
    input_dim: int = 256,
    num_tasks: int = 1,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create the appropriate task head.

    Args:
        task_type: 'classification' or 'regression'.
        input_dim: Input dimension.
        num_tasks: Number of output tasks.
        hidden_dim: Hidden dimension.
        dropout: Dropout rate.

    Returns:
        ClassificationHead or RegressionHead instance.
    """
    if task_type == "classification":
        return ClassificationHead(
            input_dim=input_dim,
            num_tasks=num_tasks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            multi_label=kwargs.get("multi_label", num_tasks > 1),
        )
    elif task_type == "regression":
        return RegressionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            output_dim=num_tasks,
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}. Use 'classification' or 'regression'.")
