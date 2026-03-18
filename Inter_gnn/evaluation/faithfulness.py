"""
Explanation faithfulness metrics: deletion AUC and insertion AUC.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data


def deletion_auc(
    model: nn.Module,
    data: Data,
    node_importance: torch.Tensor,
    num_steps: int = 10,
) -> float:
    """
    Deletion AUC: progressively remove most important nodes.

    Starting from the full graph, iteratively mask the most important
    atoms and measure prediction degradation. Lower AUC = more faithful.

    Args:
        model: Trained InterGNN model.
        data: Input graph Data object.
        node_importance: (N,) importance scores per atom.
        num_steps: Number of deletion steps.

    Returns:
        Area under the deletion curve (lower = better explanation).
    """
    model.eval()
    device = data.x.device
    n = data.x.shape[0]
    batch = torch.zeros(n, dtype=torch.long, device=device)

    # Get baseline prediction
    with torch.no_grad():
        base_out = model(data.x, data.edge_index, data.edge_attr, batch)
        base_pred = base_out["prediction"].cpu().numpy().flatten()

    # Sort atoms by importance (descending)
    sorted_idx = torch.argsort(node_importance, descending=True)

    # Progressive deletion
    predictions = [base_pred]
    fractions = [0.0]
    step_size = max(1, n // num_steps)

    x_masked = data.x.clone()
    for step in range(1, num_steps + 1):
        # Mask atoms in this step
        start = (step - 1) * step_size
        end = min(step * step_size, n)
        atoms_to_mask = sorted_idx[start:end]
        x_masked[atoms_to_mask] = 0.0

        with torch.no_grad():
            out = model(x_masked, data.edge_index, data.edge_attr, batch)
            pred = out["prediction"].cpu().numpy().flatten()

        predictions.append(pred)
        fractions.append(end / n)

    # Compute AUC via trapezoidal rule
    pred_changes = [np.mean(np.abs(p - base_pred)) for p in predictions]
    auc = float(np.trapz(pred_changes, fractions))

    return auc


def insertion_auc(
    model: nn.Module,
    data: Data,
    node_importance: torch.Tensor,
    num_steps: int = 10,
) -> float:
    """
    Insertion AUC: progressively insert most important nodes.

    Starting from an empty graph, iteratively unmask the most important
    atoms and measure prediction recovery. Higher AUC = more faithful.

    Returns:
        Area under the insertion curve (higher = better explanation).
    """
    model.eval()
    device = data.x.device
    n = data.x.shape[0]
    batch = torch.zeros(n, dtype=torch.long, device=device)

    # Get target prediction
    with torch.no_grad():
        target_out = model(data.x, data.edge_index, data.edge_attr, batch)
        target_pred = target_out["prediction"].cpu().numpy().flatten()

    # Sort by importance (descending)
    sorted_idx = torch.argsort(node_importance, descending=True)

    # Start from all-zero features
    x_inserted = torch.zeros_like(data.x)
    step_size = max(1, n // num_steps)

    predictions = [np.zeros_like(target_pred)]
    fractions = [0.0]

    for step in range(1, num_steps + 1):
        start = (step - 1) * step_size
        end = min(step * step_size, n)
        atoms_to_insert = sorted_idx[start:end]
        x_inserted[atoms_to_insert] = data.x[atoms_to_insert]

        with torch.no_grad():
            out = model(x_inserted, data.edge_index, data.edge_attr, batch)
            pred = out["prediction"].cpu().numpy().flatten()

        predictions.append(pred)
        fractions.append(end / n)

    # Compute similarity to target prediction
    similarities = [
        1.0 - np.mean(np.abs(p - target_pred)) for p in predictions
    ]
    auc = float(np.trapz(similarities, fractions))

    return auc


def sufficiency_score(
    model: nn.Module,
    data: Data,
    node_mask: torch.Tensor,
) -> float:
    """
    Sufficiency: does the explanation subgraph alone reproduce the prediction?

    Returns a score in [0, 1] where 1 = masked subgraph fully reproduces prediction.
    """
    model.eval()
    device = data.x.device
    batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

    with torch.no_grad():
        full_pred = model(data.x, data.edge_index, data.edge_attr, batch)["prediction"]
        masked_x = data.x * node_mask.unsqueeze(-1)
        sub_pred = model(masked_x, data.edge_index, data.edge_attr, batch)["prediction"]

    diff = torch.abs(full_pred - sub_pred).mean().item()
    return max(0.0, 1.0 - diff)


def necessity_score(
    model: nn.Module,
    data: Data,
    node_mask: torch.Tensor,
) -> float:
    """
    Necessity: does removing the explanation change the prediction?

    Returns a score in [0, 1] where 1 = removing explanation fully changes prediction.
    """
    model.eval()
    device = data.x.device
    batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

    with torch.no_grad():
        full_pred = model(data.x, data.edge_index, data.edge_attr, batch)["prediction"]
        complement_x = data.x * (1.0 - node_mask.unsqueeze(-1))
        comp_pred = model(complement_x, data.edge_index, data.edge_attr, batch)["prediction"]

    diff = torch.abs(full_pred - comp_pred).mean().item()
    return min(1.0, diff)
