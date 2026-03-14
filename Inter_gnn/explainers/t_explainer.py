"""
T-GNNExplainer: Temporal event subgraph explainer.

Identifies the most important temporal subgraph patterns by learning
soft masks over edges and nodes that best preserve the model's prediction
on temporal/sequential molecular data.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class TGNNExplainer:
    """
    Temporal GNN Explainer.

    Learns a subgraph mask that identifies the most important nodes and
    edges for a prediction. Unlike CF-GNNExplainer (which seeks to change
    predictions), T-GNNExplainer finds subgraphs that suffice to produce
    the same prediction — focusing on necessary and sufficient substructures.

    Optimizes:
        max P(y=y_orig | G_sub) - λ|M| (sparsity)

    Args:
        model: Trained InterGNN model.
        lr: Learning rate for mask optimization.
        num_iterations: Number of optimization steps.
        node_mask_weight: Weight of node sparsity penalty.
        edge_mask_weight: Weight of edge sparsity penalty.
        entropy_weight: Entropy regularization (encourages discrete masks).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.01,
        num_iterations: int = 300,
        node_mask_weight: float = 0.1,
        edge_mask_weight: float = 1.0,
        entropy_weight: float = 0.1,
    ):
        self.model = model
        self.lr = lr
        self.num_iterations = num_iterations
        self.node_mask_weight = node_mask_weight
        self.edge_mask_weight = edge_mask_weight
        self.entropy_weight = entropy_weight

    def _entropy_loss(self, mask: torch.Tensor) -> torch.Tensor:
        """Binary entropy penalty to push mask toward 0 or 1."""
        p = torch.sigmoid(mask).clamp(1e-8, 1 - 1e-8)
        return -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()

    def explain(
        self,
        data: Data,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Generate a sufficient subgraph explanation for a single graph.

        Args:
            data: Input molecular graph.
            threshold: Mask threshold for selecting important elements.

        Returns:
            Dict with:
                - 'node_mask': (N,) soft node importance mask
                - 'edge_mask': (E,) soft edge importance mask
                - 'important_nodes': indices of selected nodes
                - 'important_edges': indices of selected edges
                - 'original_pred': original prediction
                - 'subgraph_pred': prediction on masked subgraph
                - 'fidelity': how well subgraph preserves original prediction
        """
        self.model.eval()
        device = data.x.device
        num_nodes = data.x.shape[0]
        num_edges = data.edge_index.shape[1]

        # Get original prediction
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        with torch.no_grad():
            orig_out = self.model(data.x, data.edge_index, data.edge_attr, batch)
            original_pred = orig_out["prediction"]

        # Initialize learnable masks
        node_mask_param = nn.Parameter(torch.zeros(num_nodes, device=device))
        edge_mask_param = nn.Parameter(torch.zeros(num_edges, device=device))

        optimizer = torch.optim.Adam([node_mask_param, edge_mask_param], lr=self.lr)

        for iteration in range(self.num_iterations):
            optimizer.zero_grad()

            node_mask_soft = torch.sigmoid(node_mask_param)
            edge_mask_soft = torch.sigmoid(edge_mask_param)

            # Apply masks to node features and edge features
            masked_x = data.x * node_mask_soft.unsqueeze(-1)
            masked_edge_attr = data.edge_attr * edge_mask_soft.unsqueeze(-1)

            # Forward through model
            out = self.model(masked_x, data.edge_index, masked_edge_attr, batch)
            subgraph_pred = out["prediction"]

            # ── Prediction preservation loss ──
            pred_loss = F.mse_loss(subgraph_pred, original_pred.detach())

            # ── Sparsity losses ──
            node_sparsity = node_mask_soft.mean()
            edge_sparsity = edge_mask_soft.mean()

            # ── Entropy regularization ──
            node_entropy = self._entropy_loss(node_mask_param)
            edge_entropy = self._entropy_loss(edge_mask_param)

            loss = (
                pred_loss
                + self.node_mask_weight * node_sparsity
                + self.edge_mask_weight * edge_sparsity
                + self.entropy_weight * (node_entropy + edge_entropy)
            )

            loss.backward()
            optimizer.step()

        # Final masks
        node_mask_final = torch.sigmoid(node_mask_param).detach()
        edge_mask_final = torch.sigmoid(edge_mask_param).detach()

        important_nodes = (node_mask_final > threshold).nonzero(as_tuple=True)[0].cpu().tolist()
        important_edges = (edge_mask_final > threshold).nonzero(as_tuple=True)[0].cpu().tolist()

        # Compute fidelity
        with torch.no_grad():
            masked_x = data.x * (node_mask_final > threshold).float().unsqueeze(-1)
            masked_ea = data.edge_attr * (edge_mask_final > threshold).float().unsqueeze(-1)
            sub_out = self.model(masked_x, data.edge_index, masked_ea, batch)
            sub_pred = sub_out["prediction"]
            fidelity = 1.0 - F.l1_loss(sub_pred, original_pred).item()

        return {
            "node_mask": node_mask_final.cpu(),
            "edge_mask": edge_mask_final.cpu(),
            "important_nodes": important_nodes,
            "important_edges": important_edges,
            "original_pred": original_pred.detach().cpu(),
            "subgraph_pred": sub_pred.detach().cpu(),
            "fidelity": fidelity,
        }

    def batch_explain(
        self,
        data_list: List[Data],
        threshold: float = 0.5,
    ) -> List[Dict]:
        """Generate subgraph explanations for a batch of graphs."""
        results = []
        for data in data_list:
            try:
                results.append(self.explain(data, threshold))
            except Exception as e:
                logger.warning(f"T-GNNExplainer failed: {e}")
                results.append({"fidelity": 0.0, "error": str(e)})
        return results
