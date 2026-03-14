"""
CF-GNNExplainer: Counterfactual explanations for GNNs.

Finds the minimal perturbation to a molecular graph that changes
the model's prediction, revealing which structural features are
most critical for the current classification.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class CFGNNExplainer:
    """
    Counterfactual Graph Neural Network Explainer.

    Searches for minimal graph edits (edge removals/additions, feature
    perturbations) that flip the model's prediction. The resulting
    counterfactual graph highlights which bonds/atoms are essential.

    Algorithm:
        1. Start with original graph and its prediction
        2. Learn a continuous edge mask (soft perturbation)
        3. Optimize: minimize edit size while maximizing prediction change
        4. Threshold mask to get discrete counterfactual graph

    Args:
        model: Trained InterGNN model.
        lr: Learning rate for mask optimization.
        num_iterations: Optimization iterations.
        edge_loss_weight: Weight of edge sparsity penalty.
        prediction_loss_weight: Weight of prediction change reward.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.01,
        num_iterations: int = 500,
        edge_loss_weight: float = 1.0,
        prediction_loss_weight: float = 2.0,
        temperature: float = 0.5,
    ):
        self.model = model
        self.lr = lr
        self.num_iterations = num_iterations
        self.edge_loss_weight = edge_loss_weight
        self.prediction_loss_weight = prediction_loss_weight
        self.temperature = temperature

    @torch.no_grad()
    def _get_prediction(self, data: Data) -> Tuple[torch.Tensor, int]:
        """Get model prediction for a single graph."""
        self.model.eval()
        batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=data.x.device)
        out = self.model(data.x, data.edge_index, data.edge_attr, batch)
        pred = out["prediction"]
        if pred.shape[-1] > 1:
            label = pred.argmax(dim=-1).item()
        else:
            label = (pred > 0.5).int().item()
        return pred, label

    def explain(
        self,
        data: Data,
        target_class: Optional[int] = None,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Generate a counterfactual explanation for a single graph.

        Args:
            data: PyTorch Geometric Data object.
            target_class: Desired counterfactual class (default: opposite of current).
            threshold: Mask threshold for determining edge removal.

        Returns:
            Dict with:
                - 'original_pred': original prediction
                - 'cf_pred': counterfactual prediction
                - 'edge_mask': continuous edge importance mask
                - 'removed_edges': indices of removed edges
                - 'num_edits': number of edge edits made
                - 'success': whether prediction was changed
        """
        self.model.eval()
        device = data.x.device

        # Get original prediction
        original_pred, original_class = self._get_prediction(data)

        if target_class is None:
            target_class = 1 - original_class  # flip for binary

        # Initialize learnable edge mask (all edges present initially)
        num_edges = data.edge_index.shape[1]
        edge_mask = nn.Parameter(torch.ones(num_edges, device=device))

        optimizer = torch.optim.Adam([edge_mask], lr=self.lr)

        best_cf_loss = float("inf")
        best_mask = edge_mask.data.clone()

        for iteration in range(self.num_iterations):
            optimizer.zero_grad()

            # Apply sigmoid to get soft mask in [0, 1]
            mask_sigmoid = torch.sigmoid(edge_mask / self.temperature)

            # Mask edge attributes
            masked_edge_attr = data.edge_attr * mask_sigmoid.unsqueeze(-1)

            # Forward pass with masked graph
            batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
            out = self.model(data.x, data.edge_index, masked_edge_attr, batch)
            cf_pred = out["prediction"]

            # ── Prediction Loss ──
            # Encourage prediction to change toward target class
            if cf_pred.shape[-1] > 1:
                # Multi-class: cross-entropy toward target
                target = torch.tensor([target_class], device=device)
                pred_loss = -F.cross_entropy(cf_pred, target)
            else:
                # Binary: push toward target
                target = torch.tensor([[float(target_class)]], device=device)
                pred_loss = F.binary_cross_entropy_with_logits(cf_pred, target)

            # ── Edge Sparsity Loss ──
            # Minimize the number of changed edges
            edge_loss = (1.0 - mask_sigmoid).sum() / num_edges

            # ── Total Loss ──
            loss = (
                self.prediction_loss_weight * pred_loss
                + self.edge_loss_weight * edge_loss
            )

            loss.backward()
            optimizer.step()

            if loss.item() < best_cf_loss:
                best_cf_loss = loss.item()
                best_mask = mask_sigmoid.detach().clone()

        # Apply threshold to get discrete counterfactual
        removed_mask = best_mask < threshold
        removed_edges = removed_mask.nonzero(as_tuple=True)[0].cpu().tolist()

        # Get counterfactual prediction
        final_edge_attr = data.edge_attr.clone()
        final_edge_attr[removed_mask] = 0.0
        batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

        with torch.no_grad():
            self.model.eval()
            cf_out = self.model(data.x, data.edge_index, final_edge_attr, batch)
            cf_final_pred = cf_out["prediction"]

        if cf_final_pred.shape[-1] > 1:
            cf_class = cf_final_pred.argmax(dim=-1).item()
        else:
            cf_class = (cf_final_pred > 0.5).int().item()

        return {
            "original_pred": original_pred.detach().cpu(),
            "original_class": original_class,
            "cf_pred": cf_final_pred.detach().cpu(),
            "cf_class": cf_class,
            "edge_mask": best_mask.cpu(),
            "removed_edges": removed_edges,
            "num_edits": len(removed_edges),
            "success": cf_class != original_class,
        }

    def batch_explain(
        self,
        data_list: List[Data],
        target_class: Optional[int] = None,
        threshold: float = 0.5,
    ) -> List[Dict]:
        """Generate counterfactual explanations for a batch of graphs."""
        results = []
        for data in data_list:
            try:
                result = self.explain(data, target_class, threshold)
                results.append(result)
            except Exception as e:
                logger.warning(f"CF explanation failed: {e}")
                results.append({"success": False, "error": str(e)})
        return results
