"""
Loss functions for InterGNN training.

Combines prediction loss with interpretability regularization losses:
prototype pull/push, motif sparsity/connectivity, concept alignment,
and explanation stability.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionLoss(nn.Module):
    """Task prediction loss (BCE for classification, MSE for regression)."""

    def __init__(self, task_type: str = "classification", num_tasks: int = 1):
        super().__init__()
        self.task_type = task_type
        if task_type == "classification":
            self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        else:
            self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        target = target.float()

        # Handle NaN labels (multi-task missing labels)
        mask = ~torch.isnan(target)
        if mask.any():
            return self.criterion(pred[mask], target[mask])
        return torch.tensor(0.0, device=pred.device)


class TotalLoss(nn.Module):
    """
    Combined training objective for InterGNN.

    Total = λ_pred * L_prediction
          + λ_pull * L_prototype_pull
          + λ_push * L_prototype_push
          + λ_div  * L_prototype_diversity
          + λ_motif * L_motif_sparsity
          + λ_conn * L_motif_connectivity
          + λ_cw   * L_concept_alignment
          + λ_decorr * L_decorrelation
          + λ_stab * L_stability

    Args:
        task_type: 'classification' or 'regression'.
        num_tasks: Number of prediction tasks.
        lambda_pred: Weight for prediction loss.
        lambda_pull: Weight for prototype pull loss.
        lambda_push: Weight for prototype push loss.
        lambda_diversity: Weight for prototype diversity loss.
        lambda_motif: Weight for motif sparsity loss.
        lambda_connectivity: Weight for motif connectivity loss.
        lambda_concept: Weight for concept alignment loss.
        lambda_decorrelation: Weight for concept decorrelation loss.
        lambda_stability: Weight for explanation stability loss.
    """

    def __init__(
        self,
        task_type: str = "classification",
        num_tasks: int = 1,
        lambda_pred: float = 1.0,
        lambda_pull: float = 0.1,
        lambda_push: float = 0.05,
        lambda_diversity: float = 0.01,
        lambda_motif: float = 0.1,
        lambda_connectivity: float = 0.05,
        lambda_concept: float = 0.1,
        lambda_decorrelation: float = 0.01,
        lambda_stability: float = 0.05,
    ):
        super().__init__()
        self.pred_loss = PredictionLoss(task_type, num_tasks)
        self.lambda_pred = lambda_pred
        self.lambda_pull = lambda_pull
        self.lambda_push = lambda_push
        self.lambda_diversity = lambda_diversity
        self.lambda_motif = lambda_motif
        self.lambda_connectivity = lambda_connectivity
        self.lambda_concept = lambda_concept
        self.lambda_decorrelation = lambda_decorrelation
        self.lambda_stability = lambda_stability

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        model: Optional[nn.Module] = None,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        cliff_pairs=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.

        Args:
            model_output: Dict from InterGNN.forward() with predictions and explanations.
            targets: Ground truth labels.
            model: InterGNN model (for stability loss computation).
            edge_index: Edge indices (for connectivity loss).
            batch: Batch tensor.
            cliff_pairs: List of cliff pair tuples.

        Returns:
            Dict with 'total' and individual loss component values.
        """
        losses = {}
        total = torch.tensor(0.0, device=targets.device)

        # ── Prediction Loss ──
        pred = model_output["prediction"]
        l_pred = self.pred_loss(pred, targets)
        losses["prediction"] = l_pred
        total = total + self.lambda_pred * l_pred

        # ── Prototype Losses ──
        if "prototype_scores" in model_output and model is not None:
            proto_layer = model.prototype_layer
            if proto_layer is not None:
                z = model_output.get("fused_emb", model_output.get("mol_graph_emb"))

                # Derive a 1-D integer class label for prototypes.
                # Single-task classification: squeeze to (B,) long.
                # Multi-label classification: use majority vote (>50% tasks active → class 1).
                if targets.dim() > 1 and targets.shape[-1] > 1:
                    # Multi-label → binary pseudo-label
                    valid = (targets >= 0).float()          # mask out -1 / NaN entries
                    active = ((targets > 0.5) & (targets >= 0)).float()
                    frac_active = active.sum(dim=-1) / valid.sum(dim=-1).clamp(min=1)
                    labels = (frac_active > 0.5).long()     # (B,)
                else:
                    labels = targets.long().squeeze(-1)

                l_pull = proto_layer.pull_loss(z, labels)
                l_push = proto_layer.push_loss(z, labels)
                l_div = proto_layer.diversity_loss()

                losses["pull"] = l_pull
                losses["push"] = l_push
                losses["diversity"] = l_div
                total = total + self.lambda_pull * l_pull
                total = total + self.lambda_push * l_push
                total = total + self.lambda_diversity * l_div

        # ── Motif Losses ──
        if "motif_mask" in model_output:
            motif_out = model_output["motif_mask"]
            if isinstance(motif_out, dict):
                masks = motif_out.get("masks")
                if masks is not None:
                    motif_head = model.motif_head if model else None
                    if motif_head is not None:
                        l_sparse = motif_head.sparsity_loss(masks)
                        losses["motif_sparsity"] = l_sparse
                        total = total + self.lambda_motif * l_sparse

                        if edge_index is not None and batch is not None:
                            l_conn = motif_head.connectivity_loss(masks, edge_index, batch)
                            losses["motif_connectivity"] = l_conn
                            total = total + self.lambda_connectivity * l_conn

        # ── Concept Whitening Losses ──
        if "concept_alignment" in model_output:
            cw_out = model_output["concept_alignment"]
            if isinstance(cw_out, dict):
                if "alignment_loss" in cw_out:
                    l_align = cw_out["alignment_loss"]
                    losses["concept_alignment"] = l_align
                    total = total + self.lambda_concept * l_align

                if model is not None and model.concept_whitening is not None:
                    z_aligned = cw_out.get("aligned")
                    if z_aligned is not None:
                        l_decorr = model.concept_whitening.decorrelation_loss(z_aligned)
                        losses["decorrelation"] = l_decorr
                        total = total + self.lambda_decorrelation * l_decorr

        losses["total"] = total
        return losses
