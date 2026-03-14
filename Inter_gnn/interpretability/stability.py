"""
Explanation stability regularizer.

Ensures explanations (saliency, motifs, prototypes) are stable under:
  1. Small molecular augmentations (atom masking, edge perturbation)
  2. Activity cliff pairs (similar structure should yield similar explanations)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExplanationStabilityLoss(nn.Module):
    """
    Stability regularizer for explanation consistency.

    Penalizes large changes in explanations when:
      - Input is slightly perturbed (augmentation robustness)
      - Two structurally similar molecules (cliff pair) have different explanations

    Args:
        augmentation_weight: Weight for augmentation stability term.
        cliff_weight: Weight for cliff-pair stability term.
        mask_prob: Probability of masking each atom feature during augmentation.
        edge_drop_prob: Probability of dropping each edge during augmentation.
    """

    def __init__(
        self,
        augmentation_weight: float = 1.0,
        cliff_weight: float = 1.0,
        mask_prob: float = 0.1,
        edge_drop_prob: float = 0.05,
    ):
        super().__init__()
        self.augmentation_weight = augmentation_weight
        self.cliff_weight = cliff_weight
        self.mask_prob = mask_prob
        self.edge_drop_prob = edge_drop_prob

    def _augment_features(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly mask atom features."""
        mask = torch.bernoulli(torch.full_like(x, 1.0 - self.mask_prob))
        return x * mask

    def _augment_edges(
        self, edge_index: torch.Tensor, num_edges: int
    ) -> torch.Tensor:
        """Randomly drop edges."""
        keep_mask = torch.rand(num_edges, device=edge_index.device) > self.edge_drop_prob
        return edge_index[:, keep_mask]

    def augmentation_stability(
        self,
        model: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        original_explanation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute stability loss between original and augmented explanations.

        Args:
            model: The InterGNN model.
            x: Original atom features.
            edge_index: Original edge connectivity.
            edge_attr: Original edge features.
            batch: Batch assignment.
            original_explanation: Explanation from the original input.

        Returns:
            Scalar stability loss.
        """
        # Augment input
        x_aug = self._augment_features(x)
        edge_index_aug = self._augment_edges(edge_index, edge_index.shape[1])

        # Match edge_attr to surviving edges
        keep_mask = torch.rand(edge_index.shape[1], device=edge_index.device) > self.edge_drop_prob
        edge_attr_aug = edge_attr[keep_mask] if edge_attr is not None else None
        edge_index_aug = edge_index[:, keep_mask]

        # Get augmented explanation
        with torch.no_grad():
            aug_importance = model.get_node_importance(x_aug, edge_index_aug, edge_attr_aug, batch)

        # Align lengths (augmentation doesn't change node count, only edges)
        min_len = min(original_explanation.shape[0], aug_importance.shape[0])
        orig = original_explanation[:min_len]
        aug = aug_importance[:min_len]

        # Cosine similarity loss (maximize similarity → minimize 1 - cos_sim)
        loss = 1.0 - F.cosine_similarity(orig.unsqueeze(0), aug.unsqueeze(0))

        return loss.mean()

    def cliff_stability(
        self,
        explanations: torch.Tensor,
        cliff_pairs: List[Tuple[int, int]],
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ensure activity cliff pairs have similar explanations.

        Molecules that are structurally similar (i.e. form a cliff pair)
        should produce similar explanation patterns, even if their
        activities differ.

        Args:
            explanations: (N,) per-node importance scores.
            cliff_pairs: List of (graph_idx_i, graph_idx_j) pairs.
            batch: Batch assignment tensor.

        Returns:
            Scalar cliff stability loss.
        """
        if not cliff_pairs:
            return torch.tensor(0.0, device=explanations.device)

        loss = torch.tensor(0.0, device=explanations.device)
        count = 0

        for idx_i, idx_j in cliff_pairs:
            mask_i = (batch == idx_i)
            mask_j = (batch == idx_j)

            if mask_i.sum() == 0 or mask_j.sum() == 0:
                continue

            exp_i = explanations[mask_i]
            exp_j = explanations[mask_j]

            # Normalize to same length via padding/truncation
            max_len = max(len(exp_i), len(exp_j))
            padded_i = F.pad(exp_i, (0, max_len - len(exp_i)))
            padded_j = F.pad(exp_j, (0, max_len - len(exp_j)))

            # L2 distance between normalized explanations
            norm_i = F.normalize(padded_i.unsqueeze(0), dim=1)
            norm_j = F.normalize(padded_j.unsqueeze(0), dim=1)
            dist = F.pairwise_distance(norm_i, norm_j)
            loss = loss + dist.mean()
            count += 1

        return loss / max(count, 1)

    def forward(
        self,
        original_explanation: torch.Tensor,
        cliff_pairs: Optional[List[Tuple[int, int]]] = None,
        batch: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
        x: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined stability loss.

        Returns:
            Weighted sum of augmentation and cliff stability losses.
        """
        total_loss = torch.tensor(0.0, device=original_explanation.device)

        if model is not None and x is not None and edge_index is not None and batch is not None:
            aug_loss = self.augmentation_stability(
                model, x, edge_index, edge_attr, batch, original_explanation
            )
            total_loss = total_loss + self.augmentation_weight * aug_loss

        if cliff_pairs is not None and batch is not None:
            cliff_loss = self.cliff_stability(original_explanation, cliff_pairs, batch)
            total_loss = total_loss + self.cliff_weight * cliff_loss

        return total_loss
