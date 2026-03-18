"""
PAGE-inspired prototype layer.

Learns a set of class-specific prototypes in the latent space.
Each test graph is classified by distance to its nearest prototype,
providing inherent case-based explanations.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeLayer(nn.Module):
    """
    Prototype-based classification module (PAGE-inspired).

    Maintains a learnable set of prototypes in latent space. Each input
    embedding is compared to all prototypes via L2 distance, yielding
    both classification logits (through distance-to-class mapping) and
    interpretable explanations (nearest prototype as case-based reasoning).

    Training uses pull/push losses:
      - Pull loss: minimize distance between sample and its class prototype
      - Push loss: maximize distance between sample and other-class prototypes
      - Diversity loss: ensure prototypes within a class are spread out

    Args:
        hidden_dim: Embedding dimension.
        num_classes: Number of output classes.
        num_prototypes_per_class: Prototypes per class.
        prototype_activation: 'log' (log distance) or 'linear'.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_prototypes_per_class: int = 5,
        prototype_activation: str = "log",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.total_prototypes = num_classes * num_prototypes_per_class
        self.prototype_activation = prototype_activation

        # Learnable prototype vectors
        self.prototypes = nn.Parameter(
            torch.randn(self.total_prototypes, hidden_dim) * 0.1
        )

        # Prototype-to-class mapping (fixed, not learned)
        self.register_buffer(
            "prototype_class_map",
            torch.arange(num_classes).repeat_interleave(num_prototypes_per_class),
        )

        # Classification layer: prototype similarities → class logits
        self.classifier = nn.Linear(self.total_prototypes, num_classes, bias=False)

        # Initialize classifier: each prototype votes for its class
        with torch.no_grad():
            weight = torch.zeros(num_classes, self.total_prototypes)
            for i in range(self.total_prototypes):
                cls = i // num_prototypes_per_class
                weight[cls, i] = 1.0
            self.classifier.weight.copy_(weight)

    def _compute_distances(self, z: torch.Tensor) -> torch.Tensor:
        """Compute L2 distances from each sample to each prototype."""
        # z: (B, D), prototypes: (P, D)
        # Output: (B, P) distances
        return torch.cdist(z.unsqueeze(0), self.prototypes.unsqueeze(0)).squeeze(0)

    def _distance_to_similarity(self, distances: torch.Tensor) -> torch.Tensor:
        """Convert distances to similarity scores."""
        if self.prototype_activation == "log":
            return torch.log(1.0 / (distances + 1e-8) + 1.0)
        else:
            return 1.0 / (distances + 1e-8)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute prototype-based predictions.

        Args:
            z: (B, hidden_dim) graph-level embeddings.

        Returns:
            Dict with:
                - 'distances': (B, P) L2 distances to all prototypes
                - 'similarities': (B, P) similarity scores
                - 'min_distances': (B, num_classes) minimum distance per class
                - 'nearest_prototype': (B,) index of nearest prototype
                - 'logits': (B, num_classes) classification logits
        """
        distances = self._compute_distances(z)  # (B, P)
        similarities = self._distance_to_similarity(distances)

        # Min distance to each class's prototypes
        min_distances = torch.zeros(
            z.shape[0], self.num_classes, device=z.device
        )
        for c in range(self.num_classes):
            class_mask = self.prototype_class_map == c
            class_distances = distances[:, class_mask]  # (B, K)
            min_distances[:, c] = class_distances.min(dim=1).values

        # Nearest prototype overall
        nearest = distances.argmin(dim=1)

        # Classification via similarity scores
        logits = self.classifier(similarities)

        return {
            "distances": distances,
            "similarities": similarities,
            "min_distances": min_distances,
            "nearest_prototype": nearest,
            "logits": logits,
        }

    def pull_loss(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Pull loss: minimize distance between samples and same-class prototypes.

        Each sample is pulled toward its nearest same-class prototype.
        """
        distances = self._compute_distances(z)  # (B, P)
        batch_size = z.shape[0]
        loss = torch.tensor(0.0, device=z.device)

        for c in range(self.num_classes):
            class_mask = (labels == c)
            if class_mask.sum() == 0:
                continue
            proto_mask = self.prototype_class_map == c
            class_distances = distances[class_mask][:, proto_mask]
            # Min distance to any same-class prototype
            min_dists = class_distances.min(dim=1).values
            loss = loss + min_dists.mean()

        return loss / max(self.num_classes, 1)

    def push_loss(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Push loss: maximize distance between samples and other-class prototypes.

        Each sample is pushed away from its nearest different-class prototype.
        Uses negative distance (minimized → maximizes actual distance).
        """
        distances = self._compute_distances(z)
        batch_size = z.shape[0]
        loss = torch.tensor(0.0, device=z.device)

        for c in range(self.num_classes):
            class_mask = (labels == c)
            if class_mask.sum() == 0:
                continue
            other_proto_mask = self.prototype_class_map != c
            other_distances = distances[class_mask][:, other_proto_mask]
            # Negative of min distance to nearest wrong-class prototype
            min_other_dists = other_distances.min(dim=1).values
            loss = loss - min_other_dists.mean()

        return loss / max(self.num_classes, 1)

    def diversity_loss(self) -> torch.Tensor:
        """
        Diversity loss: encourage spread among same-class prototypes.

        Penalizes prototypes within the same class for being too close.
        """
        loss = torch.tensor(0.0, device=self.prototypes.device)

        for c in range(self.num_classes):
            proto_mask = self.prototype_class_map == c
            class_protos = self.prototypes[proto_mask]  # (K, D)
            if class_protos.shape[0] < 2:
                continue

            dists = torch.cdist(class_protos.unsqueeze(0), class_protos.unsqueeze(0)).squeeze(0)
            # Mask diagonal
            mask = ~torch.eye(class_protos.shape[0], dtype=torch.bool, device=dists.device)
            pairwise_dists = dists[mask]
            # Negative mean distance (minimize → push apart)
            loss = loss - pairwise_dists.mean()

        return loss / max(self.num_classes, 1)
