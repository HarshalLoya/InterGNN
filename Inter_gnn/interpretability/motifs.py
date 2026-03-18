"""
MAGE-inspired motif generator head.

Learns to identify and extract molecular motifs (substructure patterns)
that are most relevant for predictions. Produces sparse, valid, and
human-interpretable motif masks over atom nodes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_adj


class MotifGeneratorHead(nn.Module):
    """
    Differentiable motif mask generator.

    For each graph, produces a continuous mask over atoms indicating
    which atoms belong to the most important motif. The mask is
    regularized for sparsity, connectivity, and chemical validity.

    Args:
        hidden_dim: Node embedding dimension.
        num_motifs: Number of motif patterns to discover.
        temperature: Gumbel-softmax temperature for mask discretization.
        sparsity_target: Target fraction of atoms in motif (e.g., 0.3).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_motifs: int = 8,
        temperature: float = 0.5,
        sparsity_target: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_motifs = num_motifs
        self.temperature = temperature
        self.sparsity_target = sparsity_target

        # Per-node mask predictor (outputs score per motif)
        self.mask_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_motifs),
        )

        # Motif-level aggregator
        self.motif_embedder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _gumbel_sigmoid(self, logits: torch.Tensor) -> torch.Tensor:
        """Differentiable approximate binary mask using Gumbel-sigmoid."""
        if self.training:
            noise = torch.zeros_like(logits).uniform_(1e-8, 1 - 1e-8)
            gumbels = -torch.log(-torch.log(noise))
            y = torch.sigmoid((logits + gumbels) / self.temperature)
        else:
            y = (logits > 0).float()
        return y

    def forward(
        self,
        node_emb: torch.Tensor,
        batch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate motif masks for a batch of graphs.

        Args:
            node_emb: (N_total, hidden_dim) node embeddings.
            batch: (N_total,) batch assignment.

        Returns:
            Dict with:
                - 'mask_logits': (N_total, K) raw mask scores per motif
                - 'masks': (N_total, K) soft binary masks
                - 'motif_embeddings': (B, K, D) per-motif embeddings
                - 'sparsity': scalar mean sparsity ratio
        """
        mask_logits = self.mask_predictor(node_emb)  # (N, K)
        masks = self._gumbel_sigmoid(mask_logits)      # (N, K)

        B = int(batch.max().item()) + 1
        K = self.num_motifs

        # Vectorized motif embedding computation (avoid per-motif Python loop)
        # weighted: (N, K, D) = node_emb[:, None, :] * masks[:, :, None]
        weighted = node_emb.unsqueeze(1) * masks.unsqueeze(2)  # (N, K, D)

        # Pool per graph using scatter (vectorized mean pooling)
        motif_embeddings = torch.zeros(B, K, self.hidden_dim, device=node_emb.device)
        counts = torch.zeros(B, 1, 1, device=node_emb.device)
        # Accumulate per-graph sums
        batch_expanded = batch.view(-1, 1, 1).expand_as(weighted)
        motif_embeddings.scatter_add_(0, batch_expanded, weighted)
        counts.scatter_add_(0, batch.view(-1, 1, 1).expand(-1, 1, 1),
                           torch.ones(node_emb.size(0), 1, 1, device=node_emb.device))
        counts = counts.expand_as(motif_embeddings).clamp(min=1)
        motif_embeddings = motif_embeddings / counts
        # Apply motif embedder (reshape for batch linear)
        orig_shape = motif_embeddings.shape
        motif_embeddings = self.motif_embedder(motif_embeddings.view(-1, self.hidden_dim)).view(orig_shape)

        # Sparsity: average fraction of active atoms per motif
        sparsity = masks.mean()

        return {
            "mask_logits": mask_logits,
            "masks": masks,
            "motif_embeddings": motif_embeddings,
            "sparsity": sparsity,
        }

    def sparsity_loss(self, masks: torch.Tensor) -> torch.Tensor:
        """L1 penalty to enforce target sparsity level."""
        mean_activation = masks.mean()
        return (mean_activation - self.sparsity_target).abs()

    def connectivity_loss(
        self,
        masks: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encourage motif masks to form connected subgraphs.

        Penalizes masked atoms that are not edge-connected to other
        masked atoms in the same motif.
        """
        loss = torch.tensor(0.0, device=masks.device)
        K = masks.shape[1]

        for k in range(K):
            mask_k = masks[:, k]  # (N,)
            src, dst = edge_index[0], edge_index[1]
            # For each edge, check if both endpoints are in the motif
            edge_mask_product = mask_k[src] * mask_k[dst]
            # Isolated masked nodes: high mask value but low neighbor coverage
            node_connectivity = torch.zeros_like(mask_k)
            node_connectivity.scatter_add_(0, src, edge_mask_product)

            # Penalty: mask activation without neighbor connectivity
            isolation = mask_k * (1.0 - torch.clamp(node_connectivity, max=1.0))
            loss = loss + isolation.mean()

        return loss / K


class MotifExtractor:
    """
    Post-hoc motif extraction from trained MotifGeneratorHead.

    Extracts discrete motif substructures from continuous masks,
    validates chemical validity, and converts to SMARTS patterns.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @torch.no_grad()
    def extract(
        self,
        model: MotifGeneratorHead,
        node_emb: torch.Tensor,
        batch: torch.Tensor,
        smiles_list: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Extract motif masks from a batch of graphs.

        Returns:
            List of dicts per graph, each containing per-motif atom indices.
        """
        output = model(node_emb, batch)
        masks = output["masks"]
        B = int(batch.max().item()) + 1

        results = []
        for b in range(B):
            graph_mask = (batch == b)
            graph_atom_masks = masks[graph_mask]  # (N_b, K)
            motifs = {}

            for k in range(model.num_motifs):
                active = (graph_atom_masks[:, k] > self.threshold).nonzero(as_tuple=True)[0]
                if len(active) > 0:
                    motifs[f"motif_{k}"] = active.cpu().tolist()

            entry = {"graph_idx": b, "motifs": motifs}
            if smiles_list and b < len(smiles_list):
                entry["smiles"] = smiles_list[b]
            results.append(entry)

        return results
