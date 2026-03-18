"""
GNN Encoders for molecules and protein targets.

Molecular encoder: edge-aware MPNN with chirality features (GINEConv).
Target encoder: residue-level GNN with positional encodings (GATConv).
Both produce graph-level embeddings via attention-weighted readout.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv, GATConv, global_mean_pool, global_add_pool,
    BatchNorm,
)
from torch_geometric.data import Data, Batch


class AttentionReadout(nn.Module):
    """Attention-weighted graph-level readout pooling."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        gate_scores = self.gate(x)  # (N, 1)
        # Softmax within each graph
        from torch_geometric.utils import softmax
        attn = softmax(gate_scores, batch, dim=0)
        # Weighted sum
        weighted = x * attn
        return global_add_pool(weighted, batch)


class MolecularGNNEncoder(nn.Module):
    """
    Edge-aware molecular graph encoder using GINEConv layers.

    Processes molecular graphs with rich atom and bond features,
    including chirality. Produces both node-level and graph-level embeddings.

    Architecture:
        - Input projection of atom/bond features
        - L layers of GINEConv message passing with edge embeddings
        - Batch normalization + dropout per layer
        - Attention-weighted readout for graph-level embedding

    Args:
        atom_feat_dim: Dimension of input atom features.
        bond_feat_dim: Dimension of input bond features.
        hidden_dim: Hidden dimension for all layers.
        num_layers: Number of message passing layers.
        dropout: Dropout rate.
        readout: Readout method ('attention', 'mean', 'sum').
    """

    def __init__(
        self,
        atom_feat_dim: int = 55,
        bond_feat_dim: int = 14,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.2,
        readout: str = "attention",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Input projections
        self.atom_embed = nn.Linear(atom_feat_dim, hidden_dim)
        self.bond_embed = nn.Linear(bond_feat_dim, hidden_dim)

        # Message passing layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.GELU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            conv = GINEConv(mlp, edge_dim=hidden_dim)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_dim))

        # Readout
        if readout == "attention":
            self.readout_fn = AttentionReadout(hidden_dim)
        elif readout == "mean":
            self.readout_fn = None
            self._readout_type = "mean"
        elif readout == "sum":
            self.readout_fn = None
            self._readout_type = "sum"
        else:
            raise ValueError(f"Unknown readout: {readout}")

        self._readout_type = readout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> dict:
        """
        Forward pass through the molecular encoder.

        Returns:
            Dict with:
                - 'node_embeddings': (N, hidden_dim) per-atom embeddings
                - 'graph_embedding': (B, hidden_dim) per-graph embedding
        """
        # Project atom and bond features (cast to float for integer-feature datasets)
        h = self.atom_embed(x.float())
        edge_emb = self.bond_embed(edge_attr.float())

        # Message passing
        node_embeddings_per_layer = [h]
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_emb)
            h = self.batch_norms[i](h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + node_embeddings_per_layer[-1]  # residual
            node_embeddings_per_layer.append(h)

        # Graph-level readout
        if self._readout_type == "attention":
            z = self.readout_fn(h, batch)
        elif self._readout_type == "mean":
            z = global_mean_pool(h, batch)
        else:
            z = global_add_pool(h, batch)

        return {
            "node_embeddings": h,
            "graph_embedding": z,
        }


class TargetGNNEncoder(nn.Module):
    """
    Residue-level GNN encoder for protein target graphs.

    Uses GATConv with multi-head attention to capture residue interactions.
    Includes positional encoding integration and attention-weighted readout.

    Args:
        residue_feat_dim: Input residue feature dimension.
        hidden_dim: Hidden dimension.
        num_layers: Number of GAT layers.
        num_heads: Number of attention heads per GAT layer.
        dropout: Dropout rate.
        readout: Readout method ('attention', 'mean', 'sum').
    """

    def __init__(
        self,
        residue_feat_dim: int = 42,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        readout: str = "attention",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.residue_embed = nn.Linear(residue_feat_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_dim
            # GATConv heads output hidden_dim // num_heads each, concat gives hidden_dim
            head_dim = hidden_dim // num_heads
            conv = GATConv(
                in_channels=in_dim,
                out_channels=head_dim,
                heads=num_heads,
                dropout=dropout,
                concat=True,
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_dim))

        if readout == "attention":
            self.readout_fn = AttentionReadout(hidden_dim)
        else:
            self.readout_fn = None

        self._readout_type = readout
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass through the target encoder.

        Returns:
            Dict with 'node_embeddings' and 'graph_embedding'.
        """
        h = self.residue_embed(x.float())

        for i in range(self.num_layers):
            h_res = h
            h = self.convs[i](h, edge_index)
            h = self.batch_norms[i](h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_res  # residual

        if self._readout_type == "attention":
            z = self.readout_fn(h, batch)
        elif self._readout_type == "mean":
            z = global_mean_pool(h, batch)
        else:
            z = global_add_pool(h, batch)

        return {
            "node_embeddings": h,
            "graph_embedding": z,
        }
