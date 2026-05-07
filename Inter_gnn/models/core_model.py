"""
InterGNN unified model.

Wires together molecular encoder, interpretability layers (prototypes, 
motifs, concept whitening), and task heads for molecule classification.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from inter_gnn.models.encoders import MolecularGNNEncoder
from inter_gnn.models.task_heads import TaskHead


class InterGNN(nn.Module):
    """
    Interpretable GNN for molecular property prediction.

    Pipeline:
        SMILES graph → MolecularGNNEncoder → [node/graph embeddings]
        ConceptWhitening → aligned latent (optional)
        PrototypeLayer → prototype_scores (optional)
        MotifHead → motif_mask (optional)
        TaskHead → classification predictions

    Args:
        atom_feat_dim: Atom feature dimension from featurizer.
        bond_feat_dim: Bond feature dimension from featurizer.
        hidden_dim: Shared hidden dimension across all modules.
        num_mol_layers: GINEConv layers for molecular encoder.
        task_type: 'classification'.
        num_tasks: Number of classification tasks.
        dropout: Global dropout rate.
        readout: Graph-level readout strategy ('attention' or 'mean').
    """

    def __init__(
        self,
        atom_feat_dim: int = 55,
        bond_feat_dim: int = 14,
        hidden_dim: int = 256,
        num_mol_layers: int = 4,
        task_type: str = "classification",
        num_tasks: int = 1,
        dropout: float = 0.2,
        readout: str = "attention",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task_type = task_type

        # ── Encoders ──
        self.mol_encoder = MolecularGNNEncoder(
            atom_feat_dim=atom_feat_dim,
            bond_feat_dim=bond_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mol_layers,
            dropout=dropout,
            readout=readout,
        )

        # ── Task head ──
        self.task_head = TaskHead(
            task_type=task_type,
            input_dim=hidden_dim,
            num_tasks=num_tasks,
            hidden_dim=hidden_dim // 2,
            dropout=dropout,
        )

        # ── Interpretability hooks (set externally) ──
        self.prototype_layer: Optional[nn.Module] = None
        self.motif_head: Optional[nn.Module] = None
        self.concept_whitening: Optional[nn.Module] = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        concept_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Returns dict with keys depending on active modules:
            - 'prediction': (B, num_tasks) task predictions
            - 'mol_node_emb': atom-level embeddings
            - 'mol_graph_emb': graph-level molecular embedding
            - 'prototype_scores': prototype distances (if active)
            - 'motif_mask': motif attention mask (if active)
            - 'concept_alignment': concept whitening output (if active)
        """
        result = {}

        # ── Molecular encoding ──
        mol_out = self.mol_encoder(x, edge_index, edge_attr, batch)
        mol_node_emb = mol_out["node_embeddings"]
        mol_graph_emb = mol_out["graph_embedding"]
        result["mol_node_emb"] = mol_node_emb
        result["mol_graph_emb"] = mol_graph_emb

        # ── Graph embedding ──
        z = mol_graph_emb

        # ── Concept whitening ──
        if self.concept_whitening is not None:
            cw_out = self.concept_whitening(z, concept_labels)
            z = cw_out["aligned"]
            result["concept_alignment"] = cw_out

        # ── Prototype layer ──
        if self.prototype_layer is not None:
            proto_out = self.prototype_layer(z)
            result["prototype_scores"] = proto_out

        # ── Motif head ──
        if self.motif_head is not None:
            motif_out = self.motif_head(mol_node_emb, batch)
            result["motif_mask"] = motif_out

        # ── Task prediction ──
        prediction = self.task_head(z)
        result["prediction"] = prediction

        return result

    def get_node_importance(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Extract node importance scores via gradient-based attribution."""
        # Clone to avoid modifying original data in-place
        # Cast to float first — integer feature tensors cannot require gradients
        x_input = x.detach().clone().float().requires_grad_(True)
        mol_out = self.mol_encoder(x_input, edge_index, edge_attr, batch)
        z = mol_out["graph_embedding"]
        pred = self.task_head(z)

        # Backprop to get gradients w.r.t. atom features
        grad_outputs = torch.ones_like(pred)
        grads = torch.autograd.grad(pred, x_input, grad_outputs=grad_outputs, create_graph=False)[0]

        # L2 norm across feature dimension as importance
        importance = torch.norm(grads, dim=-1)  # (N_atoms,)
        return importance.detach()
