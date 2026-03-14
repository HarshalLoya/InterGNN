"""
InterGNN unified model.

Wires together molecular encoder, target encoder, cross-attention fusion,
interpretability layers (prototypes, motifs, concept whitening), and task heads.
Supports both molecule-only and drug-target interaction prediction modes.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from inter_gnn.models.encoders import MolecularGNNEncoder, TargetGNNEncoder
from inter_gnn.models.attention import CrossAttentionFusion, BilinearFusion
from inter_gnn.models.task_heads import TaskHead


class InterGNN(nn.Module):
    """
    Interpretable GNN for drug discovery.

    Full pipeline:
        SMILES graph → MolecularGNNEncoder → [node/graph embeddings]
        Protein graph → TargetGNNEncoder   → [node/graph embeddings]
        CrossAttention(mol_nodes, target_nodes) → fused_embedding
        PrototypeLayer → prototype_scores (optional)
        MotifHead → motif_mask (optional)
        ConceptWhitening → aligned latent (optional)
        TaskHead → predictions

    In molecule-only mode (no target), the fusion step is skipped and
    molecular graph emb goes directly to the task head.

    Args:
        atom_feat_dim: Atom feature dimension from featurizer.
        bond_feat_dim: Bond feature dimension from featurizer.
        residue_feat_dim: Residue feature dimension for protein encoder.
        hidden_dim: Shared hidden dimension across all modules.
        num_mol_layers: GINEConv layers for molecular encoder.
        num_target_layers: GATConv layers for target encoder.
        num_attn_heads: Attention heads for cross-attention.
        task_type: 'classification' or 'regression'.
        num_tasks: Number of prediction tasks.
        dropout: Global dropout rate.
        use_target: If True, enable drug-target interaction mode.
        fusion_type: 'cross_attention' or 'bilinear'.
    """

    def __init__(
        self,
        atom_feat_dim: int = 55,
        bond_feat_dim: int = 14,
        residue_feat_dim: int = 42,
        hidden_dim: int = 256,
        num_mol_layers: int = 4,
        num_target_layers: int = 3,
        num_attn_heads: int = 4,
        task_type: str = "classification",
        num_tasks: int = 1,
        dropout: float = 0.2,
        use_target: bool = False,
        fusion_type: str = "cross_attention",
        readout: str = "attention",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_target = use_target
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

        if use_target:
            self.target_encoder = TargetGNNEncoder(
                residue_feat_dim=residue_feat_dim,
                hidden_dim=hidden_dim,
                num_layers=num_target_layers,
                num_heads=num_attn_heads,
                dropout=dropout,
                readout=readout,
            )

            if fusion_type == "cross_attention":
                self.fusion = CrossAttentionFusion(
                    mol_dim=hidden_dim,
                    target_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_attn_heads,
                    dropout=dropout,
                )
            else:
                self.fusion = BilinearFusion(hidden_dim, hidden_dim, hidden_dim)
        else:
            self.target_encoder = None
            self.fusion = None

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
        x_target: Optional[torch.Tensor] = None,
        edge_index_target: Optional[torch.Tensor] = None,
        edge_attr_target: Optional[torch.Tensor] = None,
        batch_target: Optional[torch.Tensor] = None,
        concept_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Returns dict with keys depending on active modules:
            - 'prediction': (B, num_tasks) task predictions
            - 'mol_node_emb': atom-level embeddings
            - 'mol_graph_emb': graph-level molecular embedding
            - 'fused_emb': fused drug-target embedding (if use_target)
            - 'attention_weights': cross-attention weights (if use_target)
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

        # ── Target encoding + Fusion ──
        if self.use_target and x_target is not None and self.target_encoder is not None:
            target_out = self.target_encoder(
                x_target, edge_index_target, batch_target, edge_attr_target
            )
            target_node_emb = target_out["node_embeddings"]
            target_graph_emb = target_out["graph_embedding"]
            result["target_node_emb"] = target_node_emb
            result["target_graph_emb"] = target_graph_emb

            fusion_out = self.fusion(
                mol_node_emb=mol_node_emb,
                target_node_emb=target_node_emb,
                mol_graph_emb=mol_graph_emb,
                target_graph_emb=target_graph_emb,
                mol_batch=batch,
                target_batch=batch_target,
            )
            z = fusion_out["fused_embedding"]
            result["fused_emb"] = z
            if "attention_weights" in fusion_out:
                result["attention_weights"] = fusion_out["attention_weights"]
        else:
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
