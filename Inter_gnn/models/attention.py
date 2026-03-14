"""
Cross-attention and fusion modules for drug-target interaction.

Implements scaled dot-product cross-attention between atom and residue
embeddings, and bilinear fusion as an alternative interaction head.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Atom-Residue Cross-Attention & Fusion module.

    Computes cross-attention between molecular atom embeddings (queries)
    and protein residue embeddings (keys/values):

        A = softmax(QK^T / sqrt(d))
        H_m_tilde = A @ V

    The fused representation concatenates the graph-level embeddings
    with pooled cross-attention output:
        z_fused = [z_m || z_p || POOL(H_m_tilde)]

    Args:
        mol_dim: Molecular node embedding dimension.
        target_dim: Target node embedding dimension.
        hidden_dim: Internal projection dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
    """

    def __init__(
        self,
        mol_dim: int = 256,
        target_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        self.W_q = nn.Linear(mol_dim, hidden_dim)       # molecular → queries
        self.W_k = nn.Linear(target_dim, hidden_dim)     # target → keys
        self.W_v = nn.Linear(target_dim, hidden_dim)     # target → values

        self.attn_dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Fusion projection: [z_m || z_p || pool(H_tilde)] → hidden_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mol_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Multi-head scaled dot-product attention.

        Args:
            Q: (B, N_mol, hidden_dim) query tensor.
            K: (B, N_target, hidden_dim) key tensor.
            V: (B, N_target, hidden_dim) value tensor.

        Returns:
            Tuple of (attended_values, attention_weights).
        """
        B, N_mol, _ = Q.shape
        _, N_target, _ = K.shape

        # Reshape for multi-head: (B, num_heads, N, head_dim)
        Q = Q.view(B, N_mol, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N_target, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N_target, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, N_mol, N_target)

        # Apply masks if provided
        if target_mask is not None:
            # target_mask: (B, N_target) → (B, 1, 1, N_target)
            mask = target_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # (B, H, N_mol, head_dim)
        attended = attended.transpose(1, 2).contiguous().view(B, N_mol, self.hidden_dim)

        return attended, attn_weights.mean(dim=1)  # average over heads for interpretability

    def forward(
        self,
        mol_node_emb: torch.Tensor,
        target_node_emb: torch.Tensor,
        mol_graph_emb: torch.Tensor,
        target_graph_emb: torch.Tensor,
        mol_batch: torch.Tensor,
        target_batch: torch.Tensor,
    ) -> dict:
        """
        Compute cross-attention fusion between drug and target.

        Args:
            mol_node_emb: (N_total_atoms, D) atom embeddings.
            target_node_emb: (N_total_residues, D) residue embeddings.
            mol_graph_emb: (B, D) molecular graph embeddings.
            target_graph_emb: (B, D) target graph embeddings.
            mol_batch: Batch assignment for atoms.
            target_batch: Batch assignment for residues.

        Returns:
            Dict with 'fused_embedding', 'attended_mol_emb', 'attention_weights'.
        """
        # Convert scatter-format to padded batch format for attention
        B = mol_graph_emb.shape[0]
        device = mol_node_emb.device

        # Get max node counts
        mol_counts = torch.bincount(mol_batch, minlength=B)
        target_counts = torch.bincount(target_batch, minlength=B)
        max_mol = int(mol_counts.max())
        max_target = int(target_counts.max())

        # Pad to (B, max_N, D)
        mol_padded = torch.zeros(B, max_mol, self.hidden_dim, device=device)
        target_padded = torch.zeros(B, max_target, self.hidden_dim, device=device)

        for i in range(B):
            mol_mask_i = (mol_batch == i)
            target_mask_i = (target_batch == i)
            n_mol = mol_mask_i.sum()
            n_target = target_mask_i.sum()
            mol_padded[i, :n_mol] = mol_node_emb[mol_mask_i][:n_mol]
            target_padded[i, :n_target] = target_node_emb[target_mask_i][:n_target]

        # Project
        Q = self.W_q(mol_padded)
        K = self.W_k(target_padded)
        V = self.W_v(target_padded)

        # Cross-attention
        H_tilde, attn_weights = self._attention(Q, K, V)
        H_tilde = self.output_proj(H_tilde)
        H_tilde = self.layer_norm(H_tilde + mol_padded)  # residual + norm

        # Pool attended embeddings → (B, D)
        pooled = H_tilde.mean(dim=1)

        # Fuse: [z_m || z_p || pool(H_tilde)]
        fused = torch.cat([mol_graph_emb, target_graph_emb, pooled], dim=-1)
        fused = self.fusion_proj(fused)

        return {
            "fused_embedding": fused,
            "attended_mol_emb": H_tilde,
            "attention_weights": attn_weights,
        }


class BilinearFusion(nn.Module):
    """
    Bilinear interaction module (ablation alternative to cross-attention).

    Computes: z_fused = W_1 z_m + W_2 z_p + z_m^T W_b z_p + b

    Args:
        mol_dim: Molecular embedding dimension.
        target_dim: Target embedding dimension.
        output_dim: Output fusion dimension.
    """

    def __init__(self, mol_dim: int = 256, target_dim: int = 256, output_dim: int = 256):
        super().__init__()
        self.W_mol = nn.Linear(mol_dim, output_dim, bias=False)
        self.W_target = nn.Linear(target_dim, output_dim, bias=False)
        self.W_bilinear = nn.Bilinear(mol_dim, target_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()

    def forward(
        self,
        mol_graph_emb: torch.Tensor,
        target_graph_emb: torch.Tensor,
        **kwargs,
    ) -> dict:
        """
        Bilinear fusion of drug and target embeddings.

        Returns:
            Dict with 'fused_embedding'.
        """
        fused = (
            self.W_mol(mol_graph_emb)
            + self.W_target(target_graph_emb)
            + self.W_bilinear(mol_graph_emb, target_graph_emb)
        )
        fused = self.layer_norm(fused)
        fused = self.activation(fused)

        return {"fused_embedding": fused}
