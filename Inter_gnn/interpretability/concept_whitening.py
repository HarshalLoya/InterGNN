"""
Concept whitening alignment layer.

Aligns latent space dimensions with predefined chemical concepts
(e.g., pharmacophores, functional groups) using a decorrelation-based
whitening transform. Provides inherent axis-aligned interpretability.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptWhiteningLayer(nn.Module):
    """
    Concept whitening module.

    Aligns specific dimensions of the latent representation with predefined
    chemical concepts using a rotation matrix learned via concept supervision.

    During training, concept labels (binary presence/absence) are used to learn
    a rotation R such that dimension i activates when concept i is present.

    Philosophy:
        - Standard whitening: decorrelate dimensions (ZCA whitening)
        - Concept alignment: rotate whitened space so axes = concepts
        - Result: each latent dimension has a human-interpretable meaning

    Args:
        hidden_dim: Full embedding dimension.
        num_concepts: Number of concepts to align.
        momentum: Running mean/covariance momentum.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_concepts: int = 30,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_concepts = min(num_concepts, hidden_dim)
        self.momentum = momentum

        # Learnable rotation for concept alignment
        self.rotation = nn.Parameter(torch.eye(hidden_dim))

        # Running statistics for whitening
        self.register_buffer("running_mean", torch.zeros(hidden_dim))
        self.register_buffer("running_cov", torch.eye(hidden_dim))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        # Concept predictor per aligned axis (for alignment loss)
        self.concept_probes = nn.ModuleList([
            nn.Linear(1, 1, bias=True)
            for _ in range(self.num_concepts)
        ])

    def _update_statistics(self, z: torch.Tensor):
        """Update running mean and covariance estimates."""
        batch_mean = z.mean(dim=0)
        centered = z - batch_mean
        batch_cov = (centered.T @ centered) / max(z.shape[0] - 1, 1)

        if self.num_batches_tracked == 0:
            self.running_mean.copy_(batch_mean)
            self.running_cov.copy_(batch_cov)
        else:
            self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
            self.running_cov.mul_(1 - self.momentum).add_(batch_cov * self.momentum)

        self.num_batches_tracked += 1

    @torch.no_grad()
    def _compute_whitening_matrix(self, cov: torch.Tensor) -> torch.Tensor:
        """Compute the ZCA whitening matrix W from a covariance matrix.

        This is deliberately **not** part of the autograd graph.  The backward
        pass of ``torch.linalg.eigh`` involves 1/(λ_i − λ_j) terms which
        explode when eigenvalues cluster (e.g. rank-deficient batch
        covariances).  Only the learned rotation should receive gradients for
        concept alignment; the whitening itself is a normalisation step.
        """
        cov = cov + 1e-3 * torch.eye(self.hidden_dim, device=cov.device)

        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        except RuntimeError:
            return torch.eye(self.hidden_dim, device=cov.device)

        if not torch.isfinite(eigenvalues).all():
            return torch.eye(self.hidden_dim, device=cov.device)

        eigenvalues = eigenvalues.clamp(min=1e-4)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(eigenvalues))
        W = eigenvectors @ D_inv_sqrt @ eigenvectors.T

        if not torch.isfinite(W).all():
            return torch.eye(self.hidden_dim, device=cov.device)

        return W

    def _whiten(self, z: torch.Tensor) -> torch.Tensor:
        """Apply ZCA whitening using running statistics.

        The whitening matrix is computed without gradients (see
        ``_compute_whitening_matrix``).  Gradients still flow through the
        centering and the matrix–vector product ``centered @ W`` w.r.t.
        ``centered``, but not through W itself.
        """
        if self.training:
            self._update_statistics(z.detach())
            mean = z.mean(dim=0).detach()
            centered = z - mean
            cov = (centered.detach().T @ centered.detach()) / max(z.shape[0] - 1, 1)
        else:
            centered = z - self.running_mean
            cov = self.running_cov

        # Compute whitening matrix (detached from autograd)
        W = self._compute_whitening_matrix(cov)

        whitened = centered @ W

        # Final NaN guard – fall back to centered input if whitening diverged
        if not torch.isfinite(whitened).all():
            return centered

        return whitened

    def forward(
        self,
        z: torch.Tensor,
        concept_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply concept-whitened transformation.

        Args:
            z: (B, hidden_dim) latent embeddings.
            concept_labels: (B, num_concepts) binary concept labels (training only).

        Returns:
            Dict with:
                - 'aligned': (B, hidden_dim) concept-aligned embeddings
                - 'concept_activations': (B, num_concepts) per-concept activation values
                - 'alignment_loss': scalar alignment loss (if labels provided)
        """
        # Whiten
        z_white = self._whiten(z)

        # Apply rotation for concept alignment
        z_aligned = z_white @ self.rotation

        # Extract concept dimensions
        concept_activations = z_aligned[:, :self.num_concepts]

        result = {
            "aligned": z_aligned,
            "concept_activations": concept_activations,
        }

        # Alignment loss: concept labels should predict concept activations
        if concept_labels is not None and self.training:
            # Handle flattened labels (PyG collation sometimes flattens (B, C) -> (B*C) or (B*C, 1))
            batch_size = z.shape[0]
            if concept_labels.numel() > batch_size and concept_labels.numel() % batch_size == 0:
                 # If total elements is multiple of batch size (and > batch size), reshape it
                 concept_labels = concept_labels.view(batch_size, -1)
            
            alignment_loss = torch.tensor(0.0, device=z.device)
            n_concepts = min(concept_labels.shape[1], self.num_concepts)

            for c in range(n_concepts):
                pred = self.concept_probes[c](concept_activations[:, c:c + 1]).squeeze(-1)
                target = concept_labels[:, c].float()
                alignment_loss = alignment_loss + F.binary_cross_entropy_with_logits(pred, target)

            alignment_loss = alignment_loss / max(n_concepts, 1)
            result["alignment_loss"] = alignment_loss

        return result

    def decorrelation_loss(self, z_aligned: torch.Tensor) -> torch.Tensor:
        """
        Decorrelation penalty on concept dimensions.

        Encourages concept-aligned dimensions to be uncorrelated,
        ensuring each dimension captures a distinct concept.
        """
        concept_dims = z_aligned[:, :self.num_concepts]  # (B, C)
        # Correlation matrix
        centered = concept_dims - concept_dims.mean(dim=0)
        cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)

        # Normalize to correlation
        std = torch.sqrt(torch.diag(cov).clamp(min=1e-8))
        corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))

        # Penalize off-diagonal correlations
        mask = ~torch.eye(self.num_concepts, device=z_aligned.device, dtype=torch.bool)
        off_diag = corr[mask]

        return off_diag.pow(2).mean()
