"""
Inter-GNN: Interpretable GNN-Based Framework for Drug Discovery and Candidate Screening.

A modular Python package implementing:
  - Data preprocessing (standardization, graph featurization, protein graphs, concepts, cliffs, splits)
  - Core GNN model (edge/chirality-aware MPNN, cross-attention fusion, task heads)
  - Intrinsic interpretability (PAGE prototypes, MAGE motifs, concept whitening)
  - Post-hoc explainability (CF-GNNExplainer, T-GNNExplainer, CIDER diagnostics)
  - Evaluation metrics (predictive, faithfulness, stability, chemical validity, causal)
  - Visualization tools (saliency, prototypes, motifs, concepts, counterfactuals)
"""

__version__ = "1.0.0"
__author__ = "Harshal Loya, Jash Chauhan, Het Gala"

from inter_gnn.models.core_model import InterGNN
from inter_gnn.training.config import InterGNNConfig

__all__ = [
    "InterGNN",
    "InterGNNConfig",
    "__version__",
]
