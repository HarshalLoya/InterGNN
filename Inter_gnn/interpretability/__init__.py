"""Intrinsic interpretability modules: prototypes (PAGE), motifs (MAGE), concept whitening."""

from inter_gnn.interpretability.prototypes import PrototypeLayer
from inter_gnn.interpretability.motifs import MotifGeneratorHead, MotifExtractor
from inter_gnn.interpretability.concept_whitening import ConceptWhiteningLayer
from inter_gnn.interpretability.stability import ExplanationStabilityLoss

__all__ = [
    "PrototypeLayer",
    "MotifGeneratorHead",
    "MotifExtractor",
    "ConceptWhiteningLayer",
    "ExplanationStabilityLoss",
]
