"""Evaluation metrics: predictive, faithfulness, stability, chemical validity, causal."""

from inter_gnn.evaluation.predictive import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from inter_gnn.evaluation.faithfulness import deletion_auc, insertion_auc
from inter_gnn.evaluation.stability_metrics import jaccard_stability, cliff_consistency
from inter_gnn.evaluation.chemical_validity import valence_check, smarts_match_rate
from inter_gnn.evaluation.causal import invariance_violation_rate, environment_alignment_score
from inter_gnn.evaluation.statistical import paired_bootstrap_test

__all__ = [
    "compute_classification_metrics",
    "compute_regression_metrics",
    "deletion_auc",
    "insertion_auc",
    "jaccard_stability",
    "cliff_consistency",
    "valence_check",
    "smarts_match_rate",
    "invariance_violation_rate",
    "environment_alignment_score",
    "paired_bootstrap_test",
]
