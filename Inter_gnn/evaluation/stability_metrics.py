"""
Explanation stability metrics: Jaccard stability and cliff consistency.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch


def jaccard_stability(
    explanations_a: List[set],
    explanations_b: List[set],
) -> float:
    """
    Jaccard stability between two sets of explanations.

    Measures how consistent explanations are across augmented inputs.

    Args:
        explanations_a: List of sets of important atom indices (original).
        explanations_b: List of sets of important atom indices (augmented).

    Returns:
        Mean Jaccard index across all pairs.
    """
    scores = []
    for set_a, set_b in zip(explanations_a, explanations_b):
        if len(set_a | set_b) == 0:
            scores.append(1.0)
        else:
            scores.append(len(set_a & set_b) / len(set_a | set_b))

    return float(np.mean(scores)) if scores else 0.0


def cliff_consistency(
    explanations: List[np.ndarray],
    cliff_pairs: List[Tuple[int, int]],
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Measure explanation consistency across activity cliff pairs.

    Structurally similar molecules (cliff pairs) should have similar
    explanation patterns, even if activities differ.

    Args:
        explanations: List of (N_i,) importance arrays, one per molecule.
        cliff_pairs: List of (idx_i, idx_j) index pairs.
        top_k: Number of top atoms to compare.

    Returns:
        Dict with consistency scores and statistics.
    """
    if not cliff_pairs:
        return {"mean_consistency": 0.0, "num_pairs": 0}

    consistencies = []

    for idx_i, idx_j in cliff_pairs:
        if idx_i >= len(explanations) or idx_j >= len(explanations):
            continue

        exp_i = explanations[idx_i]
        exp_j = explanations[idx_j]

        # Get top-k atom indices
        k_i = min(top_k, len(exp_i))
        k_j = min(top_k, len(exp_j))

        top_i = set(np.argsort(exp_i)[-k_i:].tolist())
        top_j = set(np.argsort(exp_j)[-k_j:].tolist())

        # Jaccard overlap of top-k
        if len(top_i | top_j) > 0:
            jaccard = len(top_i & top_j) / len(top_i | top_j)
        else:
            jaccard = 1.0

        consistencies.append(jaccard)

    return {
        "mean_consistency": float(np.mean(consistencies)) if consistencies else 0.0,
        "std_consistency": float(np.std(consistencies)) if consistencies else 0.0,
        "min_consistency": float(np.min(consistencies)) if consistencies else 0.0,
        "max_consistency": float(np.max(consistencies)) if consistencies else 0.0,
        "num_pairs": len(consistencies),
    }


def rank_correlation_stability(
    importances_a: np.ndarray,
    importances_b: np.ndarray,
) -> float:
    """
    Spearman rank correlation between two importance vectors.

    Args:
        importances_a: (N,) first importance vector.
        importances_b: (N,) second importance vector.

    Returns:
        Spearman correlation coefficient.
    """
    from scipy.stats import spearmanr

    min_len = min(len(importances_a), len(importances_b))
    if min_len < 2:
        return 0.0

    rho, _ = spearmanr(importances_a[:min_len], importances_b[:min_len])
    return float(rho) if not np.isnan(rho) else 0.0
