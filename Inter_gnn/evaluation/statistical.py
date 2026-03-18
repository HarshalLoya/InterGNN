"""
Statistical testing: paired bootstrap and randomization tests.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def paired_bootstrap_test(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    num_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Paired bootstrap significance test between two model scores.

    Tests whether model A is significantly better than model B.

    Args:
        metric_a: (N,) per-sample metric for model A.
        metric_b: (N,) per-sample metric for model B.
        num_bootstrap: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Dict with p-value, mean difference, confidence interval.
    """
    rng = np.random.RandomState(seed)
    n = len(metric_a)
    assert len(metric_b) == n, "Both metric arrays must have the same length"

    observed_diff = float(np.mean(metric_a) - np.mean(metric_b))
    bootstrap_diffs = []

    for _ in range(num_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diff = np.mean(metric_a[idx]) - np.mean(metric_b[idx])
        bootstrap_diffs.append(diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Two-sided p-value
    p_value = float(np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff)))

    # 95% CI
    ci_lower = float(np.percentile(bootstrap_diffs, 2.5))
    ci_upper = float(np.percentile(bootstrap_diffs, 97.5))

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "significant_at_005": p_value < 0.05,
        "num_bootstrap": num_bootstrap,
    }


def randomization_test(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    targets: np.ndarray,
    metric_fn,
    num_permutations: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Randomization (permutation) test for comparing two models.

    Randomly swaps predictions between A and B to build a null distribution,
    then computes p-value for the observed difference.

    Args:
        predictions_a: (N,) predictions from model A.
        predictions_b: (N,) predictions from model B.
        targets: (N,) ground truth.
        metric_fn: Callable(predictions, targets) → float (higher = better).
        num_permutations: Number of random permutations.
        seed: Random seed.

    Returns:
        Dict with p-value and test statistics.
    """
    rng = np.random.RandomState(seed)
    n = len(predictions_a)

    score_a = metric_fn(predictions_a, targets)
    score_b = metric_fn(predictions_b, targets)
    observed_diff = score_a - score_b

    count_extreme = 0

    for _ in range(num_permutations):
        # Random swap mask
        swap = rng.rand(n) > 0.5
        perm_a = np.where(swap, predictions_b, predictions_a)
        perm_b = np.where(swap, predictions_a, predictions_b)

        perm_score_a = metric_fn(perm_a, targets)
        perm_score_b = metric_fn(perm_b, targets)
        perm_diff = perm_score_a - perm_score_b

        if np.abs(perm_diff) >= np.abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (num_permutations + 1)

    return {
        "score_a": float(score_a),
        "score_b": float(score_b),
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "significant_at_005": p_value < 0.05,
        "num_permutations": num_permutations,
    }
