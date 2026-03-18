"""
Causal invariance evaluation scores (CIDER-style metrics).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def invariance_violation_rate(
    env_predictions: Dict[str, np.ndarray],
    threshold: float = 0.1,
) -> float:
    """
    Fraction of predictions that violate cross-environment invariance.

    A prediction violates invariance if the model's output for the same
    molecule changes by more than `threshold` across environments.

    Args:
        env_predictions: Dict mapping environment name to (N,) predictions.
        threshold: Maximum allowed prediction difference.

    Returns:
        Fraction of violations.
    """
    if len(env_predictions) < 2:
        return 0.0

    envs = list(env_predictions.values())
    n = min(len(e) for e in envs)
    violations = 0

    for i in range(n):
        preds_i = [e[i] for e in envs if i < len(e)]
        if len(preds_i) < 2:
            continue
        max_diff = max(preds_i) - min(preds_i)
        if max_diff > threshold:
            violations += 1

    return violations / max(n, 1)


def environment_alignment_score(
    feature_importances_by_env: Dict[str, List[np.ndarray]],
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Measure how well feature importances align across environments.

    For invariant features: high importance in all environments.
    For spurious features: high importance in one env only.

    Args:
        feature_importances_by_env: Dict[env_name, list of importance arrays].
        top_k: Top features to compare.

    Returns:
        Dict with alignment and variance scores.
    """
    env_names = list(feature_importances_by_env.keys())
    if len(env_names) < 2:
        return {"alignment": 1.0, "variance": 0.0}

    # Average importance per feature, per environment
    env_means = {}
    for env_name, imps in feature_importances_by_env.items():
        if imps:
            stacked = np.array([imp for imp in imps if len(imp) > 0])
            if len(stacked) > 0:
                env_means[env_name] = np.mean(stacked, axis=0)

    if len(env_means) < 2:
        return {"alignment": 1.0, "variance": 0.0}

    # Cross-environment Jaccard of top-k features
    topk_sets = {}
    for env_name, mean_imp in env_means.items():
        k = min(top_k, len(mean_imp))
        topk_sets[env_name] = set(np.argsort(mean_imp)[-k:].tolist())

    jaccard_scores = []
    for i, env_i in enumerate(env_names):
        for j, env_j in enumerate(env_names):
            if j <= i or env_i not in topk_sets or env_j not in topk_sets:
                continue
            union = topk_sets[env_i] | topk_sets[env_j]
            if len(union) > 0:
                jaccard_scores.append(
                    len(topk_sets[env_i] & topk_sets[env_j]) / len(union)
                )

    # Cross-environment variance per feature
    all_means = np.array(list(env_means.values()))
    per_feature_var = np.var(all_means, axis=0)

    return {
        "alignment": float(np.mean(jaccard_scores)) if jaccard_scores else 0.0,
        "mean_feature_variance": float(np.mean(per_feature_var)),
        "max_feature_variance": float(np.max(per_feature_var)),
    }
