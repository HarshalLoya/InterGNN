"""
Predictive performance metrics for classification and regression tasks.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    accuracy_score, f1_score, matthews_corrcoef,
    mean_squared_error, mean_absolute_error, r2_score,
)
from scipy.stats import pearsonr, spearmanr


def compute_classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        predictions: (N,) or (N, T) predicted probabilities.
        targets: (N,) or (N, T) binary ground truth.
        threshold: Decision threshold for binary metrics.

    Returns:
        Dict with ROC-AUC, PR-AUC, accuracy, F1, MCC.
    """
    metrics = {}

    # Handle multi-task: ensure both are 2D
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    num_tasks = predictions.shape[1]
    aucs, pr_aucs, accs, f1s, mccs = [], [], [], [], []

    for t in range(num_tasks):
        pred_t = predictions[:, t]
        target_t = targets[:, t]

        # Remove NaN targets
        valid = ~np.isnan(target_t)
        if valid.sum() < 2:
            continue

        pred_t = pred_t[valid]
        target_t = target_t[valid].astype(int)

        # Skip if single class
        if len(np.unique(target_t)) < 2:
            continue

        try:
            aucs.append(roc_auc_score(target_t, pred_t))
        except ValueError:
            pass

        try:
            pr_aucs.append(average_precision_score(target_t, pred_t))
        except ValueError:
            pass

        pred_binary = (pred_t >= threshold).astype(int)
        accs.append(accuracy_score(target_t, pred_binary))
        f1s.append(f1_score(target_t, pred_binary, zero_division=0))
        mccs.append(matthews_corrcoef(target_t, pred_binary))

    metrics["roc_auc"] = float(np.mean(aucs)) if aucs else 0.0
    metrics["pr_auc"] = float(np.mean(pr_aucs)) if pr_aucs else 0.0
    metrics["accuracy"] = float(np.mean(accs)) if accs else 0.0
    metrics["f1_score"] = float(np.mean(f1s)) if f1s else 0.0
    metrics["mcc"] = float(np.mean(mccs)) if mccs else 0.0

    return metrics


def compute_regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        predictions: (N,) or (N, 1) predicted values.
        targets: (N,) or (N, 1) ground truth values.

    Returns:
        Dict with RMSE, MAE, R², Pearson r, Spearman ρ, CI.
    """
    pred = predictions.flatten()
    target = targets.flatten()

    valid = ~(np.isnan(pred) | np.isnan(target))
    pred = pred[valid]
    target = target[valid]

    metrics = {}
    metrics["rmse"] = float(np.sqrt(mean_squared_error(target, pred)))
    metrics["mae"] = float(mean_absolute_error(target, pred))
    metrics["r2"] = float(r2_score(target, pred)) if len(target) > 1 else 0.0

    if len(target) > 2:
        r, p = pearsonr(target, pred)
        metrics["pearson_r"] = float(r)
        metrics["pearson_p"] = float(p)

        rho, p_s = spearmanr(target, pred)
        metrics["spearman_rho"] = float(rho)

        # Concordance Index
        metrics["ci"] = _concordance_index(target, pred)
    else:
        metrics["pearson_r"] = 0.0
        metrics["spearman_rho"] = 0.0
        metrics["ci"] = 0.0

    return metrics


def _concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute concordance index (C-index) for ranking quality. Vectorized for speed."""
    n = len(y_true)
    if n < 2:
        return 0.5

    # For very large datasets, subsample to keep runtime manageable on CPU
    max_pairs = 500_000
    if n * (n - 1) // 2 > max_pairs:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=min(n, int(np.sqrt(2 * max_pairs)) + 1), replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
        n = len(y_true)

    # Vectorized: compare all pairs using broadcasting
    true_diff = y_true[:, None] - y_true[None, :]  # (n, n)
    pred_diff = y_pred[:, None] - y_pred[None, :]  # (n, n)

    # Upper triangle only (i < j)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    # Only pairs where true values differ
    valid = mask & (true_diff != 0)

    if not valid.any():
        return 0.5

    concordant = np.sum((true_diff[valid] > 0) & (pred_diff[valid] > 0)) + \
                 np.sum((true_diff[valid] < 0) & (pred_diff[valid] < 0))
    total = np.sum(valid)

    return float(concordant / total) if total > 0 else 0.5
