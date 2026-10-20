"""Evaluation metrics and calibration diagnostics for probabilistic shot quality models."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def calculate_expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5
) -> float:
    """Calculate Expected Calibration Error (ECE) across equal-width probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    # Clip to valid range [0, n_bins - 1]
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        n_in_bin = np.sum(mask)
        if n_in_bin > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (n_in_bin / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_binary_classification_metrics(
    y_true: np.ndarray | pd.Series, y_prob: np.ndarray
) -> Dict[str, Any]:
    """Compute log loss, Brier score, ROC-AUC, and calibration error.

    Returns dict with metrics, handling degenerate edge cases (e.g. single-class folds).
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1 - 1e-15)

    n_samples = len(y_t)
    if n_samples == 0:
        return {
            "n_samples": 0,
            "log_loss": None,
            "brier_score": None,
            "roc_auc": None,
            "ece": None,
            "mean_prediction": None,
            "positive_rate": None,
        }

    # Brier score
    brier = float(brier_score_loss(y_t, y_p))

    # Log loss
    ll = float(log_loss(y_t, y_p, labels=[0, 1]))

    # ROC AUC (only defined if at least one positive and one negative sample)
    n_pos = int(np.sum(y_t == 1))
    n_neg = int(np.sum(y_t == 0))
    if n_pos > 0 and n_neg > 0:
        auc = float(roc_auc_score(y_t, y_p))
    else:
        auc = None

    # ECE
    ece = calculate_expected_calibration_error(y_t, y_p, n_bins=5)

    return {
        "n_samples": n_samples,
        "positive_count": n_pos,
        "positive_rate": float(n_pos / n_samples),
        "mean_prediction": float(np.mean(y_p)),
        "log_loss": ll,
        "brier_score": brier,
        "roc_auc": auc,
        "ece": ece,
    }
