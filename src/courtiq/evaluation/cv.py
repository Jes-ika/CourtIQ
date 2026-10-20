"""Game-grouped cross-validation and paired statistical hypothesis testing."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import LeaveOneGroupOut

from courtiq.evaluation.metrics import compute_binary_classification_metrics
from courtiq.models.baseline import build_baseline_pipeline

logger = logging.getLogger(__name__)


def compute_t_distribution_ci(
    values: List[float], confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Compute mean, standard error, and confidence interval using Student's t-distribution.

    Appropriate for small numbers of independent games (e.g. n=10).
    """
    valid_vals = [v for v in values if v is not None and not np.isnan(v)]
    n = len(valid_vals)
    if n < 2:
        mean_val = valid_vals[0] if n == 1 else 0.0
        return float(mean_val), 0.0, 0.0

    mean = float(np.mean(valid_vals))
    std = float(np.std(valid_vals, ddof=1))
    se = float(std / np.sqrt(n))
    t_crit = float(stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
    margin = t_crit * se
    return mean, se, margin


def run_game_grouped_experiment(
    X_m0: pd.DataFrame,
    X_m1: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    m0_numeric: List[str],
    m0_categorical: List[str],
    m1_numeric: List[str],
    m1_categorical: List[str],
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run rigorous LeaveOneGroupOut cross-validation comparing M0 Baseline and M1 Context Model.

    Every observation from a held-out game is excluded from training.
    Evaluates both models on identical held-out folds.
    """
    unique_games = sorted(groups.unique())
    n_games = len(unique_games)

    if n_games < 2:
        raise ValueError(
            f"Game-grouped validation requires at least 2 distinct games, found {n_games}."
        )

    logo = LeaveOneGroupOut()

    m0_oof_preds = np.zeros(len(y))
    m1_oof_preds = np.zeros(len(y))

    fold_results: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_m0, y, groups)):
        held_out_game = groups.iloc[test_idx[0]]
        test_size = len(test_idx)

        # Train & test slices
        X_m0_train, X_m0_test = X_m0.iloc[train_idx], X_m0.iloc[test_idx]
        X_m1_train, X_m1_test = X_m1.iloc[train_idx], X_m1.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Build and fit M0
        m0_model = build_baseline_pipeline(
            numeric_features=m0_numeric,
            categorical_features=m0_categorical,
            random_state=random_state,
        )
        m0_model.fit(X_m0_train, y_train)
        pred_m0_test = m0_model.predict_proba(X_m0_test)[:, 1]
        m0_oof_preds[test_idx] = pred_m0_test

        # Build and fit M1
        m1_model = build_baseline_pipeline(
            numeric_features=m1_numeric,
            categorical_features=m1_categorical,
            random_state=random_state,
        )
        m1_model.fit(X_m1_train, y_train)
        pred_m1_test = m1_model.predict_proba(X_m1_test)[:, 1]
        m1_oof_preds[test_idx] = pred_m1_test

        # Metrics for held-out fold
        m0_metrics = compute_binary_classification_metrics(y_test, pred_m0_test)
        m1_metrics = compute_binary_classification_metrics(y_test, pred_m1_test)

        delta_log_loss = (
            (m1_metrics["log_loss"] - m0_metrics["log_loss"])
            if m0_metrics["log_loss"] is not None and m1_metrics["log_loss"] is not None
            else None
        )
        delta_brier = (
            (m1_metrics["brier_score"] - m0_metrics["brier_score"])
            if m0_metrics["brier_score"] is not None and m1_metrics["brier_score"] is not None
            else None
        )
        delta_auc = (
            (m1_metrics["roc_auc"] - m0_metrics["roc_auc"])
            if m0_metrics["roc_auc"] is not None and m1_metrics["roc_auc"] is not None
            else None
        )

        fold_results.append({
            "fold": fold_idx + 1,
            "held_out_game": str(held_out_game),
            "n_shots": test_size,
            "m0_log_loss": m0_metrics["log_loss"],
            "m1_log_loss": m1_metrics["log_loss"],
            "delta_log_loss": delta_log_loss,
            "m0_brier": m0_metrics["brier_score"],
            "m1_brier": m1_metrics["brier_score"],
            "delta_brier": delta_brier,
            "m0_auc": m0_metrics["roc_auc"],
            "m1_auc": m1_metrics["roc_auc"],
            "delta_auc": delta_auc,
            "m0_ece": m0_metrics["ece"],
            "m1_ece": m1_metrics["ece"],
        })

    # Overall pooled out-of-fold metrics across all shots
    overall_m0 = compute_binary_classification_metrics(y, m0_oof_preds)
    overall_m1 = compute_binary_classification_metrics(y, m1_oof_preds)

    overall_delta_log_loss = overall_m1["log_loss"] - overall_m0["log_loss"]
    overall_delta_brier = overall_m1["brier_score"] - overall_m0["brier_score"]
    overall_delta_auc = (
        (overall_m1["roc_auc"] - overall_m0["roc_auc"])
        if overall_m1["roc_auc"] is not None and overall_m0["roc_auc"] is not None
        else None
    )

    # Uncertainty intervals across fold deltas using Student's t-distribution
    delta_ll_values = [f["delta_log_loss"] for f in fold_results if f["delta_log_loss"] is not None]
    delta_brier_values = [f["delta_brier"] for f in fold_results if f["delta_brier"] is not None]
    delta_auc_values = [f["delta_auc"] for f in fold_results if f["delta_auc"] is not None]

    mean_dll, se_dll, margin_dll = compute_t_distribution_ci(delta_ll_values)
    mean_dbrier, se_dbrier, margin_dbrier = compute_t_distribution_ci(delta_brier_values)
    mean_dauc, se_dauc, margin_dauc = compute_t_distribution_ci(delta_auc_values)

    # Paired t-tests on fold differences
    pval_dll = float(stats.ttest_1samp(delta_ll_values, 0.0).pvalue) if len(delta_ll_values) > 1 else None
    pval_dbrier = float(stats.ttest_1samp(delta_brier_values, 0.0).pvalue) if len(delta_brier_values) > 1 else None
    pval_dauc = float(stats.ttest_1samp(delta_auc_values, 0.0).pvalue) if len(delta_auc_values) > 1 else None

    return {
        "n_games": n_games,
        "n_total_shots": len(y),
        "games_list": [str(g) for g in unique_games],
        "overall_m0": overall_m0,
        "overall_m1": overall_m1,
        "overall_deltas": {
            "delta_log_loss": overall_delta_log_loss,
            "delta_brier": overall_delta_brier,
            "delta_roc_auc": overall_delta_auc,
        },
        "fold_level_summary": {
            "delta_log_loss": {
                "mean": mean_dll,
                "se": se_dll,
                "ci_95": [mean_dll - margin_dll, mean_dll + margin_dll],
                "p_value": pval_dll,
            },
            "delta_brier": {
                "mean": mean_dbrier,
                "se": se_dbrier,
                "ci_95": [mean_dbrier - margin_dbrier, mean_dbrier + margin_dbrier],
                "p_value": pval_dbrier,
            },
            "delta_roc_auc": {
                "mean": mean_dauc,
                "se": se_dauc,
                "ci_95": [mean_dauc - margin_dauc, mean_dauc + margin_dauc],
                "p_value": pval_dauc,
            },
        },
        "folds": fold_results,
        "oof_m0_preds": m0_oof_preds,
        "oof_m1_preds": m1_oof_preds,
    }
