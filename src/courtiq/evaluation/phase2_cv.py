"""Phase 2 evaluation and cross-validation comparing M0, M1, M2, and kinematic ablations."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut

from courtiq.evaluation.metrics import (
    calculate_expected_calibration_error,
    compute_binary_classification_metrics,
)
from courtiq.models.baseline import build_baseline_pipeline

logger = logging.getLogger(__name__)

# Feature group specifications
M0_NUMERIC = ["distance", "shotClock", "releaseTime", "dribblesBefore"]
M0_CATEGORICAL = ["shotType", "three", "catchAndShoot", "createdFromPaint", "period"]

M1_DEFENSIVE_NUMERIC = ["closestDefDist"]
M1_DEFENSIVE_CATEGORICAL = ["contestLevel", "contested"]

M2_KINEMATIC_NUMERIC = [
    "defender_closing_velocity",
    "contest_angle_cos",
    "shooter_velocity_magnitude",
]

# Ablation feature configurations
MODEL_CONFIGS: Dict[str, Dict[str, List[str]]] = {
    "M0_baseline": {
        "numeric": M0_NUMERIC,
        "categorical": M0_CATEGORICAL,
        "description": "Baseline: shot location and basic shot characteristics",
    },
    "M1_context": {
        "numeric": M0_NUMERIC + M1_DEFENSIVE_NUMERIC,
        "categorical": M0_CATEGORICAL + M1_DEFENSIVE_CATEGORICAL,
        "description": "Release Context: M0 + static defender distance and contest level",
    },
    "M2_full": {
        "numeric": M0_NUMERIC + M1_DEFENSIVE_NUMERIC + M2_KINEMATIC_NUMERIC,
        "categorical": M0_CATEGORICAL + M1_DEFENSIVE_CATEGORICAL,
        "description": "Full M2: M1 + closing velocity + contest angle + shooter velocity",
    },
    "M2_ablation_closing_only": {
        "numeric": M0_NUMERIC + M1_DEFENSIVE_NUMERIC + ["defender_closing_velocity"],
        "categorical": M0_CATEGORICAL + M1_DEFENSIVE_CATEGORICAL,
        "description": "Ablation: M1 + defender closing velocity",
    },
    "M2_ablation_angle_only": {
        "numeric": M0_NUMERIC + M1_DEFENSIVE_NUMERIC + ["contest_angle_cos"],
        "categorical": M0_CATEGORICAL + M1_DEFENSIVE_CATEGORICAL,
        "description": "Ablation: M1 + contest angle cosine",
    },
    "M2_ablation_shooter_vel_only": {
        "numeric": M0_NUMERIC + M1_DEFENSIVE_NUMERIC + ["shooter_velocity_magnitude"],
        "categorical": M0_CATEGORICAL + M1_DEFENSIVE_CATEGORICAL,
        "description": "Ablation: M1 + shooter velocity magnitude",
    },
    "M2_ablation_closing_angle": {
        "numeric": M0_NUMERIC
        + M1_DEFENSIVE_NUMERIC
        + ["defender_closing_velocity", "contest_angle_cos"],
        "categorical": M0_CATEGORICAL + M1_DEFENSIVE_CATEGORICAL,
        "description": "Ablation: M1 + closing velocity + contest angle",
    },
    "M2_ablation_delta_dist": {
        "numeric": M0_NUMERIC
        + M1_DEFENSIVE_NUMERIC
        + ["delta_defender_dist", "contest_angle_cos", "shooter_velocity_magnitude"],
        "categorical": M0_CATEGORICAL + M1_DEFENSIVE_CATEGORICAL,
        "description": "Alternative M2: M1 + distance delta + contest angle + shooter velocity",
    },
}


def fit_and_predict_platt_calibration(
    train_scores: np.ndarray,
    train_y: np.ndarray,
    test_scores: np.ndarray,
) -> np.ndarray:
    """Fit a univariate logistic calibrator on train log-odds and transform test log-odds."""
    # Clip scores to prevent infinite log-odds
    eps = 1e-12
    p_tr = np.clip(train_scores, eps, 1.0 - eps)
    p_te = np.clip(test_scores, eps, 1.0 - eps)

    # Convert to log-odds
    logit_tr = np.log(p_tr / (1.0 - p_tr)).reshape(-1, 1)
    logit_te = np.log(p_te / (1.0 - p_te)).reshape(-1, 1)

    calibrator = LogisticRegression(C=1e5, solver="lbfgs", max_iter=500)
    calibrator.fit(logit_tr, train_y)
    cal_probs = calibrator.predict_proba(logit_te)[:, 1]
    return cal_probs


def compute_comprehensive_metrics(
    y_true: np.ndarray | pd.Series, y_prob: np.ndarray
) -> Dict[str, Any]:
    """Compute log loss, Brier score, ROC-AUC, and multi-bin ECEs."""
    base_metrics = compute_binary_classification_metrics(y_true, y_prob)
    y_arr = np.asarray(y_true, dtype=int)
    prob_arr = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1.0 - 1e-15)

    base_metrics["ece_5"] = calculate_expected_calibration_error(y_arr, prob_arr, n_bins=5)
    base_metrics["ece_10"] = calculate_expected_calibration_error(y_arr, prob_arr, n_bins=10)
    base_metrics["ece_15"] = calculate_expected_calibration_error(y_arr, prob_arr, n_bins=15)
    return base_metrics


def run_phase2_experiment(
    df_shots: pd.DataFrame,
    df_kinematics: pd.DataFrame,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run Phase 2 Leave-One-Game-Out cross-validation comparing M0, M1, M2, and ablations.

    Args:
        df_shots: Cleaned shot DataFrame containing M0/M1 columns and 'outcome', 'id', 'gameId'.
        df_kinematics: Kinematics DataFrame containing pre-release features and tracking flags.
        random_state: Random state for reproducibility.

    Returns:
        Dictionary containing all pooled, per-game, ablation, and calibration results.
    """
    # Merge kinematic features onto cleaned shots ensuring exact row matching
    merged_df = df_shots.merge(
        df_kinematics[
            [
                "id",
                "defender_closing_velocity",
                "delta_defender_dist",
                "contest_angle_cos",
                "contest_angle_deg",
                "shooter_velocity_magnitude",
                "both_detected_pre",
                "shooter_detected_pre",
                "defender_detected_pre",
                "shooter_pred_error_pre",
                "defender_pred_error_pre",
            ]
        ],
        on="id",
        how="inner",
    )

    if len(merged_df) != len(df_shots):
        raise ValueError(
            f"Row count mismatch after merging kinematics: {len(merged_df)} vs {len(df_shots)}"
        )

    y = merged_df["outcome"].astype(int).values
    groups = merged_df["gameId"].astype(str).values
    unique_games = sorted(np.unique(groups))
    n_games = len(unique_games)

    logo = LeaveOneGroupOut()

    # Dictionary to store out-of-fold predictions
    oof_preds: Dict[str, np.ndarray] = {
        model_name: np.zeros(len(y)) for model_name in MODEL_CONFIGS
    }
    oof_cal_preds: Dict[str, np.ndarray] = {
        model_name: np.zeros(len(y)) for model_name in MODEL_CONFIGS
    }

    fold_metrics_by_model: Dict[str, List[Dict[str, Any]]] = {
        model_name: [] for model_name in MODEL_CONFIGS
    }

    # Iterate over LOGO folds
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(merged_df, y, groups)):
        held_out_game = groups[test_idx[0]]
        y_train, y_test = y[train_idx], y[test_idx]

        df_train = merged_df.iloc[train_idx]
        df_test = merged_df.iloc[test_idx]

        for model_name, cfg in MODEL_CONFIGS.items():
            num_cols = cfg["numeric"]
            cat_cols = cfg["categorical"]

            # Build pipeline strictly inside fold
            pipe = build_baseline_pipeline(
                numeric_features=num_cols,
                categorical_features=cat_cols,
                random_state=random_state,
            )

            # Fit on training fold
            pipe.fit(df_train[num_cols + cat_cols], y_train)

            # Raw uncalibrated predictions
            p_train = pipe.predict_proba(df_train[num_cols + cat_cols])[:, 1]
            p_test = pipe.predict_proba(df_test[num_cols + cat_cols])[:, 1]

            oof_preds[model_name][test_idx] = p_test

            # In-fold Platt calibration
            p_test_cal = fit_and_predict_platt_calibration(p_train, y_train, p_test)
            oof_cal_preds[model_name][test_idx] = p_test_cal

            # Fold-level metrics
            fold_m = compute_comprehensive_metrics(y_test, p_test)
            fold_m["fold"] = fold_idx + 1
            fold_m["held_out_game"] = str(held_out_game)
            fold_metrics_by_model[model_name].append(fold_m)

    # Compute overall pooled metrics
    overall_results: Dict[str, Any] = {}
    overall_cal_results: Dict[str, Any] = {}

    for model_name in MODEL_CONFIGS:
        overall_results[model_name] = compute_comprehensive_metrics(
            y, oof_preds[model_name]
        )
        overall_cal_results[model_name] = compute_comprehensive_metrics(
            y, oof_cal_preds[model_name]
        )

    # Detection quality breakdown: detected vs extrapolated
    is_detected_mask = merged_df["both_detected_pre"].values.astype(bool)
    detected_metrics: Dict[str, Any] = {}
    extrapolated_metrics: Dict[str, Any] = {}

    for model_name in ["M0_baseline", "M1_context", "M2_full"]:
        detected_metrics[model_name] = compute_comprehensive_metrics(
            y[is_detected_mask], oof_preds[model_name][is_detected_mask]
        )
        extrapolated_metrics[model_name] = compute_comprehensive_metrics(
            y[~is_detected_mask], oof_preds[model_name][~is_detected_mask]
        )

    # Per-game deltas relative to M1
    m1_folds = fold_metrics_by_model["M1_context"]
    m2_folds = fold_metrics_by_model["M2_full"]
    m0_folds = fold_metrics_by_model["M0_baseline"]

    per_game_comparisons = []
    for i in range(n_games):
        g = m1_folds[i]["held_out_game"]
        n_s = m1_folds[i]["n_samples"]

        ll_m0 = m0_folds[i]["log_loss"]
        ll_m1 = m1_folds[i]["log_loss"]
        ll_m2 = m2_folds[i]["log_loss"]

        br_m0 = m0_folds[i]["brier_score"]
        br_m1 = m1_folds[i]["brier_score"]
        br_m2 = m2_folds[i]["brier_score"]

        auc_m0 = m0_folds[i]["roc_auc"]
        auc_m1 = m1_folds[i]["roc_auc"]
        auc_m2 = m2_folds[i]["roc_auc"]

        per_game_comparisons.append({
            "game_id": g,
            "n_shots": n_s,
            "m0_log_loss": ll_m0,
            "m1_log_loss": ll_m1,
            "m2_log_loss": ll_m2,
            "delta_m2_minus_m1_ll": ll_m2 - ll_m1,
            "m2_improves_ll": bool(ll_m2 < ll_m1),
            "m0_brier": br_m0,
            "m1_brier": br_m1,
            "m2_brier": br_m2,
            "delta_m2_minus_m1_brier": br_m2 - br_m1,
            "m2_improves_brier": bool(br_m2 < br_m1),
            "m0_auc": auc_m0,
            "m1_auc": auc_m1,
            "m2_auc": auc_m2,
            "delta_m2_minus_m1_auc": (
                auc_m2 - auc_m1 if auc_m2 and auc_m1 else None
            ),
            "m2_improves_auc": bool(auc_m2 > auc_m1) if auc_m2 and auc_m1 else None,
        })

    # Summary counts of per-game improvement
    m2_ll_improves_count = sum(1 for c in per_game_comparisons if c["m2_improves_ll"])
    m2_brier_improves_count = sum(
        1 for c in per_game_comparisons if c["m2_improves_brier"]
    )
    m2_auc_improves_count = sum(
        1 for c in per_game_comparisons if c["m2_improves_auc"] is True
    )

    # Cluster bootstrap for delta log loss (resampling games)
    rng = np.random.default_rng(random_state)
    boot_deltas = []
    for _ in range(2000):
        sampled_games = rng.choice(unique_games, size=n_games, replace=True)
        # Gather all shots from sampled games
        idx_list = []
        for g in sampled_games:
            idx_list.extend(np.where(groups == g)[0])
        idx_arr = np.array(idx_list)

        ll_m1_b = compute_binary_classification_metrics(
            y[idx_arr], oof_preds["M1_context"][idx_arr]
        )["log_loss"]
        ll_m2_b = compute_binary_classification_metrics(
            y[idx_arr], oof_preds["M2_full"][idx_arr]
        )["log_loss"]
        boot_deltas.append(ll_m2_b - ll_m1_b)

    boot_deltas = np.array(boot_deltas)
    boot_ci = [
        float(np.percentile(boot_deltas, 2.5)),
        float(np.percentile(boot_deltas, 97.5)),
    ]

    return {
        "n_games": n_games,
        "n_shots": len(y),
        "games_list": unique_games,
        "overall_uncalibrated": overall_results,
        "overall_calibrated": overall_cal_results,
        "per_game_comparisons": per_game_comparisons,
        "m2_improvement_counts": {
            "log_loss_improved_games": m2_ll_improves_count,
            "log_loss_degraded_games": n_games - m2_ll_improves_count,
            "brier_improved_games": m2_brier_improves_count,
            "brier_degraded_games": n_games - m2_brier_improves_count,
            "auc_improved_games": m2_auc_improves_count,
            "auc_degraded_games": n_games - m2_auc_improves_count,
        },
        "bootstrap_m2_minus_m1_delta_log_loss": {
            "mean": float(np.mean(boot_deltas)),
            "std": float(np.std(boot_deltas)),
            "ci_95": boot_ci,
        },
        "detection_stratified_evaluation": {
            "detected_subset_n": int(np.sum(is_detected_mask)),
            "extrapolated_subset_n": int(np.sum(~is_detected_mask)),
            "detected_metrics": detected_metrics,
            "extrapolated_metrics": extrapolated_metrics,
        },
        "oof_predictions": oof_preds,
        "oof_calibrated_predictions": oof_cal_preds,
        "merged_dataframe": merged_df,
    }
