"""Phase 3A: Controlled velocity experiments and leakage-safe nested calibration."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut

from courtiq.evaluation.metrics import (
    calculate_expected_calibration_error,
    compute_binary_classification_metrics,
)
from courtiq.models.baseline import build_baseline_pipeline

logger = logging.getLogger(__name__)

# Base feature groups
M0_NUMERIC = ["distance", "shotClock", "releaseTime", "dribblesBefore"]
M0_CATEGORICAL = ["shotType", "three", "catchAndShoot", "createdFromPaint", "period"]

M1_DEFENSIVE_NUMERIC = ["closestDefDist"]
M1_DEFENSIVE_CATEGORICAL = ["contestLevel", "contested"]

BASE_M1_NUMERIC = M0_NUMERIC + M1_DEFENSIVE_NUMERIC
BASE_M1_CATEGORICAL = M0_CATEGORICAL + M1_DEFENSIVE_CATEGORICAL

# Phase 3A Experimental Models
PHASE3A_MODELS: Dict[str, Dict[str, Any]] = {
    "M1_benchmark": {
        "numeric": BASE_M1_NUMERIC,
        "categorical": BASE_M1_CATEGORICAL,
        "description": "M1 Benchmark: release characteristics and static defensive context",
    },
    "M1_plus_speed_rel": {
        "numeric": BASE_M1_NUMERIC + ["shooter_speed_rel"],
        "categorical": BASE_M1_CATEGORICAL,
        "description": "M1 + release-lookback shooter speed [t - 0.4s, t]",
    },
    "M1_plus_speed_early": {
        "numeric": BASE_M1_NUMERIC + ["shooter_speed_early"],
        "categorical": BASE_M1_CATEGORICAL,
        "description": "M1 + earlier-window shooter speed [t - 0.8s, t - 0.4s]",
    },
    "M1_plus_directional": {
        "numeric": BASE_M1_NUMERIC + ["shooter_vel_radial", "shooter_vel_tangential"],
        "categorical": BASE_M1_CATEGORICAL,
        "description": "M1 + directional components (radial toward hoop + lateral tangential)",
    },
    "M1_plus_radial_only": {
        "numeric": BASE_M1_NUMERIC + ["shooter_vel_radial"],
        "categorical": BASE_M1_CATEGORICAL,
        "description": "Ablation: M1 + radial velocity toward/away from hoop",
    },
    "M1_plus_tangential_only": {
        "numeric": BASE_M1_NUMERIC + ["shooter_vel_tangential"],
        "categorical": BASE_M1_CATEGORICAL,
        "description": "Ablation: M1 + lateral tangential speed",
    },
}


def fit_platt_calibrator(
    inner_oof_probs: np.ndarray, y_train: np.ndarray
) -> LogisticRegression:
    """Fit a univariate logistic calibrator (Platt scaling) on inner cross-fitted predictions."""
    eps = 1e-12
    p_clipped = np.clip(inner_oof_probs, eps, 1.0 - eps)
    logit_scores = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

    # Use mild L2 regularization (C=1.0) to prevent extreme calibration coefficients
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
    calibrator.fit(logit_scores, y_train)
    return calibrator


def apply_platt_calibrator(
    calibrator: LogisticRegression, test_probs: np.ndarray
) -> np.ndarray:
    """Apply a fitted Platt calibrator to unseen test probabilities."""
    eps = 1e-12
    p_clipped = np.clip(test_probs, eps, 1.0 - eps)
    logit_scores = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logit_scores)[:, 1]


def compute_multi_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Compute primary and secondary metrics across multiple bin configurations."""
    m = compute_binary_classification_metrics(y_true, y_prob)
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1.0 - 1e-15)

    m["ece_5"] = calculate_expected_calibration_error(y_arr, p_arr, n_bins=5)
    m["ece_10"] = calculate_expected_calibration_error(y_arr, p_arr, n_bins=10)
    m["ece_15"] = calculate_expected_calibration_error(y_arr, p_arr, n_bins=15)
    return m


def run_phase3a_evaluation(
    df_merged: pd.DataFrame,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute Phase 3A Leave-One-Game-Out CV with leakage-safe nested calibration.

    Outer Loop: Leave-One-Game-Out across 10 matches.
    Inner Loop: 9-fold Leave-One-Game-Out across training matches to generate
                cross-fitted predictions strictly for calibrator fitting.
    """
    y = df_merged["outcome"].astype(int).values
    groups = df_merged["gameId"].astype(str).values
    unique_games = sorted(np.unique(groups))
    n_games = len(unique_games)

    outer_logo = LeaveOneGroupOut()
    inner_logo = LeaveOneGroupOut()

    # Out-of-fold prediction storage
    raw_oof_preds: Dict[str, np.ndarray] = {
        name: np.zeros(len(y)) for name in PHASE3A_MODELS
    }
    cal_oof_preds: Dict[str, np.ndarray] = {
        name: np.zeros(len(y)) for name in PHASE3A_MODELS
    }

    per_fold_metrics: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in PHASE3A_MODELS
    }
    calibration_provenance: List[Dict[str, Any]] = []

    logger.info("Beginning Phase 3A outer Leave-One-Game-Out evaluation...")

    for outer_fold, (train_idx, test_idx) in enumerate(
        outer_logo.split(df_merged, y, groups)
    ):
        test_game = str(groups[test_idx[0]])
        df_train_outer = df_merged.iloc[train_idx].copy()
        df_test_outer = df_merged.iloc[test_idx].copy()

        y_train_outer = y[train_idx]
        y_test_outer = y[test_idx]
        groups_train_outer = groups[train_idx]

        fold_prov: Dict[str, Any] = {
            "outer_fold": outer_fold + 1,
            "test_game": test_game,
            "n_train_shots": len(train_idx),
            "n_test_shots": len(test_idx),
            "models": {},
        }

        for model_name, cfg in PHASE3A_MODELS.items():
            num_cols = cfg["numeric"]
            cat_cols = cfg["categorical"]
            all_cols = num_cols + cat_cols

            # -------------------------------------------------------------
            # Step 1: Inner cross-validation to get cross-fitted training predictions
            # -------------------------------------------------------------
            inner_oof_probs = np.zeros(len(y_train_outer))

            for inner_train_idx, inner_val_idx in inner_logo.split(
                df_train_outer, y_train_outer, groups_train_outer
            ):
                df_in_train = df_train_outer.iloc[inner_train_idx]
                df_in_val = df_train_outer.iloc[inner_val_idx]

                y_in_train = y_train_outer[inner_train_idx]

                inner_pipe = build_baseline_pipeline(
                    numeric_features=num_cols,
                    categorical_features=cat_cols,
                    random_state=random_state,
                )
                inner_pipe.fit(df_in_train[all_cols], y_in_train)
                inner_oof_probs[inner_val_idx] = inner_pipe.predict_proba(
                    df_in_val[all_cols]
                )[:, 1]

            # -------------------------------------------------------------
            # Step 2: Fit calibrator on inner cross-fitted predictions ONLY
            # -------------------------------------------------------------
            calibrator = fit_platt_calibrator(inner_oof_probs, y_train_outer)
            cal_coef = float(calibrator.coef_[0][0])
            cal_intercept = float(calibrator.intercept_[0])

            # -------------------------------------------------------------
            # Step 3: Fit final base model on full outer training partition
            # -------------------------------------------------------------
            outer_pipe = build_baseline_pipeline(
                numeric_features=num_cols,
                categorical_features=cat_cols,
                random_state=random_state,
            )
            outer_pipe.fit(df_train_outer[all_cols], y_train_outer)

            # -------------------------------------------------------------
            # Step 4: Predict untouched outer test game and apply calibrator
            # -------------------------------------------------------------
            p_test_raw = outer_pipe.predict_proba(df_test_outer[all_cols])[:, 1]
            p_test_cal = apply_platt_calibrator(calibrator, p_test_raw)

            raw_oof_preds[model_name][test_idx] = p_test_raw
            cal_oof_preds[model_name][test_idx] = p_test_cal

            # Fold metrics
            fold_m = compute_multi_metrics(y_test_outer, p_test_raw)
            fold_m["fold"] = outer_fold + 1
            fold_m["held_out_game"] = test_game
            fold_m["calibrated_log_loss"] = compute_multi_metrics(
                y_test_outer, p_test_cal
            )["log_loss"]
            fold_m["calibrated_ece_10"] = compute_multi_metrics(
                y_test_outer, p_test_cal
            )["ece_10"]
            per_fold_metrics[model_name].append(fold_m)

            fold_prov["models"][model_name] = {
                "calibrator_slope": cal_coef,
                "calibrator_intercept": cal_intercept,
                "inner_oof_log_loss": float(
                    compute_binary_classification_metrics(
                        y_train_outer, inner_oof_probs
                    )["log_loss"]
                ),
            }

        calibration_provenance.append(fold_prov)

    # -----------------------------------------------------------------
    # Compute overall pooled metrics
    # -----------------------------------------------------------------
    pooled_raw_metrics: Dict[str, Any] = {}
    pooled_cal_metrics: Dict[str, Any] = {}

    for name in PHASE3A_MODELS:
        pooled_raw_metrics[name] = compute_multi_metrics(y, raw_oof_preds[name])
        pooled_cal_metrics[name] = compute_multi_metrics(y, cal_oof_preds[name])

    # -----------------------------------------------------------------
    # Paired cluster bootstrap across matches (B = 2,000)
    # -----------------------------------------------------------------
    rng = np.random.default_rng(random_state)
    boot_results: Dict[str, Dict[str, Any]] = {}
    m1_raw = raw_oof_preds["M1_benchmark"]

    for name in PHASE3A_MODELS:
        if name == "M1_benchmark":
            continue
        cand_raw = raw_oof_preds[name]
        boot_deltas_ll = []
        boot_deltas_brier = []
        boot_deltas_auc = []

        for _ in range(2000):
            sampled_games = rng.choice(unique_games, size=n_games, replace=True)
            idx_list = []
            for g in sampled_games:
                idx_list.extend(np.where(groups == g)[0])
            idx_arr = np.array(idx_list)

            m_base = compute_binary_classification_metrics(
                y[idx_arr], m1_raw[idx_arr]
            )
            m_cand = compute_binary_classification_metrics(
                y[idx_arr], cand_raw[idx_arr]
            )

            boot_deltas_ll.append(m_cand["log_loss"] - m_base["log_loss"])
            boot_deltas_brier.append(m_cand["brier_score"] - m_base["brier_score"])
            if m_cand["roc_auc"] and m_base["roc_auc"]:
                boot_deltas_auc.append(m_cand["roc_auc"] - m_base["roc_auc"])

        boot_deltas_ll = np.array(boot_deltas_ll)
        boot_deltas_brier = np.array(boot_deltas_brier)
        boot_deltas_auc = np.array(boot_deltas_auc)

        boot_results[name] = {
            "delta_log_loss": {
                "mean": float(np.mean(boot_deltas_ll)),
                "std": float(np.std(boot_deltas_ll)),
                "ci_95": [
                    float(np.percentile(boot_deltas_ll, 2.5)),
                    float(np.percentile(boot_deltas_ll, 97.5)),
                ],
            },
            "delta_brier": {
                "mean": float(np.mean(boot_deltas_brier)),
                "std": float(np.std(boot_deltas_brier)),
                "ci_95": [
                    float(np.percentile(boot_deltas_brier, 2.5)),
                    float(np.percentile(boot_deltas_brier, 97.5)),
                ],
            },
            "delta_roc_auc": {
                "mean": (
                    float(np.mean(boot_deltas_auc)) if len(boot_deltas_auc) else None
                ),
                "ci_95": (
                    [
                        float(np.percentile(boot_deltas_auc, 2.5)),
                        float(np.percentile(boot_deltas_auc, 97.5)),
                    ]
                    if len(boot_deltas_auc)
                    else None
                ),
            },
        }

    # -----------------------------------------------------------------
    # Per-game paired differences vs M1
    # -----------------------------------------------------------------
    per_game_summary: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in PHASE3A_MODELS if name != "M1_benchmark"
    }
    m1_per_game = per_fold_metrics["M1_benchmark"]

    for name in PHASE3A_MODELS:
        if name == "M1_benchmark":
            continue
        cand_per_game = per_fold_metrics[name]
        for i in range(n_games):
            g = unique_games[i]
            ll_base = m1_per_game[i]["log_loss"]
            ll_cand = cand_per_game[i]["log_loss"]
            br_base = m1_per_game[i]["brier_score"]
            br_cand = cand_per_game[i]["brier_score"]
            auc_base = m1_per_game[i]["roc_auc"]
            auc_cand = cand_per_game[i]["roc_auc"]

            per_game_summary[name].append({
                "game_id": g,
                "n_shots": m1_per_game[i]["n_samples"],
                "m1_log_loss": ll_base,
                "cand_log_loss": ll_cand,
                "delta_log_loss": ll_cand - ll_base,
                "improves_log_loss": bool(ll_cand < ll_base),
                "delta_brier": br_cand - br_base,
                "improves_brier": bool(br_cand < br_base),
                "delta_auc": (
                    auc_cand - auc_base if auc_cand and auc_base else None
                ),
                "improves_auc": (
                    bool(auc_cand > auc_base) if auc_cand and auc_base else None
                ),
            })

    # -----------------------------------------------------------------
    # Tracking quality sensitivity: Detected vs Extrapolated
    # -----------------------------------------------------------------
    detected_mask = df_merged["both_detected_pre10"].values.astype(bool)
    detected_subset_metrics: Dict[str, Any] = {}
    extrapolated_subset_metrics: Dict[str, Any] = {}

    for name in PHASE3A_MODELS:
        detected_subset_metrics[name] = compute_multi_metrics(
            y[detected_mask], raw_oof_preds[name][detected_mask]
        )
        extrapolated_subset_metrics[name] = compute_multi_metrics(
            y[~detected_mask], raw_oof_preds[name][~detected_mask]
        )

    return {
        "n_games": n_games,
        "n_total_shots": len(y),
        "games_list": unique_games,
        "pooled_raw_metrics": pooled_raw_metrics,
        "pooled_calibrated_metrics": pooled_cal_metrics,
        "bootstrap_results": boot_results,
        "per_game_summary": per_game_summary,
        "calibration_provenance": calibration_provenance,
        "detection_sensitivity": {
            "detected_count": int(np.sum(detected_mask)),
            "extrapolated_count": int(np.sum(~detected_mask)),
            "detected_metrics": detected_subset_metrics,
            "extrapolated_metrics": extrapolated_subset_metrics,
        },
        "raw_oof_preds": raw_oof_preds,
        "cal_oof_preds": cal_oof_preds,
        "df_merged": df_merged,
    }
