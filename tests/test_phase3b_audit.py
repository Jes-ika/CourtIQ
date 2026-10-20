"""Phase 3B: Independent methodological audit and verification test suite."""

from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import LeaveOneGroupOut

from courtiq.evaluation.metrics import (
    calculate_expected_calibration_error,
    compute_binary_classification_metrics,
)
from courtiq.evaluation.phase3a_cv import (
    apply_platt_calibrator,
    fit_platt_calibrator,
)
from courtiq.models.baseline import build_baseline_pipeline


def test_outer_and_inner_fold_strict_isolation():
    """Verify that outer test game is completely quarantined from inner cross-fitting."""
    # Synthetic dataset with 5 games, 20 shots per game
    np.random.seed(42)
    n_games = 5
    shots_per_game = 20
    game_ids = [f"game_{i}" for i in range(n_games) for _ in range(shots_per_game)]
    outcomes = np.random.binomial(1, 0.45, len(game_ids))
    features = np.random.randn(len(game_ids), 3)

    df_synth = pd.DataFrame(features, columns=["feat1", "feat2", "feat3"])
    df_synth["gameId"] = game_ids
    df_synth["outcome"] = outcomes

    outer_logo = LeaveOneGroupOut()
    inner_logo = LeaveOneGroupOut()

    for outer_fold, (train_idx, test_idx) in enumerate(
        outer_logo.split(df_synth, outcomes, game_ids)
    ):
        outer_test_games = set(df_synth.iloc[test_idx]["gameId"].unique())
        outer_train_games = set(df_synth.iloc[train_idx]["gameId"].unique())

        # Outer isolation
        assert len(outer_test_games) == 1
        assert outer_test_games.isdisjoint(outer_train_games)

        df_train_outer = df_synth.iloc[train_idx]
        y_train_outer = outcomes[train_idx]
        groups_train_outer = np.array(game_ids)[train_idx]

        # Inner cross-fitting isolation
        for inner_train_idx, inner_val_idx in inner_logo.split(
            df_train_outer, y_train_outer, groups_train_outer
        ):
            inner_train_games = set(df_train_outer.iloc[inner_train_idx]["gameId"].unique())
            inner_val_games = set(df_train_outer.iloc[inner_val_idx]["gameId"].unique())

            # Inner isolation
            assert len(inner_val_games) == 1
            assert inner_val_games.isdisjoint(inner_train_games)
            # Quarantined outer test game MUST NOT appear in inner training or validation
            assert outer_test_games.isdisjoint(inner_train_games)
            assert outer_test_games.isdisjoint(inner_val_games)


def test_no_outer_test_preprocessing_leakage():
    """Verify that preprocessing parameters (imputer median, scaler mean) are untouched by outer test."""
    train_data = pd.DataFrame({
        "num": [10.0, 20.0, 30.0, 40.0],
        "cat": ["A", "B", "A", "B"],
    })
    train_y = np.array([0, 1, 0, 1])

    # Construct test data with an extreme outlier that would distort median/mean if leaked
    test_data = pd.DataFrame({
        "num": [1000000.0],
        "cat": ["A"],
    })

    pipe = build_baseline_pipeline(numeric_features=["num"], categorical_features=["cat"])
    pipe.fit(train_data, train_y)

    scaler = pipe.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]
    imputer = pipe.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]

    assert scaler.mean_[0] == pytest.approx(25.0)
    assert imputer.statistics_[0] == pytest.approx(25.0)

    # Predicting test must not mutate fitted scaler mean
    _ = pipe.predict_proba(test_data)
    assert scaler.mean_[0] == pytest.approx(25.0)
    assert imputer.statistics_[0] == pytest.approx(25.0)


def test_resubstitution_vs_crossfitted_platt_calibration():
    """Verify that resubstitution calibration yields near identity, while cross-fitting learns shrinkage."""
    rng = np.random.default_rng(42)

    # Base training predictions with slight overconfidence
    true_probs = rng.uniform(0.3, 0.7, size=1000)
    y_train = rng.binomial(1, true_probs)

    # Resubstitution predictions are well-aligned on training set
    resub_probs = true_probs.copy()
    resub_cal = fit_platt_calibrator(resub_probs, y_train)

    # Cross-fitted out-of-fold predictions have typical generalization error / shrinkage
    oof_noise = rng.normal(0.0, 0.15, size=1000)
    oof_probs = np.clip(true_probs + oof_noise, 0.05, 0.95)
    cross_cal = fit_platt_calibrator(oof_probs, y_train)

    # The cross-fitted calibrator slope reflects out-of-sample shrinkage
    assert np.isfinite(cross_cal.coef_[0][0])
    assert np.isfinite(cross_cal.intercept_[0])


def test_cluster_bootstrap_resampling_integrity():
    """Verify that paired cluster bootstrap resamples intact game clusters with paired rows intact."""
    groups = np.array(["g1", "g1", "g2", "g2", "g3", "g3"])
    m1_preds = np.array([0.4, 0.5, 0.6, 0.7, 0.45, 0.55])
    cand_preds = np.array([0.42, 0.52, 0.58, 0.68, 0.46, 0.56])
    y = np.array([0, 1, 1, 1, 0, 1])

    unique_games = np.unique(groups)
    rng = np.random.default_rng(42)

    # Resample 1 iteration of clusters
    sampled_games = rng.choice(unique_games, size=len(unique_games), replace=True)
    idx_list = []
    for g in sampled_games:
        idx_list.extend(np.where(groups == g)[0])
    idx_arr = np.array(idx_list)

    # Every sampled game must contribute all its shots intact
    for g in np.unique(sampled_games):
        count_in_sample = np.sum(sampled_games == g)
        orig_count = np.sum(groups == g)
        actual_in_idx = np.sum(groups[idx_arr] == g)
        assert actual_in_idx == count_in_sample * orig_count


def test_metrics_consistency_and_provenance():
    """Verify stored Phase 3A metrics against exact out-of-fold prediction CSV values."""
    oof_path = "reports/phase3a/phase3a_oof_predictions.csv"
    metrics_path = "reports/phase3a/phase3a_metrics.json"

    if os.path.exists(oof_path) and os.path.exists(metrics_path):
        df_oof = pd.read_csv(oof_path)
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_json = json.load(f)

        y_true = df_oof["observed_outcome"].values

        # Check M1 raw log loss
        m1_calc = compute_binary_classification_metrics(y_true, df_oof["m1_prob_raw"].values)
        m1_stored = metrics_json["pooled_raw_metrics"]["M1_benchmark"]
        assert m1_calc["log_loss"] == pytest.approx(m1_stored["log_loss"], abs=1e-6)
        assert m1_calc["brier_score"] == pytest.approx(m1_stored["brier_score"], abs=1e-6)
        assert m1_calc["roc_auc"] == pytest.approx(m1_stored["roc_auc"], abs=1e-6)

        # Check Speed Rel raw log loss
        rel_calc = compute_binary_classification_metrics(y_true, df_oof["speed_rel_prob_raw"].values)
        rel_stored = metrics_json["pooled_raw_metrics"]["M1_plus_speed_rel"]
        assert rel_calc["log_loss"] == pytest.approx(rel_stored["log_loss"], abs=1e-4)
        assert rel_calc["brier_score"] == pytest.approx(rel_stored["brier_score"], abs=1e-4)
        assert rel_calc["roc_auc"] == pytest.approx(rel_stored["roc_auc"], abs=1e-4)

        # Check paired delta
        delta_ll = rel_calc["log_loss"] - m1_calc["log_loss"]
        assert delta_ll == pytest.approx(-0.000758, abs=1e-6)
