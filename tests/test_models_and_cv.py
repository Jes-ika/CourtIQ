"""Unit tests for models, game-grouped cross-validation, and metrics."""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from courtiq.evaluation.metrics import (
    calculate_expected_calibration_error,
    compute_binary_classification_metrics,
)
from courtiq.evaluation.cv import run_game_grouped_experiment
from courtiq.features.pipeline import ShotDataPipeline


def test_game_group_separation(sample_synthetic_shots_df):
    """Verify that in every fold, the held-out game ID is strictly absent from the training partition."""
    pipeline = ShotDataPipeline()
    df_clean, _ = pipeline.clean_and_filter(sample_synthetic_shots_df)
    X_m0, _, y, groups = pipeline.extract_features(df_clean)

    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(X_m0, y, groups):
        train_games = set(groups.iloc[train_idx].unique())
        test_games = set(groups.iloc[test_idx].unique())

        # No intersection between training games and test game
        assert len(train_games.intersection(test_games)) == 0
        assert len(test_games) == 1


def test_experiment_reproducibility(sample_synthetic_shots_df):
    """Verify that running the experiment twice with the same random seed yields identical metrics."""
    pipeline = ShotDataPipeline()
    df_clean, _ = pipeline.clean_and_filter(sample_synthetic_shots_df)
    X_m0, X_m1, y, groups = pipeline.extract_features(df_clean)

    m0_num = pipeline.m0_numeric
    m0_cat = [c for c in pipeline.m0_categorical if c in X_m0.columns]
    m1_num = m0_num + pipeline.m1_defensive_numeric
    m1_cat = m0_cat + [c for c in pipeline.m1_defensive_categorical if c in X_m1.columns]

    res1 = run_game_grouped_experiment(
        X_m0, X_m1, y, groups, m0_num, m0_cat, m1_num, m1_cat, random_state=42
    )
    res2 = run_game_grouped_experiment(
        X_m0, X_m1, y, groups, m0_num, m0_cat, m1_num, m1_cat, random_state=42
    )

    assert res1["overall_m0"]["log_loss"] == res2["overall_m0"]["log_loss"]
    assert res1["overall_m1"]["log_loss"] == res2["overall_m1"]["log_loss"]
    assert res1["overall_m0"]["brier_score"] == res2["overall_m0"]["brier_score"]
    assert res1["overall_m1"]["brier_score"] == res2["overall_m1"]["brier_score"]


def test_metrics_computation():
    # Perfect predictions
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.01, 0.99, 0.02, 0.98])

    metrics = compute_binary_classification_metrics(y_true, y_prob)
    assert metrics["brier_score"] < 0.01
    assert metrics["log_loss"] < 0.05
    assert metrics["roc_auc"] == 1.0

    # Single-class edge case (ROC AUC undefined)
    y_single = np.array([1, 1, 1])
    y_prob_s = np.array([0.8, 0.7, 0.9])
    metrics_s = compute_binary_classification_metrics(y_single, y_prob_s)
    assert metrics_s["roc_auc"] is None  # Undefined mathematically
    assert metrics_s["log_loss"] is not None

    # ECE calculation
    ece = calculate_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_cv_insufficient_games():
    """Verify ValueError is raised if fewer than 2 games are provided for grouped CV."""
    df_single_game = pd.DataFrame({
        "id": ["s1", "s2"],
        "gameId": [101, 101],
        "distance": [10.0, 20.0],
        "outcome": [1, 0],
    })
    with pytest.raises(ValueError, match="at least 2 distinct games"):
        run_game_grouped_experiment(
            X_m0=df_single_game[["distance"]],
            X_m1=df_single_game[["distance"]],
            y=df_single_game["outcome"],
            groups=df_single_game["gameId"],
            m0_numeric=["distance"],
            m0_categorical=[],
            m1_numeric=["distance"],
            m1_categorical=[],
        )
