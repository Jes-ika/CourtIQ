"""Regression and audit verification tests for CourtIQ Phase 1.5."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import LeaveOneGroupOut

from courtiq.evaluation.metrics import (
    calculate_expected_calibration_error,
    compute_binary_classification_metrics,
)
from courtiq.evaluation.cv import run_game_grouped_experiment
from courtiq.features.pipeline import ShotDataPipeline


def test_oof_predictions_bounds_and_alignment(sample_synthetic_shots_df):
    """Verify that OOF predictions are bounded, aligned, and have identical length."""
    pipeline = ShotDataPipeline()
    df_clean, _ = pipeline.clean_and_filter(sample_synthetic_shots_df)
    X_m0, X_m1, y, groups = pipeline.extract_features(df_clean)

    res = run_game_grouped_experiment(
        X_m0=X_m0,
        X_m1=X_m1,
        y=y,
        groups=groups,
        m0_numeric=pipeline.m0_numeric,
        m0_categorical=[c for c in pipeline.m0_categorical if c in X_m0.columns],
        m1_numeric=pipeline.m0_numeric + pipeline.m1_defensive_numeric,
        m1_categorical=[c for c in pipeline.m0_categorical if c in X_m0.columns]
        + [c for c in pipeline.m1_defensive_categorical if c in X_m1.columns],
        random_state=42,
    )

    oof_m0 = res["oof_m0_preds"]
    oof_m1 = res["oof_m1_preds"]

    # Must match sample count exactly
    assert len(oof_m0) == len(df_clean)
    assert len(oof_m1) == len(df_clean)

    # Probabilities must be strictly between 0 and 1
    assert np.all(oof_m0 >= 0.0) and np.all(oof_m0 <= 1.0)
    assert np.all(oof_m1 >= 0.0) and np.all(oof_m1 <= 1.0)

    # Predictions must not contain NaNs
    assert not np.isnan(oof_m0).any()
    assert not np.isnan(oof_m1).any()


def test_calibration_ece_bounds():
    """Verify Expected Calibration Error (ECE) satisfies theoretical properties."""
    # Perfectly calibrated case: 50% ones with predicted prob 0.5
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    ece_zero = calculate_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert pytest.approx(ece_zero, abs=1e-6) == 0.0

    # Completely uncalibrated case: all zeros with predicted prob 1.0
    y_zeros = np.zeros(10, dtype=int)
    y_ones = np.ones(10, dtype=float)
    ece_max = calculate_expected_calibration_error(y_zeros, y_ones, n_bins=5)
    assert pytest.approx(ece_max, abs=1e-6) == 1.0


def test_held_out_game_strict_isolation(sample_synthetic_shots_df):
    """Verify that every test game is completely absent from training partition."""
    pipeline = ShotDataPipeline()
    df_clean, _ = pipeline.clean_and_filter(sample_synthetic_shots_df)
    X_m0, _, y, groups = pipeline.extract_features(df_clean)

    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(X_m0, y, groups):
        train_g = set(groups.iloc[train_idx])
        test_g = set(groups.iloc[test_idx])
        assert train_g.isdisjoint(test_g), f"Leakage detected! {train_g.intersection(test_g)}"
        assert len(test_g) == 1
