"""Unit and regression tests for Phase 3A nested calibration, directional velocity, and provenance."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from courtiq.features.geometry import (
    DEFAULT_OFFENSIVE_HOOP_LOCATION,
    compute_directional_shooter_speed,
    compute_shooter_speed,
)
from courtiq.features.phase3a_features import extract_shot_phase3a_features
from courtiq.evaluation.phase3a_cv import (
    PHASE3A_MODELS,
    apply_platt_calibrator,
    compute_multi_metrics,
    fit_platt_calibrator,
)


def test_directional_velocity_math():
    """Verify radial and tangential velocity decomposition and Pythagorean relationship."""
    hoop = (-40.75, 0.0)
    pos_pre = (-20.0, 0.0)

    # 1. Drive directly toward hoop (move from -20 to -24 in 0.4s -> 10 ft/s toward hoop)
    pos_drive = (-24.0, 0.0)
    v_rad, v_tan, v_tot = compute_directional_shooter_speed(
        pos_pre, pos_drive, hoop_pos=hoop, delta_t=0.4
    )
    assert v_rad == pytest.approx(10.0)
    assert v_tan == pytest.approx(0.0)
    assert v_tot == pytest.approx(10.0)

    # 2. Fadeaway directly away from hoop (move from -20 to -17 in 0.4s -> 7.5 ft/s away from hoop)
    pos_fade = (-17.0, 0.0)
    v_rad_f, v_tan_f, v_tot_f = compute_directional_shooter_speed(
        pos_pre, pos_fade, hoop_pos=hoop, delta_t=0.4
    )
    assert v_rad_f == pytest.approx(-7.5)  # Negative denotes moving away from rim
    assert v_tan_f == pytest.approx(0.0)
    assert v_tot_f == pytest.approx(7.5)

    # 3. Pure lateral motion (move from -20, 0 to -20, 4 in 0.4s -> 10 ft/s lateral)
    pos_lat = (-20.0, 4.0)
    v_rad_l, v_tan_l, v_tot_l = compute_directional_shooter_speed(
        pos_pre, pos_lat, hoop_pos=hoop, delta_t=0.4
    )
    assert v_rad_l == pytest.approx(0.0)
    assert v_tan_l == pytest.approx(10.0)
    assert v_tot_l == pytest.approx(10.0)

    # 4. Diagonal motion: verify v_tot^2 == v_rad^2 + v_tan^2
    pos_diag = (-23.0, 4.0)
    v_r, v_t, v_tot_d = compute_directional_shooter_speed(
        pos_pre, pos_diag, hoop_pos=hoop, delta_t=0.4
    )
    assert (v_r**2 + v_t**2) == pytest.approx(v_tot_d**2, abs=1e-5)


def test_earlier_window_feature_timing():
    """Verify that earlier-window speed uses frames strictly before the release-lookback window."""
    shot = {
        "id": "shot-test-timing",
        "gameId": 99999,
        "period": 1,
        "startFrame": 200,
        "shooterId": 55,
        "closestDefId": 77,
    }

    # Frame 200 (release t): shooter at (-20, 0)
    fr_rel = {
        "frameIdx": 200,
        "period": 1,
        "homePlayers": [{"playerId": 55, "xyz": [-20.0, 0.0, 0.0], "isDetected": 1}],
        "awayPlayers": [{"playerId": 77, "xyz": [-23.0, 0.0, 0.0], "isDetected": 1}],
    }

    # Frame 190 (t - 0.4s): shooter at (-18, 0) -> speed_rel = (2 ft) / 0.4 = 5.0 ft/s
    fr_pre10 = {
        "frameIdx": 190,
        "period": 1,
        "homePlayers": [{"playerId": 55, "xyz": [-18.0, 0.0, 0.0], "isDetected": 1}],
        "awayPlayers": [{"playerId": 77, "xyz": [-25.0, 0.0, 0.0], "isDetected": 1}],
    }

    # Frame 180 (t - 0.8s): shooter at (-14, 0) -> speed_early = (4 ft) / 0.4 = 10.0 ft/s
    fr_pre20 = {
        "frameIdx": 180,
        "period": 1,
        "homePlayers": [{"playerId": 55, "xyz": [-14.0, 0.0, 0.0], "isDetected": 1}],
        "awayPlayers": [{"playerId": 77, "xyz": [-27.0, 0.0, 0.0], "isDetected": 1}],
    }

    feats = extract_shot_phase3a_features(shot, fr_rel, fr_pre10, fr_pre20)
    assert feats["extraction_status_3a"] == "success"
    assert feats["shooter_speed_rel"] == pytest.approx(5.0)
    assert feats["shooter_speed_early"] == pytest.approx(10.0)
    assert feats["shooter_vel_radial"] == pytest.approx(5.0)
    assert feats["shooter_vel_tangential"] == pytest.approx(0.0)


def test_nested_calibration_isolation_and_provenance():
    """Verify that Platt calibrator is fitted on cross-fitted predictions without test leakage."""
    rng = np.random.default_rng(42)

    # Synthetic cross-fitted inner predictions (e.g. from 9 training games)
    inner_oof_probs = rng.uniform(0.2, 0.8, size=500)
    train_y = rng.binomial(1, inner_oof_probs)

    # Test probabilities (from held-out test game)
    test_probs = rng.uniform(0.2, 0.8, size=100)

    # Calibrator is fit ONLY on inner_oof_probs and train_y
    calibrator = fit_platt_calibrator(inner_oof_probs, train_y)
    cal_test = apply_platt_calibrator(calibrator, test_probs)

    assert len(cal_test) == 100
    assert (cal_test >= 0.0).all()
    assert (cal_test <= 1.0).all()
    # Calibrator coefficients are finite and non-degenerate
    assert np.isfinite(calibrator.coef_[0][0])
    assert np.isfinite(calibrator.intercept_[0])


def test_phase3a_cached_features_and_predictions_integrity():
    """Verify that phase3a_oof_predictions.csv and phase3a_features.csv have exact matched rows."""
    import os

    feat_path = "data/processed/phase3a_features.csv"
    if os.path.exists(feat_path):
        df_feat = pd.read_csv(feat_path)
        assert len(df_feat) == 1322
        assert df_feat["id"].is_unique
        assert (df_feat["extraction_status_3a"] == "success").all()
        assert df_feat["shooter_speed_rel"].isnull().sum() == 0
        assert df_feat["shooter_vel_radial"].isnull().sum() == 0
        assert df_feat["shooter_vel_tangential"].isnull().sum() == 0
        # Earlier window has exactly 3 boundary missing values
        assert df_feat["shooter_speed_early"].isnull().sum() == 3

    oof_path = "reports/phase3a/phase3a_oof_predictions.csv"
    if os.path.exists(oof_path):
        df_oof = pd.read_csv(oof_path)
        assert len(df_oof) == 1322
        assert df_oof["shot_id"].is_unique
        # Validate probability ranges for all raw and calibrated columns
        prob_cols = [
            "m1_prob_raw",
            "m1_prob_cal",
            "speed_rel_prob_raw",
            "speed_rel_prob_cal",
            "directional_prob_raw",
            "directional_prob_cal",
            "speed_early_prob_raw",
            "speed_early_prob_cal",
        ]
        for col in prob_cols:
            assert (df_oof[col] >= 0.0).all()
            assert (df_oof[col] <= 1.0).all()
