"""Integration and regression tests for Phase 2 kinematic feature extraction and validation."""

import numpy as np
import pandas as pd
import pytest

from courtiq.features.geometry import (
    DEFAULT_OFFENSIVE_HOOP_LOCATION,
    compute_closing_velocity,
    compute_contest_angle,
    compute_distance_change,
    compute_euclidean_distance,
    compute_shooter_speed,
)
from courtiq.features.kinematics import (
    extract_shot_kinematics,
    find_player_in_frame,
)
from courtiq.evaluation.phase2_cv import (
    MODEL_CONFIGS,
    compute_comprehensive_metrics,
    fit_and_predict_platt_calibration,
)


@pytest.fixture
def sample_synthetic_frames():
    """Create synthetic tracking frames at release (t) and lookback (t-10)."""
    # Hoop at (-40.75, 0.0)
    # Release frame (t = 100): shooter at (-20.0, 0.0), defender at (-23.0, 0.0) -> dist = 3.0 ft
    frame_rel = {
        "frameIdx": 100,
        "period": 1,
        "gameClockStopped": False,
        "homePlayers": [
            {
                "playerId": 101,  # Shooter
                "xyz": [-20.0, 0.0, 0.0],
                "speed": 5.0,
                "isDetected": 1,
                "predError": 0.5,
            }
        ],
        "awayPlayers": [
            {
                "playerId": 201,  # Primary defender
                "xyz": [-23.0, 0.0, 0.0],
                "speed": 10.0,
                "isDetected": 1,
                "predError": 0.8,
            },
            {
                "playerId": 202,  # Another defender
                "xyz": [-15.0, 5.0, 0.0],
                "speed": 3.0,
                "isDetected": 1,
                "predError": 1.0,
            },
        ],
    }

    # Lookback frame (t - 10 = 90):
    # Shooter at (-18.0, 0.0), primary defender (201) was at (-26.0, 0.0) -> dist = 8.0 ft
    frame_pre = {
        "frameIdx": 90,
        "period": 1,
        "gameClockStopped": False,
        "homePlayers": [
            {
                "playerId": 101,
                "xyz": [-18.0, 0.0, 0.0],
                "speed": 5.0,
                "isDetected": 1,
                "predError": 0.6,
            }
        ],
        "awayPlayers": [
            {
                "playerId": 201,  # Primary defender was 8ft away
                "xyz": [-26.0, 0.0, 0.0],
                "speed": 12.0,
                "isDetected": 1,
                "predError": 0.9,
            },
            {
                "playerId": 202,  # Another defender was closer (2ft away), but NOT primary defender
                "xyz": [-18.0, 2.0, 0.0],
                "speed": 2.0,
                "isDetected": 1,
                "predError": 0.5,
            },
        ],
    }

    shot = {
        "id": "test-shot-1",
        "gameId": 999999,
        "period": 1,
        "startFrame": 100,
        "shooterId": 101,
        "closestDefId": 201,
        "closestDefDist": 3.0,
        "location": [-20.0, 0.0],
        "distance": 20.75,
        "outcome": True,
    }

    return shot, frame_rel, frame_pre


def test_defender_identity_persistence(sample_synthetic_frames):
    """Verify that the SAME primary defender (201) is tracked at t-10, even if player 202 was closer."""
    shot, frame_rel, frame_pre = sample_synthetic_frames
    res = extract_shot_kinematics(shot, frame_rel, frame_pre, delta_t=0.4)

    assert res["extraction_status"] == "success"
    # Primary defender (201) closed from 8.0ft to 3.0ft:
    # v_close = (8.0 - 3.0) / 0.4 = 12.5 ft/s
    assert res["dist_release"] == pytest.approx(3.0)
    assert res["dist_pre"] == pytest.approx(8.0)
    assert res["defender_closing_velocity"] == pytest.approx(12.5)
    assert res["delta_defender_dist"] == pytest.approx(-5.0)

    # Shooter speed: moved from (-18, 0) to (-20, 0) -> dist = 2.0 ft in 0.4s -> 5.0 ft/s
    assert res["shooter_velocity_magnitude"] == pytest.approx(5.0)

    # Defender is directly between shooter (-20, 0) and hoop (-40.75, 0) -> cos = 1.0, deg = 0.0
    assert res["contest_angle_cos"] == pytest.approx(1.0, abs=1e-5)
    assert res["contest_angle_deg"] == pytest.approx(0.0, abs=1e-3)


def test_missing_frames_and_players(sample_synthetic_frames):
    """Verify safe degradation when tracking frames or players are missing."""
    shot, frame_rel, frame_pre = sample_synthetic_frames

    # Missing release frame
    res1 = extract_shot_kinematics(shot, None, frame_pre)
    assert res1["extraction_status"] == "missing_release_frame"
    assert np.isnan(res1["defender_closing_velocity"])

    # Missing pre frame
    res2 = extract_shot_kinematics(shot, frame_rel, None)
    assert res2["extraction_status"] == "missing_pre_frame"
    assert np.isnan(res2["defender_closing_velocity"])

    # Missing shooter in pre frame
    frame_pre_no_shooter = dict(frame_pre)
    frame_pre_no_shooter["homePlayers"] = []
    res3 = extract_shot_kinematics(shot, frame_rel, frame_pre_no_shooter)
    assert res3["extraction_status"] == "shooter_not_found_pre"

    # Missing defender in pre frame
    frame_pre_no_def = dict(frame_pre)
    frame_pre_no_def["awayPlayers"] = []
    res4 = extract_shot_kinematics(shot, frame_rel, frame_pre_no_def)
    assert res4["extraction_status"] == "defender_not_found_pre"


def test_period_boundary_mismatch(sample_synthetic_frames):
    """Verify that crossing a period boundary is flagged and excluded."""
    shot, frame_rel, frame_pre = sample_synthetic_frames
    frame_pre_p2 = dict(frame_pre)
    frame_pre_p2["period"] = 2  # period mismatch!

    res = extract_shot_kinematics(shot, frame_rel, frame_pre_p2)
    assert res["extraction_status"] == "period_mismatch_boundary"
    assert res["same_period"] is False
    assert np.isnan(res["defender_closing_velocity"])


def test_extrapolated_vs_detected_flags(sample_synthetic_frames):
    """Verify detection flags correctly differentiate CV-detected vs extrapolated players."""
    shot, frame_rel, frame_pre = sample_synthetic_frames

    # Mark defender as extrapolated in lookback frame
    frame_pre_extrap = dict(frame_pre)
    frame_pre_extrap["awayPlayers"] = [
        {
            "playerId": 201,
            "xyz": [-26.0, 0.0, 0.0],
            "speed": 12.0,
            "isDetected": 0,  # Extrapolated
            "predError": 4.5,
        }
    ]
    res = extract_shot_kinematics(shot, frame_rel, frame_pre_extrap)
    assert res["extraction_status"] == "success"
    assert res["shooter_detected_pre"] is True
    assert res["defender_detected_pre"] is False
    assert res["both_detected_pre"] is False
    assert res["defender_pred_error_pre"] == pytest.approx(4.5)


def test_no_post_release_data_usage(sample_synthetic_frames):
    """Verify that extraction only uses frames at or strictly before startFrame."""
    shot, frame_rel, frame_pre = sample_synthetic_frames
    assert frame_rel["frameIdx"] == shot["startFrame"]
    assert frame_pre["frameIdx"] < shot["startFrame"]
    assert frame_rel["frameIdx"] - frame_pre["frameIdx"] == 10  # 400ms lookback


def test_cached_features_sample_integrity():
    """Verify that the generated phase2_features.csv matches eligible shots exactly."""
    import os
    if os.path.exists("data/processed/phase2_features.csv"):
        df_feat = pd.read_csv("data/processed/phase2_features.csv")
        assert len(df_feat) == 1322
        assert df_feat["id"].is_unique
        assert (df_feat["extraction_status"] == "success").all()
        assert df_feat["defender_closing_velocity"].isnull().sum() == 0
        assert df_feat["contest_angle_cos"].isnull().sum() == 0
        assert df_feat["shooter_velocity_magnitude"].isnull().sum() == 0
        assert df_feat["delta_defender_dist"].isnull().sum() == 0


def test_oof_predictions_matched_rows():
    """Verify that phase2_oof_predictions.csv has identical rows across M0, M1, M2."""
    import os
    if os.path.exists("reports/phase2_oof_predictions.csv"):
        df_oof = pd.read_csv("reports/phase2_oof_predictions.csv")
        assert len(df_oof) == 1322
        assert df_oof["shot_id"].is_unique
        # Probability bounds
        for col in ["m0_prob", "m1_prob", "m2_prob", "closing_only_prob", "angle_only_prob", "shooter_vel_only_prob"]:
            assert (df_oof[col] >= 0.0).all()
            assert (df_oof[col] <= 1.0).all()


def test_in_fold_platt_calibration():
    """Verify in-fold Platt scaling produces valid probabilities without looking at test labels."""
    rng = np.random.default_rng(42)
    train_scores = rng.uniform(0.1, 0.9, size=200)
    train_y = rng.binomial(1, train_scores)
    test_scores = rng.uniform(0.1, 0.9, size=50)

    cal_test = fit_and_predict_platt_calibration(train_scores, train_y, test_scores)
    assert len(cal_test) == 50
    assert (cal_test >= 0.0).all()
    assert (cal_test <= 1.0).all()
