"""Unit tests for shot-event to tracking-frame alignment."""

import pytest
from courtiq.data.alignment import calculate_2d_euclidean_distance, verify_shot_tracking_alignment


def test_2d_euclidean_distance():
    p1 = [0.0, 0.0]
    p2 = [3.0, 4.0]
    dist = calculate_2d_euclidean_distance(p1, p2)
    assert pytest.approx(dist, rel=1e-5) == 5.0

    p3 = [-10.5, 5.2]
    p4 = [-10.5, 5.2]
    assert calculate_2d_euclidean_distance(p3, p4) == 0.0


def test_shot_tracking_alignment_success():
    shot = {
        "id": "shot-1",
        "startFrame": 500,
        "period": 1,
        "shooterId": 10,
        "closestDefId": 20,
        "closestDefDist": 5.0,
    }

    tracking_frame = {
        "frameIdx": 500,
        "period": 1,
        "homePlayers": [
            {"playerId": 10, "xyz": [0.0, 0.0, 0.0]},
            {"playerId": 99, "xyz": [10.0, 10.0, 0.0]},
        ],
        "awayPlayers": [
            {"playerId": 20, "xyz": [3.0, 4.0, 0.0]},
        ],
    }

    result = verify_shot_tracking_alignment(shot, tracking_frame, tolerance_frames=0)
    assert result["is_aligned"] is True
    assert result["frame_diff"] == 0
    assert result["period_match"] is True
    assert result["shooter_found_in_tracking"] is True
    assert result["defender_found_in_tracking"] is True
    assert pytest.approx(result["tracking_calculated_dist"], rel=1e-4) == 5.0
    assert result["distance_aligned"] is True


def test_shot_tracking_alignment_frame_mismatch():
    shot = {"id": "shot-1", "startFrame": 500, "period": 1}
    tracking_frame = {"frameIdx": 505, "period": 1}

    result = verify_shot_tracking_alignment(shot, tracking_frame, tolerance_frames=0)
    assert result["is_aligned"] is False
    assert "Frame mismatch" in result["reason"]

    # With tolerance = 5, it should pass
    result_tol = verify_shot_tracking_alignment(shot, tracking_frame, tolerance_frames=5)
    assert result_tol["is_aligned"] is True


def test_shot_tracking_alignment_period_mismatch():
    shot = {"id": "shot-1", "startFrame": 500, "period": 1}
    tracking_frame = {"frameIdx": 500, "period": 2}

    result = verify_shot_tracking_alignment(shot, tracking_frame, tolerance_frames=0)
    assert result["is_aligned"] is False
    assert "Period mismatch" in result["reason"]
