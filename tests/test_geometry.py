"""Unit tests for spatial geometry, kinematics, and contest angle calculations."""

import math
import pytest
from courtiq.features.geometry import (
    DEFAULT_OFFENSIVE_HOOP_LOCATION,
    compute_closing_velocity,
    compute_contest_angle,
    compute_distance_change,
    compute_euclidean_distance,
    compute_shooter_speed,
    is_approaching,
)


def test_euclidean_distance_known_values():
    """Verify standard Pythagorean distances."""
    assert compute_euclidean_distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)
    assert compute_euclidean_distance((-10.0, 5.0), (-10.0, 5.0)) == pytest.approx(0.0)


def test_closing_velocity_approaching():
    """Verify that a defender moving closer yields a positive closing velocity."""
    dist_pre = 10.0  # 10 feet away at t - 400ms
    dist_release = 4.0  # 4 feet away at release
    delta_t = 0.4  # 400ms = 10 frames at 25fps

    v_close = compute_closing_velocity(dist_pre, dist_release, delta_t)
    delta_d = compute_distance_change(dist_pre, dist_release)

    # Closing velocity should be (10 - 4) / 0.4 = 15 ft/s
    assert v_close == pytest.approx(15.0)
    assert v_close > 0.0
    assert is_approaching(dist_pre, dist_release) is True
    assert delta_d == pytest.approx(-6.0)


def test_closing_velocity_receding():
    """Verify that a defender moving away yields a negative closing velocity."""
    dist_pre = 3.0  # 3 feet away at t - 400ms
    dist_release = 7.0  # 7 feet away at release (e.g., bit on pump fake, fell back)
    delta_t = 0.4

    v_close = compute_closing_velocity(dist_pre, dist_release, delta_t)
    delta_d = compute_distance_change(dist_pre, dist_release)

    # Closing velocity should be (3 - 7) / 0.4 = -10 ft/s
    assert v_close == pytest.approx(-10.0)
    assert v_close < 0.0
    assert is_approaching(dist_pre, dist_release) is False
    assert delta_d == pytest.approx(4.0)


def test_closing_velocity_static():
    """Verify that a stationary defender relative to shooter yields zero closing velocity."""
    dist_pre = 5.0
    dist_release = 5.0
    v_close = compute_closing_velocity(dist_pre, dist_release, delta_t=0.4)
    assert v_close == pytest.approx(0.0)
    assert is_approaching(dist_pre, dist_release) is False


def test_closing_velocity_invalid_delta_t():
    """Verify that non-positive delta_t raises ValueError."""
    with pytest.raises(ValueError):
        compute_closing_velocity(5.0, 4.0, delta_t=0.0)
    with pytest.raises(ValueError):
        compute_closing_velocity(5.0, 4.0, delta_t=-0.2)


def test_shooter_speed():
    """Verify shooter speed calculation from 2D coordinates."""
    pos_pre = (-20.0, 0.0)
    pos_release = (-24.0, 3.0)  # moved 5 feet in 0.4s
    speed = compute_shooter_speed(pos_pre, pos_release, delta_t=0.4)
    assert speed == pytest.approx(5.0 / 0.4)  # 12.5 ft/s


def test_contest_angle_directly_in_front():
    """Verify defender directly on the line between shooter and hoop gives cos=1, theta=0."""
    hoop = (-40.75, 0.0)
    shooter = (-20.0, 0.0)  # Shooter at x = -20, y = 0
    # Hoop is at x = -40.75. So direct path to hoop is in negative x direction.
    # Defender directly in front of shooter (closer to hoop):
    defender = (-25.0, 0.0)

    cos_theta, theta_deg = compute_contest_angle(shooter, defender, hoop_pos=hoop)
    assert cos_theta == pytest.approx(1.0, abs=1e-5)
    assert theta_deg == pytest.approx(0.0, abs=1e-3)


def test_contest_angle_directly_behind():
    """Verify defender directly behind shooter gives cos=-1, theta=180."""
    hoop = (-40.75, 0.0)
    shooter = (-20.0, 0.0)
    # Defender is trailing behind shooter (positive x direction, farther from hoop)
    defender = (-15.0, 0.0)

    cos_theta, theta_deg = compute_contest_angle(shooter, defender, hoop_pos=hoop)
    assert cos_theta == pytest.approx(-1.0, abs=1e-5)
    assert theta_deg == pytest.approx(180.0, abs=1e-3)


def test_contest_angle_orthogonal_lateral():
    """Verify lateral defender gives cos=0, theta=90."""
    hoop = (-40.75, 0.0)
    shooter = (-20.0, 0.0)
    # Line to hoop is (-20.75, 0). Pure lateral offset is (0, 4.0)
    defender = (-20.0, 4.0)

    cos_theta, theta_deg = compute_contest_angle(shooter, defender, hoop_pos=hoop)
    assert cos_theta == pytest.approx(0.0, abs=1e-5)
    assert theta_deg == pytest.approx(90.0, abs=1e-3)


def test_contest_angle_45_degrees():
    """Verify 45-degree diagonal contest angle gives cos=sqrt(2)/2."""
    hoop = (-40.75, 0.0)
    shooter = (-20.0, 0.0)
    # Vector to hoop is (-1, 0) normalized.
    # 45 degrees toward hoop and left: (-3, 3) offset
    defender = (-23.0, 3.0)

    cos_theta, theta_deg = compute_contest_angle(shooter, defender, hoop_pos=hoop)
    expected_cos = math.cos(math.radians(45.0))
    assert cos_theta == pytest.approx(expected_cos, abs=1e-4)
    assert theta_deg == pytest.approx(45.0, abs=1e-2)


def test_contest_angle_degenerate_positions():
    """Verify degenerate case where defender is at shooter position returns safe default."""
    hoop = (-40.75, 0.0)
    shooter = (-20.0, 0.0)
    defender = (-20.0, 0.0)

    cos_theta, theta_deg = compute_contest_angle(shooter, defender, hoop_pos=hoop)
    assert cos_theta == 0.0
    assert theta_deg == 90.0
