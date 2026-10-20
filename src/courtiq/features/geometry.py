"""Mathematical geometry and kinematics for tracking-derived basketball features."""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

# SkillCorner ACB calibrated offensive hoop coordinates (in feet)
# Center of court is (0.0, 0.0); offensive hoop is at x = -40.75, y = 0.0
DEFAULT_OFFENSIVE_HOOP_LOCATION = (-40.75, 0.0)


def compute_euclidean_distance(
    p1: Sequence[float],
    p2: Sequence[float],
) -> float:
    """Compute 2D Euclidean distance between two points in feet."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_closing_velocity(
    dist_pre: float,
    dist_release: float,
    delta_t: float = 0.4,
) -> float:
    """Compute defender closing velocity relative to the shooter.

    Formula:
        v_close = (dist_pre - dist_release) / delta_t

    Sign Convention:
        - v_close > 0: Defender is approaching / closing in on the shooter (distance decreasing).
        - v_close < 0: Defender is receding / moving away from the shooter (distance increasing).
        - v_close == 0: Defender maintains constant distance relative to shooter.

    Args:
        dist_pre: Defender distance to shooter at lookback frame (t - delta_t), in feet.
        dist_release: Defender distance to shooter at release frame (t), in feet.
        delta_t: Lookback duration in seconds (default 0.4s = 10 frames at 25 fps).

    Returns:
        Closing velocity in feet per second (ft/s).
    """
    if delta_t <= 0:
        raise ValueError(f"delta_t must be positive, got {delta_t}")
    return (dist_pre - dist_release) / delta_t


def compute_distance_change(
    dist_pre: float,
    dist_release: float,
) -> float:
    """Compute net change in defender distance between pre-release and release.

    Formula:
        delta_dist = dist_release - dist_pre

    Negative values indicate the defender got closer.
    """
    return dist_release - dist_pre


def compute_shooter_speed(
    pos_pre: Sequence[float],
    pos_release: Sequence[float],
    delta_t: float = 0.4,
) -> float:
    """Compute shooter scalar translation speed over the lookback window.

    Args:
        pos_pre: 2D coordinates at t - delta_t [x, y].
        pos_release: 2D coordinates at t [x, y].
        delta_t: Duration in seconds.

    Returns:
        Scalar speed in feet per second (ft/s).
    """
    if delta_t <= 0:
        raise ValueError(f"delta_t must be positive, got {delta_t}")
    dist = compute_euclidean_distance(pos_pre, pos_release)
    return dist / delta_t


def compute_directional_shooter_speed(
    pos_pre: Sequence[float],
    pos_release: Sequence[float],
    hoop_pos: Sequence[float] = DEFAULT_OFFENSIVE_HOOP_LOCATION,
    delta_t: float = 0.4,
) -> Tuple[float, float, float]:
    """Compute directional components of shooter displacement relative to the offensive hoop.

    Decomposes displacement into:
    - Radial velocity (v_radial): signed component along shooter-to-hoop axis (positive = toward hoop, negative = fadeaway).
    - Tangential velocity (v_tangential): non-negative component orthogonal to hoop axis (lateral movement).
    - Total scalar speed (v_total): Euclidean translation speed.

    Returns:
        Tuple of (v_radial, v_tangential, v_total) in feet per second (ft/s).
    """
    if delta_t <= 0:
        raise ValueError(f"delta_t must be positive, got {delta_t}")

    # Vector to hoop from shooter at pre-release frame
    dh_x = hoop_pos[0] - pos_pre[0]
    dh_y = hoop_pos[1] - pos_pre[1]
    norm_dh = math.hypot(dh_x, dh_y)

    # Shooter displacement vector over lookback window
    disp_x = pos_release[0] - pos_pre[0]
    disp_y = pos_release[1] - pos_pre[1]
    total_dist = math.hypot(disp_x, disp_y)
    v_total = total_dist / delta_t

    if norm_dh < 1e-6:
        # Shooter is at the hoop
        return 0.0, v_total, v_total

    # Unit vector toward hoop
    uh_x = dh_x / norm_dh
    uh_y = dh_y / norm_dh

    # Unit vector perpendicular to hoop axis
    up_x = -uh_y
    up_y = uh_x

    # Projections
    rad_disp = disp_x * uh_x + disp_y * uh_y
    tan_disp = abs(disp_x * up_x + disp_y * up_y)

    v_radial = rad_disp / delta_t
    v_tangential = tan_disp / delta_t

    return v_radial, v_tangential, v_total


def compute_contest_angle(
    pos_shooter: Sequence[float],
    pos_defender: Sequence[float],
    hoop_pos: Sequence[float] = DEFAULT_OFFENSIVE_HOOP_LOCATION,
) -> Tuple[float, float]:
    """Compute contest angle and cosine between shooter-to-hoop and shooter-to-defender vectors.

    Formulation:
        u_hoop = (hoop_pos - pos_shooter) / ||hoop_pos - pos_shooter||
        u_def  = (pos_defender - pos_shooter) / ||pos_defender - pos_shooter||
        cos(theta) = dot(u_hoop, u_def)

    Interpretation:
        - cos(theta) approx 1.0 (theta approx 0 deg): Defender is directly in front of the shooter
          along the direct line-of-sight to the basket (maximal contest obstruction).
        - cos(theta) approx 0.0 (theta approx 90 deg): Defender is positioned laterally (side contest).
        - cos(theta) < 0.0 (theta > 90 deg): Defender is trailing or behind the shooter.

    Returns:
        Tuple of (cos_theta, theta_degrees).
    """
    # Vector from shooter to hoop
    vh_x = hoop_pos[0] - pos_shooter[0]
    vh_y = hoop_pos[1] - pos_shooter[1]
    norm_vh = math.hypot(vh_x, vh_y)

    # Vector from shooter to defender
    vd_x = pos_defender[0] - pos_shooter[0]
    vd_y = pos_defender[1] - pos_shooter[1]
    norm_vd = math.hypot(vd_x, vd_y)

    # Handle degenerate zero-distance positions safely
    if norm_vh < 1e-6 or norm_vd < 1e-6:
        return 0.0, 90.0

    # Unit vectors
    uh_x, uh_y = vh_x / norm_vh, vh_y / norm_vh
    ud_x, ud_y = vd_x / norm_vd, vd_y / norm_vd

    # Dot product
    cos_theta = uh_x * ud_x + uh_y * ud_y
    # Clamp for numerical stability
    cos_theta = max(-1.0, min(1.0, cos_theta))

    theta_rad = math.acos(cos_theta)
    theta_deg = math.degrees(theta_rad)

    return cos_theta, theta_deg


def is_approaching(dist_pre: float, dist_release: float) -> bool:
    """Return True if defender moved closer to the shooter during the lookback."""
    return dist_release < dist_pre
