"""Pre-release kinematic feature extraction from raw optical tracking frames.

Extracts dynamic contest kinematics and spatial geometry for basketball shot attempts:
- defender_closing_velocity: rate of distance closure by primary defender (ft/s)
- delta_defender_dist: net change in distance over lookback window (ft)
- contest_angle_cos: cosine of angle between shooter-to-hoop and shooter-to-defender vectors
- shooter_velocity_magnitude: shooter displacement speed over lookback window (ft/s)
- tracking quality and detection flags for sensitivity auditing
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from courtiq.features.geometry import (
    DEFAULT_OFFENSIVE_HOOP_LOCATION,
    compute_closing_velocity,
    compute_contest_angle,
    compute_distance_change,
    compute_euclidean_distance,
    compute_shooter_speed,
)

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_FRAMES = 10  # 400 ms at 25 fps
DEFAULT_DELTA_T = 0.40  # seconds


def find_player_in_frame(
    frame: Optional[Dict[str, Any]], player_id: Optional[int]
) -> Optional[Dict[str, Any]]:
    """Look up player object by playerId in a tracking frame."""
    if not frame or player_id is None:
        return None
    players = frame.get("homePlayers", []) + frame.get("awayPlayers", [])
    for p in players:
        if p.get("playerId") == player_id:
            return p
    return None


def extract_shot_kinematics(
    shot: Dict[str, Any] | pd.Series,
    frame_release: Optional[Dict[str, Any]],
    frame_pre: Optional[Dict[str, Any]],
    delta_t: float = DEFAULT_DELTA_T,
    hoop_pos: Sequence[float] = DEFAULT_OFFENSIVE_HOOP_LOCATION,
) -> Dict[str, Any]:
    """Extract pre-release motion and contest geometry features for a single shot.

    Args:
        shot: Shot event dictionary or Series containing 'startFrame', 'shooterId',
              'closestDefId', 'period', 'id', 'gameId'.
        frame_release: Tracking frame at release (startFrame).
        frame_pre: Tracking frame at lookback (startFrame - lookback_frames).
        delta_t: Lookback window duration in seconds (default 0.4s).
        hoop_pos: Coordinates of the offensive hoop [x, y] in feet.

    Returns:
        Dictionary of kinematic features and tracking quality indicators.
    """
    shot_id = shot.get("id") if isinstance(shot, dict) else shot["id"]
    game_id = shot.get("gameId") if isinstance(shot, dict) else shot["gameId"]
    shooter_id = shot.get("shooterId") if isinstance(shot, dict) else shot["shooterId"]
    def_id = shot.get("closestDefId") if isinstance(shot, dict) else shot["closestDefId"]
    shot_period = shot.get("period") if isinstance(shot, dict) else shot["period"]

    res: Dict[str, Any] = {
        "id": shot_id,
        "gameId": game_id,
        "defender_closing_velocity": np.nan,
        "delta_defender_dist": np.nan,
        "contest_angle_cos": np.nan,
        "contest_angle_deg": np.nan,
        "shooter_velocity_magnitude": np.nan,
        "dist_release": np.nan,
        "dist_pre": np.nan,
        "shooter_detected_release": False,
        "defender_detected_release": False,
        "shooter_detected_pre": False,
        "defender_detected_pre": False,
        "both_detected_pre": False,
        "shooter_pred_error_pre": np.nan,
        "defender_pred_error_pre": np.nan,
        "pre_frame_live": False,
        "same_period": False,
        "extraction_status": "unprocessed",
    }

    if frame_release is None:
        res["extraction_status"] = "missing_release_frame"
        return res

    if frame_pre is None:
        res["extraction_status"] = "missing_pre_frame"
        return res

    # Verify period continuity
    rel_period = frame_release.get("period")
    pre_period = frame_pre.get("period")
    same_period = (rel_period == pre_period == shot_period)
    res["same_period"] = same_period

    if not same_period:
        res["extraction_status"] = "period_mismatch_boundary"
        return res

    # Live clock check
    pre_clock_stopped = frame_pre.get("gameClockStopped", True)
    res["pre_frame_live"] = not pre_clock_stopped

    # Locate shooter and primary defender at release
    s_rel = find_player_in_frame(frame_release, shooter_id)
    d_rel = find_player_in_frame(frame_release, def_id)

    # Locate SAME shooter and SAME primary defender at lookback frame
    s_pre = find_player_in_frame(frame_pre, shooter_id)
    d_pre = find_player_in_frame(frame_pre, def_id)

    if s_rel is None:
        res["extraction_status"] = "shooter_not_found_release"
        return res

    if d_rel is None:
        res["extraction_status"] = "defender_not_found_release"
        return res

    if s_pre is None:
        res["extraction_status"] = "shooter_not_found_pre"
        return res

    if d_pre is None:
        res["extraction_status"] = "defender_not_found_pre"
        return res

    # Tracking detection quality flags
    res["shooter_detected_release"] = bool(s_rel.get("isDetected", 0) == 1)
    res["defender_detected_release"] = bool(d_rel.get("isDetected", 0) == 1)
    res["shooter_detected_pre"] = bool(s_pre.get("isDetected", 0) == 1)
    res["defender_detected_pre"] = bool(d_pre.get("isDetected", 0) == 1)
    res["both_detected_pre"] = bool(
        res["shooter_detected_pre"] and res["defender_detected_pre"]
    )
    res["shooter_pred_error_pre"] = s_pre.get("predError", np.nan)
    res["defender_pred_error_pre"] = d_pre.get("predError", np.nan)

    # Extract 2D coordinates [x, y]
    s_rel_pos = s_rel["xyz"][:2]
    d_rel_pos = d_rel["xyz"][:2]
    s_pre_pos = s_pre["xyz"][:2]
    d_pre_pos = d_pre["xyz"][:2]

    # Calculate spatial distances
    dist_rel = compute_euclidean_distance(s_rel_pos, d_rel_pos)
    dist_pre = compute_euclidean_distance(s_pre_pos, d_pre_pos)

    res["dist_release"] = dist_rel
    res["dist_pre"] = dist_pre

    # Calculate kinematics
    res["defender_closing_velocity"] = compute_closing_velocity(
        dist_pre=dist_pre, dist_release=dist_rel, delta_t=delta_t
    )
    res["delta_defender_dist"] = compute_distance_change(
        dist_pre=dist_pre, dist_release=dist_rel
    )
    cos_theta, deg = compute_contest_angle(
        pos_shooter=s_rel_pos, pos_defender=d_rel_pos, hoop_pos=hoop_pos
    )
    res["contest_angle_cos"] = cos_theta
    res["contest_angle_deg"] = deg

    res["shooter_velocity_magnitude"] = compute_shooter_speed(
        pos_pre=s_pre_pos, pos_release=s_rel_pos, delta_t=delta_t
    )
    res["extraction_status"] = "success"

    return res


def extract_match_kinematics(
    match_dir: Path | str,
    shots_df: pd.DataFrame,
    lookback_frames: int = DEFAULT_LOOKBACK_FRAMES,
    delta_t: float = DEFAULT_DELTA_T,
    hoop_pos: Sequence[float] = DEFAULT_OFFENSIVE_HOOP_LOCATION,
) -> pd.DataFrame:
    """Extract kinematics for all shots belonging to a single match directory.

    Streams the tracking file once to retrieve required frames.
    """
    m_dir = Path(match_dir)
    game_id = int(m_dir.name)

    g_shots = shots_df[shots_df["gameId"] == game_id].copy()
    if g_shots.empty:
        return pd.DataFrame()

    tracking_file = m_dir / f"{game_id}_tracking_data.jsonl.gz"
    if not tracking_file.exists():
        candidates = list(m_dir.glob("*_tracking_data.jsonl.gz"))
        if candidates:
            tracking_file = candidates[0]
        else:
            raise FileNotFoundError(f"Tracking file not found in {m_dir}")

    # Build set of needed frame indices
    needed_frames: Set[int] = set()
    for _, s in g_shots.iterrows():
        sf = int(s["startFrame"])
        needed_frames.add(sf)
        needed_frames.add(sf - lookback_frames)

    # Single-pass streaming read of tracking frames
    frames_dict: Dict[int, Dict[str, Any]] = {}
    with gzip.open(tracking_file, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            fr = json.loads(line)
            fidx = fr.get("frameIdx")
            if fidx in needed_frames:
                frames_dict[fidx] = fr

    records = []
    for _, shot in g_shots.iterrows():
        sf = int(shot["startFrame"])
        fr_rel = frames_dict.get(sf)
        fr_pre = frames_dict.get(sf - lookback_frames)

        rec = extract_shot_kinematics(
            shot=shot,
            frame_release=fr_rel,
            frame_pre=fr_pre,
            delta_t=delta_t,
            hoop_pos=hoop_pos,
        )
        records.append(rec)

    return pd.DataFrame(records)


def extract_all_kinematics(
    match_dirs: Sequence[Path | str],
    shots_df: pd.DataFrame,
    lookback_frames: int = DEFAULT_LOOKBACK_FRAMES,
    delta_t: float = DEFAULT_DELTA_T,
    hoop_pos: Sequence[float] = DEFAULT_OFFENSIVE_HOOP_LOCATION,
) -> pd.DataFrame:
    """Extract kinematics for all shots across multiple matches.

    Preserves stable row ordering and shot IDs.
    """
    match_dfs = []
    for m_dir in match_dirs:
        m_path = Path(m_dir)
        df_m = extract_match_kinematics(
            match_dir=m_path,
            shots_df=shots_df,
            lookback_frames=lookback_frames,
            delta_t=delta_t,
            hoop_pos=hoop_pos,
        )
        if not df_m.empty:
            match_dfs.append(df_m)

    if not match_dfs:
        return pd.DataFrame()

    full_df = pd.concat(match_dfs, ignore_index=True)
    # Re-align with shots_df ordering
    ordered_df = shots_df[["id"]].merge(full_df, on="id", how="left")
    return ordered_df
