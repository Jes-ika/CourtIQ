"""Phase 3A feature extraction: directional shooter movement and earlier lookback windows."""

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
    compute_directional_shooter_speed,
    compute_euclidean_distance,
    compute_shooter_speed,
)

logger = logging.getLogger(__name__)


def find_player(frame: Optional[Dict[str, Any]], player_id: Optional[int]) -> Optional[Dict[str, Any]]:
    """Look up player object by playerId in a tracking frame."""
    if not frame or player_id is None:
        return None
    for p in frame.get("homePlayers", []) + frame.get("awayPlayers", []):
        if p.get("playerId") == player_id:
            return p
    return None


def extract_shot_phase3a_features(
    shot: Dict[str, Any] | pd.Series,
    frame_rel: Optional[Dict[str, Any]],
    frame_pre10: Optional[Dict[str, Any]],
    frame_pre20: Optional[Dict[str, Any]],
    hoop_pos: Sequence[float] = DEFAULT_OFFENSIVE_HOOP_LOCATION,
) -> Dict[str, Any]:
    """Extract directional and multi-window shooter movement features for a single shot.

    Timestamps:
    - Release (t): frame_rel (startFrame)
    - 400ms prior (t - 0.4s): frame_pre10 (startFrame - 10)
    - 800ms prior (t - 0.8s): frame_pre20 (startFrame - 20)

    Features:
    - shooter_speed_rel: scalar speed over [t - 0.4s, t] (reproduces existing feature)
    - shooter_speed_early: scalar speed over [t - 0.8s, t - 0.4s]
    - shooter_vel_radial: signed velocity toward hoop over [t - 0.4s, t]
    - shooter_vel_tangential: unsigned lateral velocity orthogonal to hoop over [t - 0.4s, t]
    """
    shot_id = shot.get("id") if isinstance(shot, dict) else shot["id"]
    game_id = shot.get("gameId") if isinstance(shot, dict) else shot["gameId"]
    shooter_id = shot.get("shooterId") if isinstance(shot, dict) else shot["shooterId"]
    period = shot.get("period") if isinstance(shot, dict) else shot["period"]

    res: Dict[str, Any] = {
        "id": shot_id,
        "gameId": game_id,
        "shooter_speed_rel": np.nan,
        "shooter_speed_early": np.nan,
        "shooter_vel_radial": np.nan,
        "shooter_vel_tangential": np.nan,
        "s_detected_rel": False,
        "s_detected_pre10": False,
        "s_detected_pre20": False,
        "both_detected_pre10": False,
        "extraction_status_3a": "unprocessed",
    }

    if frame_rel is None or frame_pre10 is None:
        res["extraction_status_3a"] = "missing_primary_frames"
        return res

    # Verify period match for 400ms lookback
    if frame_rel.get("period") != period or frame_pre10.get("period") != period:
        res["extraction_status_3a"] = "period_mismatch_400ms"
        return res

    s_rel = find_player(frame_rel, shooter_id)
    s_pre10 = find_player(frame_pre10, shooter_id)

    if s_rel is None or s_pre10 is None:
        res["extraction_status_3a"] = "shooter_not_found_400ms"
        return res

    pos_rel = s_rel["xyz"][:2]
    pos_pre10 = s_pre10["xyz"][:2]

    res["s_detected_rel"] = bool(s_rel.get("isDetected", 0) == 1)
    res["s_detected_pre10"] = bool(s_pre10.get("isDetected", 0) == 1)

    # 1. Release-lookback speed [t - 0.4s, t] (existing feature)
    speed_rel = compute_shooter_speed(pos_pre10, pos_rel, delta_t=0.40)
    res["shooter_speed_rel"] = speed_rel

    # 2. Directional decomposition over [t - 0.4s, t]
    v_rad, v_tan, _ = compute_directional_shooter_speed(
        pos_pre=pos_pre10,
        pos_release=pos_rel,
        hoop_pos=hoop_pos,
        delta_t=0.40,
    )
    res["shooter_vel_radial"] = v_rad
    res["shooter_vel_tangential"] = v_tan

    # 3. Earlier-window speed [t - 0.8s, t - 0.4s]
    if frame_pre20 is not None and frame_pre20.get("period") == period:
        s_pre20 = find_player(frame_pre20, shooter_id)
        if s_pre20 is not None:
            pos_pre20 = s_pre20["xyz"][:2]
            res["s_detected_pre20"] = bool(s_pre20.get("isDetected", 0) == 1)
            res["shooter_speed_early"] = compute_shooter_speed(
                pos_pre20, pos_pre10, delta_t=0.40
            )

    # Detection flag for defender at pre10 if available
    def_id = shot.get("closestDefId") if isinstance(shot, dict) else shot["closestDefId"]
    d_pre10 = find_player(frame_pre10, def_id)
    d_detected_pre10 = bool(d_pre10.get("isDetected", 0) == 1) if d_pre10 else False
    res["both_detected_pre10"] = bool(res["s_detected_pre10"] and d_detected_pre10)

    res["extraction_status_3a"] = "success"
    return res


def extract_all_phase3a_features(
    match_dirs: Sequence[Path | str],
    shots_df: pd.DataFrame,
    hoop_pos: Sequence[float] = DEFAULT_OFFENSIVE_HOOP_LOCATION,
) -> pd.DataFrame:
    """Extract Phase 3A directional and multi-window shooter motion across all matches."""
    records: List[Dict[str, Any]] = []

    for m_dir in match_dirs:
        m_path = Path(m_dir)
        gid = int(m_path.name)
        g_shots = shots_df[shots_df["gameId"] == gid].copy()
        if g_shots.empty:
            continue

        track_file = m_path / f"{gid}_tracking_data.jsonl.gz"
        if not track_file.exists():
            candidates = list(m_path.glob("*_tracking_data.jsonl.gz"))
            if candidates:
                track_file = candidates[0]
            else:
                raise FileNotFoundError(f"Tracking file not found in {m_path}")

        needed_frames: Set[int] = set()
        for _, s in g_shots.iterrows():
            sf = int(s["startFrame"])
            needed_frames.add(sf)
            needed_frames.add(sf - 10)
            needed_frames.add(sf - 20)

        frames_dict: Dict[int, Dict[str, Any]] = {}
        with gzip.open(track_file, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                fr = json.loads(line)
                fidx = fr.get("frameIdx")
                if fidx in needed_frames:
                    frames_dict[fidx] = fr

        for _, shot in g_shots.iterrows():
            sf = int(shot["startFrame"])
            fr_rel = frames_dict.get(sf)
            fr_pre10 = frames_dict.get(sf - 10)
            fr_pre20 = frames_dict.get(sf - 20)

            rec = extract_shot_phase3a_features(
                shot=shot,
                frame_rel=fr_rel,
                frame_pre10=fr_pre10,
                frame_pre20=fr_pre20,
                hoop_pos=hoop_pos,
            )
            records.append(rec)

    full_df = pd.DataFrame(records)
    # Align to shots_df ordering
    ordered_df = shots_df[["id"]].merge(full_df, on="id", how="left")
    return ordered_df
