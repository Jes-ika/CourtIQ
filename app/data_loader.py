"""Data loading and verification utilities for the CourtIQ research dashboard.

Ensures strict adherence to data integrity, cached loading, and zero row leakage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Authoritative stored metrics from reports/baseline_results.json and reports/phase3a/phase3a_metrics.json
AUTHORITATIVE_METRICS: Dict[str, Dict[str, Any]] = {
    "M0_baseline": {
        "name": "M0 Baseline (Spatial/Mechanics)",
        "raw_features": 9,
        "design_cols": 21,
        "log_loss": 0.655161,
        "delta_vs_m1": 0.037891,
        "brier_score": 0.231456,
        "roc_auc": 0.641026,
        "ece_10": 0.030552,
        "ece_5": 0.020006,
        "status": "Deprecated Baseline",
        "description": "Shot distance, shot clock, touch-to-release time, dribbles, shot type, 3PT flag, catch-and-shoot, paint creation, period",
    },
    "M1_benchmark": {
        "name": "M1 Benchmark (Frozen Official)",
        "raw_features": 12,
        "design_cols": 29,  # 28 in Fold 6
        "log_loss": 0.617270,
        "delta_vs_m1": 0.0,
        "brier_score": 0.215942,
        "roc_auc": 0.694100,
        "ece_10": 0.051075,
        "ece_5": 0.042527,
        "status": "Official Frozen Benchmark",
        "description": "M0 features + closest defender distance, contest level, contested indicator (5 continuous + 7 categorical)",
    },
    "M1_plus_speed_rel": {
        "name": "M1 + Speed Rel [t-0.4s, t]",
        "raw_features": 13,
        "design_cols": 30,  # 29 in Fold 6
        "log_loss": 0.616512,
        "delta_vs_m1": -0.000758,
        "brier_score": 0.215511,
        "roc_auc": 0.696695,
        "ece_10": 0.044691,
        "ece_5": 0.040302,
        "ci_95": [-0.002177, 0.000854],
        "game_wins": "7 / 10 Games",
        "status": "Exploratory Hypothesis (Unconfirmed)",
        "description": "M1 + release-lookback shooter speed over 10 frames (400 ms) preceding release",
    },
    "M1_plus_speed_early": {
        "name": "M1 + Speed Early [t-0.8s, t-0.4s]",
        "raw_features": 13,
        "design_cols": 30,
        "log_loss": 0.617179,
        "delta_vs_m1": -0.000091,
        "brier_score": 0.215850,
        "roc_auc": 0.694897,
        "ece_10": 0.044617,
        "ece_5": 0.038236,
        "ci_95": [-0.001136, 0.000901],
        "game_wins": "5 / 10 Games",
        "status": "Signal Decayed",
        "description": "M1 + pre-gather shooter speed over frames t-20 to t-10 (approach momentum)",
    },
    "M1_plus_directional": {
        "name": "M1 + Directional (v_rad + v_tan)",
        "raw_features": 14,
        "design_cols": 31,
        "log_loss": 0.618350,
        "delta_vs_m1": 0.001080,
        "brier_score": 0.216403,
        "roc_auc": 0.692545,
        "ece_10": 0.050836,
        "ece_5": 0.042004,
        "ci_95": [0.000211, 0.002238],
        "game_wins": "1 / 10 Games",
        "status": "Statistically Worse (Negative Result)",
        "description": "M1 + signed radial velocity toward hoop and lateral tangential speed",
    },
}

# Funnel accounting constants verified against executable pipeline and raw dynamic events
FUNNEL_STATS: Dict[str, Any] = {
    "total_raw_events": 1459,
    "unlocated_excluded": 26,
    "fouled_misses_excluded": 111,
    "eligible_shots_retained": 1322,
    "observed_makes": 597,
    "observed_misses": 725,
    "observed_make_rate": 597 / 1322,  # 0.4515885...
    "fouled_makes_retained": 32,
    "total_fouled_events": 143,
    "total_games": 10,
}


def load_merged_shot_data(base_path: Path | str = ".") -> pd.DataFrame:
    """Load and join processed feature CSVs and out-of-fold predictions.

    Verifies strict 1:1 join integrity on shot_id, ensuring exactly 1,322 rows.
    """
    base = Path(base_path)
    p3a_oof_path = base / "reports" / "phase3a" / "phase3a_oof_predictions.csv"
    p15_oof_path = base / "reports" / "phase1_5_oof_predictions.csv"
    p2_feat_path = base / "data" / "processed" / "phase2_features.csv"

    if not p3a_oof_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {p3a_oof_path}")
    if not p15_oof_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {p15_oof_path}")
    if not p2_feat_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {p2_feat_path}")

    df_3a = pd.read_csv(p3a_oof_path)
    df_15 = pd.read_csv(p15_oof_path)
    df_p2 = pd.read_csv(p2_feat_path)

    # Verify input row counts
    if len(df_3a) != 1322 or len(df_15) != 1322 or len(df_p2) != 1322:
        raise ValueError(
            f"Expected exactly 1,322 rows in all artifacts. "
            f"Got: df_3a={len(df_3a)}, df_15={len(df_15)}, df_p2={len(df_p2)}"
        )

    # 1:1 Join Phase 3A OOF with Phase 1.5 M0 probabilities
    merged = df_3a.merge(
        df_15[["shot_id", "m0_prob"]],
        on="shot_id",
        how="inner",
        validate="1:1",
    )

    # 1:1 Join with Phase 2 geometric & kinematic features
    p2_subset = df_p2[
        [
            "id",
            "dist_release",
            "contest_angle_deg",
            "defender_closing_velocity",
            "shooter_velocity_magnitude",
            "both_detected_pre",
        ]
    ].rename(columns={"id": "shot_id"})

    merged = merged.merge(
        p2_subset,
        on="shot_id",
        how="inner",
        validate="1:1",
    )

    if len(merged) != 1322:
        raise ValueError(f"Post-join row count mismatch! Expected 1,322 rows, got {len(merged)}")

    # Add friendly display columns
    merged["outcome_label"] = merged["observed_outcome"].map({1: "Made", 0: "Missed"})
    merged["tracking_status"] = merged["both_detected_pre10"].map(
        {True: "Directly Detected", False: "Extrapolated / Occluded"}
    )
    merged["prob_diff_m1_m0"] = merged["m1_prob_raw"] - merged["m0_prob"]
    merged["prob_diff_speed_m1"] = merged["speed_rel_prob_raw"] - merged["m1_prob_raw"]

    return merged


def load_phase3a_metrics_json(base_path: Path | str = ".") -> Dict[str, Any]:
    """Load the Phase 3A stored metrics JSON artifact."""
    base = Path(base_path)
    json_path = base / "reports" / "phase3a" / "phase3a_metrics.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_per_game_metrics(base_path: Path | str = ".") -> pd.DataFrame:
    """Load per-game metrics from phase1_5_per_game_metrics.csv."""
    base = Path(base_path)
    csv_path = base / "reports" / "phase1_5_per_game_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {csv_path}")
    return pd.read_csv(csv_path)
