"""Data cleaning, filtering, and feature extraction pipeline."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default strict leakage blacklist
DEFAULT_LEAKAGE_BLACKLIST = [
    "outcome",
    "shotQuality",
    "fouled",
    "foulId",
    "assisted",
    "assistOpp",
    "passerId",
    "blocked",
    "blockerId",
    "endFrame",
    "endWallClock",
    "endGameClock",
]

DEFAULT_M0_NUMERIC = ["distance", "shotClock", "releaseTime", "dribblesBefore"]
DEFAULT_M0_CATEGORICAL = ["shotType", "three", "catchAndShoot", "createdFromPaint", "period"]
DEFAULT_M1_DEFENSIVE_NUMERIC = ["closestDefDist"]
DEFAULT_M1_DEFENSIVE_CATEGORICAL = ["contestLevel", "contested"]


class ShotDataPipeline:
    """Validated, reproducible transformation pipeline for basketball shot data."""

    def __init__(
        self,
        exclude_fouled_misses: bool = True,
        require_valid_coordinates: bool = True,
        leakage_blacklist: Optional[List[str]] = None,
        m0_numeric: Optional[List[str]] = None,
        m0_categorical: Optional[List[str]] = None,
        m1_defensive_numeric: Optional[List[str]] = None,
        m1_defensive_categorical: Optional[List[str]] = None,
    ):
        self.exclude_fouled_misses = exclude_fouled_misses
        self.require_valid_coordinates = require_valid_coordinates
        self.leakage_blacklist = set(leakage_blacklist or DEFAULT_LEAKAGE_BLACKLIST)
        self.m0_numeric = m0_numeric or DEFAULT_M0_NUMERIC
        self.m0_categorical = m0_categorical or DEFAULT_M0_CATEGORICAL
        self.m1_defensive_numeric = m1_defensive_numeric or DEFAULT_M1_DEFENSIVE_NUMERIC
        self.m1_defensive_categorical = m1_defensive_categorical or DEFAULT_M1_DEFENSIVE_CATEGORICAL

        # Proactively check configured features against leakage blacklist
        configured_features = (
            self.m0_numeric
            + self.m0_categorical
            + self.m1_defensive_numeric
            + self.m1_defensive_categorical
        )
        for col in configured_features:
            if col in self.leakage_blacklist:
                raise ValueError(
                    f"Feature leakage detected! Configured feature '{col}' is in leakage blacklist."
                )

    def clean_and_filter(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Clean and filter raw shot events, recording explicit exclusion reasons."""
        if df.empty:
            return df.copy(), {"total_input": 0, "total_output": 0}

        initial_count = len(df)
        exclusions = {"total_input": initial_count}
        df_clean = df.copy()

        # 1. Missing target label (outcome)
        if "outcome" in df_clean.columns:
            missing_outcome = df_clean["outcome"].isnull()
            n_missing_outcome = int(missing_outcome.sum())
            exclusions["missing_target_label"] = n_missing_outcome
            if n_missing_outcome > 0:
                df_clean = df_clean[~missing_outcome]
        else:
            raise KeyError("Target column 'outcome' not found in input DataFrame")

        # 2. Duplicate shot IDs
        if "id" in df_clean.columns:
            dup_mask = df_clean.duplicated(subset=["id"], keep="first")
            n_duplicates = int(dup_mask.sum())
            exclusions["duplicate_shot_id"] = n_duplicates
            if n_duplicates > 0:
                df_clean = df_clean[~dup_mask]

        # 3. Missing gameId
        if "gameId" in df_clean.columns:
            missing_gid = df_clean["gameId"].isnull()
            n_missing_gid = int(missing_gid.sum())
            exclusions["missing_game_id"] = n_missing_gid
            if n_missing_gid > 0:
                df_clean = df_clean[~missing_gid]

        # 4. Unlocated / missing spatial coordinates (distance or location)
        if self.require_valid_coordinates:
            missing_coord = pd.Series(False, index=df_clean.index)
            if "distance" in df_clean.columns:
                missing_coord = missing_coord | df_clean["distance"].isnull()
            if "location" in df_clean.columns:
                missing_coord = missing_coord | df_clean["location"].isnull()
            n_unlocated = int(missing_coord.sum())
            exclusions["unlocated_or_missing_coordinates"] = n_unlocated
            if n_unlocated > 0:
                df_clean = df_clean[~missing_coord]

        # 5. Fouled misses (FIBA official box-score convention: free throws awarded, not an FGA)
        if self.exclude_fouled_misses and "fouled" in df_clean.columns:
            fouled_miss = (df_clean["fouled"] == True) & (df_clean["outcome"] == False)
            n_fouled_misses = int(fouled_miss.sum())
            exclusions["fouled_miss_not_fga"] = n_fouled_misses
            if n_fouled_misses > 0:
                df_clean = df_clean[~fouled_miss]

        exclusions["total_dropped"] = initial_count - len(df_clean)
        exclusions["total_output"] = len(df_clean)

        return df_clean, exclusions

    def extract_features(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Extract M0 baseline features, M1 context features, target, and game group series.

        Returns:
            X_m0: Features for baseline model (distance, basic shot characteristics).
            X_m1: Features for context model (M0 + defensive context).
            y: Target binary series (1 for made, 0 for missed).
            groups: Series of game IDs for grouped cross-validation.
        """
        # Target variable encoding (boolean to int: 1=made, 0=missed)
        y = df["outcome"].astype(int)
        groups = df["gameId"].astype(str)

        # M0 features
        m0_cols = [c for c in self.m0_numeric + self.m0_categorical if c in df.columns]
        # Safety check: ensure no leaked column is in m0_cols
        for col in m0_cols:
            if col in self.leakage_blacklist:
                raise ValueError(f"Feature leakage detected! '{col}' is in leakage blacklist.")

        X_m0 = df[m0_cols].copy()

        # M1 features = M0 + defensive context
        m1_cols = m0_cols + [
            c
            for c in self.m1_defensive_numeric + self.m1_defensive_categorical
            if c in df.columns
        ]
        for col in m1_cols:
            if col in self.leakage_blacklist:
                raise ValueError(f"Feature leakage detected! '{col}' is in leakage blacklist.")

        X_m1 = df[m1_cols].copy()

        return X_m0, X_m1, y, groups
