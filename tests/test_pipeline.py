"""Unit tests for the CourtIQ data cleaning and feature pipeline."""

import pytest
import pandas as pd
import numpy as np

from courtiq.features.pipeline import ShotDataPipeline


def test_pipeline_normal_execution(sample_synthetic_shots_df):
    pipeline = ShotDataPipeline()
    df_clean, exclusions = pipeline.clean_and_filter(sample_synthetic_shots_df)

    assert len(df_clean) == len(sample_synthetic_shots_df)
    assert exclusions["total_dropped"] == 0

    X_m0, X_m1, y, groups = pipeline.extract_features(df_clean)

    assert len(X_m0) == len(df_clean)
    assert len(X_m1) == len(df_clean)
    assert len(y) == len(df_clean)
    assert len(groups) == len(df_clean)
    assert set(y.unique()).issubset({0, 1})
    assert "distance" in X_m0.columns
    assert "closestDefDist" in X_m1.columns
    assert "closestDefDist" not in X_m0.columns


def test_pipeline_missing_labels(sample_synthetic_shots_df):
    df_corrupted = sample_synthetic_shots_df.copy()
    df_corrupted["outcome"] = df_corrupted["outcome"].astype(object)
    df_corrupted.loc[0, "outcome"] = None
    df_corrupted.loc[2, "outcome"] = np.nan

    pipeline = ShotDataPipeline()
    df_clean, exclusions = pipeline.clean_and_filter(df_corrupted)

    assert len(df_clean) == len(sample_synthetic_shots_df) - 2
    assert exclusions["missing_target_label"] == 2
    assert exclusions["total_dropped"] == 2
    assert df_clean["outcome"].isnull().sum() == 0


def test_pipeline_duplicate_events(sample_synthetic_shots_df):
    df_duplicated = pd.concat([sample_synthetic_shots_df, sample_synthetic_shots_df.iloc[[0, 1]]], ignore_index=True)

    pipeline = ShotDataPipeline()
    df_clean, exclusions = pipeline.clean_and_filter(df_duplicated)

    assert len(df_clean) == len(sample_synthetic_shots_df)
    assert exclusions["duplicate_shot_id"] == 2
    assert df_clean["id"].duplicated().sum() == 0


def test_pipeline_unlocated_shots(sample_synthetic_shots_df):
    df_unlocated = sample_synthetic_shots_df.copy()
    df_unlocated.loc[1, "distance"] = np.nan
    df_unlocated.loc[3, "location"] = None

    pipeline = ShotDataPipeline(require_valid_coordinates=True)
    df_clean, exclusions = pipeline.clean_and_filter(df_unlocated)

    assert len(df_clean) == len(sample_synthetic_shots_df) - 2
    assert exclusions["unlocated_or_missing_coordinates"] == 2


def test_pipeline_fouled_misses_filtering(sample_synthetic_shots_df):
    df_fouled = sample_synthetic_shots_df.copy()
    # Mark shot 0 as fouled made (and-one) -> should be KEPT
    df_fouled.loc[0, "fouled"] = True
    df_fouled.loc[0, "outcome"] = True

    # Mark shot 1 as fouled missed (free throws awarded, not FGA in FIBA) -> should be DROPPED
    df_fouled.loc[1, "fouled"] = True
    df_fouled.loc[1, "outcome"] = False

    pipeline = ShotDataPipeline(exclude_fouled_misses=True)
    df_clean, exclusions = pipeline.clean_and_filter(df_fouled)

    assert len(df_clean) == len(sample_synthetic_shots_df) - 1
    assert exclusions["fouled_miss_not_fga"] == 1
    assert "shot-101-1" in df_clean["id"].values
    assert "shot-101-2" not in df_clean["id"].values


def test_feature_leakage_prevention(sample_synthetic_shots_df):
    """Verify that attempting to configure a leaked post-shot column raises ValueError."""
    # Attempt to smuggle 'shotQuality' or 'outcome' or 'assisted' into M0
    with pytest.raises(ValueError, match="Feature leakage detected"):
        ShotDataPipeline(m0_numeric=["distance", "shotQuality"])

    with pytest.raises(ValueError, match="Feature leakage detected"):
        ShotDataPipeline(m0_categorical=["shotType", "outcome"])


def test_empty_and_malformed_input():
    pipeline = ShotDataPipeline()

    # Empty DataFrame
    df_empty = pd.DataFrame()
    df_clean, exclusions = pipeline.clean_and_filter(df_empty)
    assert df_clean.empty
    assert exclusions["total_input"] == 0

    # Malformed DataFrame missing 'outcome'
    df_no_outcome = pd.DataFrame({"id": ["shot-1"], "distance": [15.0]})
    with pytest.raises(KeyError, match="outcome"):
        pipeline.clean_and_filter(df_no_outcome)


def test_outcome_encoding(sample_synthetic_shots_df):
    pipeline = ShotDataPipeline()
    df_clean, _ = pipeline.clean_and_filter(sample_synthetic_shots_df)
    _, _, y, _ = pipeline.extract_features(df_clean)

    # Must be integer type (0 or 1), not boolean or object
    assert y.dtype in (np.int32, np.int64, int)
    assert set(y.unique()).issubset({0, 1})
