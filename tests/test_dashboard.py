"""Unit and regression tests for CourtIQ dashboard data loader and metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pytest
import numpy as np
import pandas as pd

from app.data_loader import (
    AUTHORITATIVE_METRICS,
    FUNNEL_STATS,
    load_merged_shot_data,
    load_per_game_metrics,
    load_phase3a_metrics_json,
)
from courtiq.evaluation.metrics import calculate_expected_calibration_error


def test_authoritative_metrics_consistency():
    """Verify that dashboard constants exactly match stored benchmark JSON artifacts."""
    with open("reports/baseline_results.json", "r", encoding="utf-8") as f:
        baseline_json = json.load(f)
    with open("reports/phase3a/phase3a_metrics.json", "r", encoding="utf-8") as f:
        phase3a_json = json.load(f)

    # Check M0
    m0_dict = AUTHORITATIVE_METRICS["M0_baseline"]
    assert m0_dict["log_loss"] == pytest.approx(baseline_json["overall_m0"]["log_loss"], abs=1e-6)
    assert m0_dict["brier_score"] == pytest.approx(baseline_json["overall_m0"]["brier_score"], abs=1e-6)
    assert m0_dict["roc_auc"] == pytest.approx(baseline_json["overall_m0"]["roc_auc"], abs=1e-6)
    assert m0_dict["log_loss"] == 0.655161
    assert m0_dict["brier_score"] == 0.231456
    assert m0_dict["roc_auc"] == 0.641026
    assert m0_dict["ece_10"] == 0.030552

    # Check M1
    m1_dict = AUTHORITATIVE_METRICS["M1_benchmark"]
    m1_stored = phase3a_json["pooled_raw_metrics"]["M1_benchmark"]
    assert m1_dict["log_loss"] == pytest.approx(m1_stored["log_loss"], abs=1e-6)
    assert m1_dict["brier_score"] == pytest.approx(m1_stored["brier_score"], abs=1e-6)
    assert m1_dict["roc_auc"] == pytest.approx(m1_stored["roc_auc"], abs=1e-6)
    assert m1_dict["ece_10"] == pytest.approx(m1_stored["ece_10"], abs=1e-6)
    assert m1_dict["log_loss"] == 0.617270

    # Check M1 + Speed Rel
    rel_dict = AUTHORITATIVE_METRICS["M1_plus_speed_rel"]
    rel_stored = phase3a_json["pooled_raw_metrics"]["M1_plus_speed_rel"]
    assert rel_dict["log_loss"] == pytest.approx(rel_stored["log_loss"], abs=1e-6)
    assert rel_dict["brier_score"] == pytest.approx(rel_stored["brier_score"], abs=1e-6)
    assert rel_dict["roc_auc"] == pytest.approx(rel_stored["roc_auc"], abs=1e-6)
    assert rel_dict["delta_vs_m1"] == pytest.approx(-0.000758, abs=1e-6)


def test_funnel_stats_accounting():
    """Verify that funnel stats match audited sequential pipeline exclusions."""
    assert FUNNEL_STATS["total_raw_events"] == 1459
    assert FUNNEL_STATS["unlocated_excluded"] == 26
    assert FUNNEL_STATS["fouled_misses_excluded"] == 111
    assert FUNNEL_STATS["eligible_shots_retained"] == 1322
    assert FUNNEL_STATS["observed_makes"] == 597
    assert FUNNEL_STATS["observed_misses"] == 725
    assert FUNNEL_STATS["fouled_makes_retained"] == 32
    assert FUNNEL_STATS["observed_makes"] + FUNNEL_STATS["observed_misses"] == 1322


def test_load_merged_shot_data_integrity():
    """Verify 1:1 join integrity and schema correctness of loaded shots."""
    df = load_merged_shot_data(".")
    assert len(df) == 1322
    assert df["shot_id"].nunique() == 1322
    assert df["observed_outcome"].isin([0, 1]).all()
    assert df["observed_outcome"].sum() == 597

    # Verify probability bounds
    for col in ["m0_prob", "m1_prob_raw", "m1_prob_cal", "speed_rel_prob_raw"]:
        assert df[col].between(0.0, 1.0).all()
        assert not df[col].isnull().any()

    # Verify tracking status and kinematic fields
    assert df["dist_release"].between(0.0, 50.0).all()
    assert df["shooter_speed_rel"].between(0.0, 35.0).all()
    assert set(df["tracking_status"].unique()) == {"Directly Detected", "Extrapolated / Occluded"}


def test_per_game_metrics_integrity():
    """Verify that per-game metrics cover all 10 matches without duplicate or missing games."""
    df_games = load_per_game_metrics(".")
    assert len(df_games) == 10
    assert df_games["game_id"].nunique() == 10
    assert df_games["shots"].sum() == 1322
    assert (df_games["delta_log_loss"] < 0).sum() == 9  # M1 wins 9 of 10 games


def test_interactive_calibration_calculation():
    """Verify that ECE computed on loaded predictions reproduces official benchmark figures."""
    df = load_merged_shot_data(".")
    y_true = df["observed_outcome"].values

    ece_m0_5 = calculate_expected_calibration_error(y_true, df["m0_prob"].values, n_bins=5)
    ece_m1_5 = calculate_expected_calibration_error(y_true, df["m1_prob_raw"].values, n_bins=5)
    assert ece_m0_5 == pytest.approx(0.020006, abs=1e-5)
    assert ece_m1_5 == pytest.approx(0.042527, abs=1e-5)

    ece_m0_10 = calculate_expected_calibration_error(y_true, df["m0_prob"].values, n_bins=10)
    ece_m1_10 = calculate_expected_calibration_error(y_true, df["m1_prob_raw"].values, n_bins=10)
    assert ece_m0_10 == pytest.approx(0.030552, abs=1e-5)
    assert ece_m1_10 == pytest.approx(0.051075, abs=1e-5)
