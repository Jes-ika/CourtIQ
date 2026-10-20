"""Execution script for CourtIQ Phase 2: evaluation, ablations, calibration, and sensitivity analysis."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from courtiq.data.loader import find_match_directories, load_shots_dataframe
from courtiq.evaluation.metrics import (
    calculate_expected_calibration_error,
    compute_binary_classification_metrics,
)
from courtiq.evaluation.phase2_cv import MODEL_CONFIGS, run_phase2_experiment
from courtiq.features.pipeline import ShotDataPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("courtiq.phase2")


def generate_reliability_diagrams(
    y_true: np.ndarray,
    oof_preds: Dict[str, np.ndarray],
    oof_cal_preds: Dict[str, np.ndarray],
    output_png: Path | str,
    n_bins: int = 10,
) -> None:
    """Generate reliability diagram comparing M0, M1, M2 uncalibrated and calibrated."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    colors = {
        "M0_baseline": "#1f77b4",
        "M1_context": "#ff7f0e",
        "M2_full": "#2ca02c",
    }
    labels = {
        "M0_baseline": "M0 Baseline",
        "M1_context": "M1 Release Context",
        "M2_full": "M2 Kinematics & Geometry",
    }

    # Plot 1: Uncalibrated
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)

    for m_key in ["M0_baseline", "M1_context", "M2_full"]:
        probs = oof_preds[m_key]
        bin_idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
        bin_accs = []
        bin_confs = []
        for b in range(n_bins):
            mask = bin_idx == b
            if np.sum(mask) > 0:
                bin_accs.append(np.mean(y_true[mask]))
                bin_confs.append(np.mean(probs[mask]))
            else:
                bin_accs.append(np.nan)
                bin_confs.append(np.nan)

        ece_val = calculate_expected_calibration_error(y_true, probs, n_bins=n_bins)
        ax1.plot(
            bin_confs,
            bin_accs,
            marker="o",
            linewidth=2,
            color=colors[m_key],
            label=f"{labels[m_key]} (ECE={ece_val:.4f})",
        )

    ax1.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax1.set_ylabel("Observed Fraction of Made Shots", fontsize=12)
    ax1.set_title("Uncalibrated Reliability Diagram (10 Bins)", fontsize=14, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Plot 2: Platt Calibrated
    ax2 = axes[1]
    ax2.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)

    for m_key in ["M0_baseline", "M1_context", "M2_full"]:
        probs = oof_cal_preds[m_key]
        bin_idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
        bin_accs = []
        bin_confs = []
        for b in range(n_bins):
            mask = bin_idx == b
            if np.sum(mask) > 0:
                bin_accs.append(np.mean(y_true[mask]))
                bin_confs.append(np.mean(probs[mask]))
            else:
                bin_accs.append(np.nan)
                bin_confs.append(np.nan)

        ece_val = calculate_expected_calibration_error(y_true, probs, n_bins=n_bins)
        ax2.plot(
            bin_confs,
            bin_accs,
            marker="s",
            linewidth=2,
            linestyle="--",
            color=colors[m_key],
            label=f"{labels[m_key]} [Platt] (ECE={ece_val:.4f})",
        )

    ax2.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax2.set_ylabel("Observed Fraction of Made Shots", fontsize=12)
    ax2.set_title("In-Fold Platt Calibrated Reliability (10 Bins)", fontsize=14, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    logger.info(f"Saved reliability diagram to {output_png}")


def main() -> None:
    scratch = r"C:\Users\jesik\.gemini\antigravity-ide\brain\6e657e94-5888-4246-919f-b9658277d018\scratch\opendata-basketball"
    matches_dir = Path(scratch) / "data" / "matches"
    match_dirs = find_match_directories(matches_dir)

    logger.info(f"Loading shots from {len(match_dirs)} matches...")
    df_raw = load_shots_dataframe(match_dirs)
    pipeline = ShotDataPipeline()
    df_clean, exclusions = pipeline.clean_and_filter(df_raw)

    feat_path = Path("data/processed/phase2_features.csv")
    if not feat_path.exists():
        logger.error(f"Features file {feat_path} does not exist!")
        return

    df_kin = pd.read_csv(feat_path)
    logger.info(f"Loaded {len(df_kin)} kinematic rows.")

    logger.info("Executing Phase 2 Leave-One-Game-Out cross-validation across all models...")
    exp = run_phase2_experiment(df_clean, df_kin, random_state=42)

    os.makedirs("reports", exist_ok=True)

    # 1. Save out-of-fold predictions CSV
    df_oof = pd.DataFrame({
        "game_id": exp["merged_dataframe"]["gameId"].values,
        "shot_id": exp["merged_dataframe"]["id"].values,
        "observed_outcome": exp["merged_dataframe"]["outcome"].astype(int).values,
        "m0_prob": np.round(exp["oof_predictions"]["M0_baseline"], 5),
        "m1_prob": np.round(exp["oof_predictions"]["M1_context"], 5),
        "m2_prob": np.round(exp["oof_predictions"]["M2_full"], 5),
        "m2_prob_calibrated": np.round(exp["oof_calibrated_predictions"]["M2_full"], 5),
        "closing_only_prob": np.round(exp["oof_predictions"]["M2_ablation_closing_only"], 5),
        "angle_only_prob": np.round(exp["oof_predictions"]["M2_ablation_angle_only"], 5),
        "shooter_vel_only_prob": np.round(exp["oof_predictions"]["M2_ablation_shooter_vel_only"], 5),
        "delta_dist_prob": np.round(exp["oof_predictions"]["M2_ablation_delta_dist"], 5),
        "both_detected_pre": exp["merged_dataframe"]["both_detected_pre"].values,
        "defender_closing_velocity": np.round(exp["merged_dataframe"]["defender_closing_velocity"].values, 3),
        "contest_angle_cos": np.round(exp["merged_dataframe"]["contest_angle_cos"].values, 3),
        "shooter_velocity_magnitude": np.round(exp["merged_dataframe"]["shooter_velocity_magnitude"].values, 3),
    })
    oof_csv_path = Path("reports/phase2_oof_predictions.csv")
    df_oof.to_csv(oof_csv_path, index=False)
    logger.info(f"Saved Phase 2 OOF predictions to {oof_csv_path} ({len(df_oof)} rows)")

    # 2. Save Ablation Results CSV
    ablation_rows = []
    m1_uncal = exp["overall_uncalibrated"]["M1_context"]
    m1_cal = exp["overall_calibrated"]["M1_context"]

    for m_key, cfg in MODEL_CONFIGS.items():
        uncal = exp["overall_uncalibrated"][m_key]
        cal = exp["overall_calibrated"][m_key]

        delta_ll = uncal["log_loss"] - m1_uncal["log_loss"]
        delta_br = uncal["brier_score"] - m1_uncal["brier_score"]
        delta_auc = uncal["roc_auc"] - m1_uncal["roc_auc"] if uncal["roc_auc"] and m1_uncal["roc_auc"] else None

        ablation_rows.append({
            "model_key": m_key,
            "description": cfg["description"],
            "features_count": len(cfg["numeric"]) + len(cfg["categorical"]),
            "log_loss_uncal": round(uncal["log_loss"], 6),
            "delta_log_loss_vs_m1": round(delta_ll, 6),
            "brier_score_uncal": round(uncal["brier_score"], 6),
            "delta_brier_vs_m1": round(delta_br, 6),
            "roc_auc_uncal": round(uncal["roc_auc"], 6) if uncal["roc_auc"] else None,
            "delta_roc_auc_vs_m1": round(delta_auc, 6) if delta_auc is not None else None,
            "ece_5_uncal": round(uncal["ece_5"], 6),
            "ece_10_uncal": round(uncal["ece_10"], 6),
            "ece_15_uncal": round(uncal["ece_15"], 6),
            "log_loss_platt_cal": round(cal["log_loss"], 6),
            "brier_score_platt_cal": round(cal["brier_score"], 6),
            "ece_10_platt_cal": round(cal["ece_10"], 6),
        })

    df_ablations = pd.DataFrame(ablation_rows)
    ablation_csv_path = Path("reports/phase2_ablation_results.csv")
    df_ablations.to_csv(ablation_csv_path, index=False)
    logger.info(f"Saved ablation results to {ablation_csv_path}")

    # 3. Save Machine-Readable JSON Metrics
    json_metrics = {
        "n_games": exp["n_games"],
        "n_total_shots": exp["n_shots"],
        "games_list": exp["games_list"],
        "overall_uncalibrated": exp["overall_uncalibrated"],
        "overall_calibrated": exp["overall_calibrated"],
        "per_game_comparisons": exp["per_game_comparisons"],
        "m2_improvement_counts": exp["m2_improvement_counts"],
        "bootstrap_m2_minus_m1_delta_log_loss": exp["bootstrap_m2_minus_m1_delta_log_loss"],
        "detection_stratified_evaluation": exp["detection_stratified_evaluation"],
    }
    json_metrics_path = Path("reports/phase2_model_metrics.json")
    with open(json_metrics_path, "w", encoding="utf-8") as f:
        json.dump(json_metrics, f, indent=2)
    logger.info(f"Saved model metrics JSON to {json_metrics_path}")

    # 4. Generate Reliability Diagram PNG
    png_path = Path("reports/phase2_calibration_reliability.png")
    generate_reliability_diagrams(
        y_true=exp["merged_dataframe"]["outcome"].astype(int).values,
        oof_preds=exp["oof_predictions"],
        oof_cal_preds=exp["oof_calibrated_predictions"],
        output_png=png_path,
        n_bins=10,
    )

    print("\n" + "=" * 95)
    print(" CourtIQ Phase 2 Model Comparison: M0 vs M1 vs M2 and Kinematic Ablations")
    print("=" * 95)
    print(f"{'Model Architecture':<30} | {'Log Loss':<10} | {'Delta M1':<10} | {'Brier':<9} | {'ROC-AUC':<9} | {'ECE(10)':<9}")
    print("-" * 95)
    for row in ablation_rows:
        auc_str = f"{row['roc_auc_uncal']:.4f}" if row['roc_auc_uncal'] else "N/A"
        print(f"{row['model_key']:<30} | {row['log_loss_uncal']:<10.6f} | {row['delta_log_loss_vs_m1']:<10.6f} | {row['brier_score_uncal']:<9.6f} | {auc_str:<9} | {row['ece_10_uncal']:<9.6f}")
    print("-" * 95)


if __name__ == "__main__":
    main()
