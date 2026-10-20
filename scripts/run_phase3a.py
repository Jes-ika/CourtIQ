"""Phase 3A execution script: nested calibration, directional velocity experiment, and uncertainty estimation."""

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
from courtiq.evaluation.metrics import calculate_expected_calibration_error
from courtiq.evaluation.phase3a_cv import PHASE3A_MODELS, run_phase3a_evaluation
from courtiq.features.pipeline import ShotDataPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("courtiq.phase3a")


def plot_phase3a_reliability(
    y_true: np.ndarray,
    raw_preds: Dict[str, np.ndarray],
    cal_preds: Dict[str, np.ndarray],
    output_png: Path | str,
    n_bins: int = 10,
) -> None:
    """Generate reliability curves comparing uncalibrated vs nested cross-fitted Platt calibrated models."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    colors = {
        "M1_benchmark": "#1f77b4",
        "M1_plus_speed_rel": "#2ca02c",
        "M1_plus_directional": "#d62728",
        "M1_plus_speed_early": "#9467bd",
    }
    labels = {
        "M1_benchmark": "M1 Benchmark",
        "M1_plus_speed_rel": "M1 + Release Speed",
        "M1_plus_directional": "M1 + Directional Speed",
        "M1_plus_speed_early": "M1 + Early Speed",
    }

    # Plot 1: Uncalibrated
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)

    for m_key in ["M1_benchmark", "M1_plus_speed_rel", "M1_plus_directional", "M1_plus_speed_early"]:
        probs = raw_preds[m_key]
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
    ax1.set_title("Uncalibrated Reliability (Outer Test Games, 10 Bins)", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Plot 2: Nested Cross-Fitted Platt Calibrated
    ax2 = axes[1]
    ax2.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)

    for m_key in ["M1_benchmark", "M1_plus_speed_rel", "M1_plus_directional", "M1_plus_speed_early"]:
        probs = cal_preds[m_key]
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
            label=f"{labels[m_key]} [Nested Platt] (ECE={ece_val:.4f})",
        )

    ax2.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax2.set_ylabel("Observed Fraction of Made Shots", fontsize=12)
    ax2.set_title("Nested Cross-Fitted Platt Calibration (10 Bins)", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    logger.info(f"Saved Phase 3A reliability plot to {output_png}")


def main() -> None:
    scratch = r"C:\Users\jesik\.gemini\antigravity-ide\brain\6e657e94-5888-4246-919f-b9658277d018\scratch\opendata-basketball"
    matches_dir = Path(scratch) / "data" / "matches"
    match_dirs = find_match_directories(matches_dir)

    logger.info(f"Loading shots from {len(match_dirs)} matches...")
    df_raw = load_shots_dataframe(match_dirs)
    pipeline = ShotDataPipeline()
    df_clean, exclusions = pipeline.clean_and_filter(df_raw)

    feat_path = Path("data/processed/phase3a_features.csv")
    if not feat_path.exists():
        logger.error(f"Features file {feat_path} does not exist!")
        return

    df_3a = pd.read_csv(feat_path)
    logger.info(f"Loaded {len(df_3a)} Phase 3A feature rows.")

    # Merge Phase 3A features onto cleaned shots ensuring exact 1:1 match
    df_merged = df_clean.merge(
        df_3a[
            [
                "id",
                "shooter_speed_rel",
                "shooter_speed_early",
                "shooter_vel_radial",
                "shooter_vel_tangential",
                "both_detected_pre10",
                "s_detected_rel",
                "s_detected_pre10",
                "s_detected_pre20",
            ]
        ],
        on="id",
        how="inner",
    )

    logger.info(f"Merged dataset shape: {df_merged.shape}. Running nested Phase 3A evaluation...")
    res = run_phase3a_evaluation(df_merged, random_state=42)

    out_dir = Path("reports/phase3a")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save OOF predictions CSV
    df_oof = pd.DataFrame({
        "game_id": df_merged["gameId"].values,
        "shot_id": df_merged["id"].values,
        "observed_outcome": df_merged["outcome"].astype(int).values,
        "m1_prob_raw": np.round(res["raw_oof_preds"]["M1_benchmark"], 5),
        "m1_prob_cal": np.round(res["cal_oof_preds"]["M1_benchmark"], 5),
        "speed_rel_prob_raw": np.round(res["raw_oof_preds"]["M1_plus_speed_rel"], 5),
        "speed_rel_prob_cal": np.round(res["cal_oof_preds"]["M1_plus_speed_rel"], 5),
        "directional_prob_raw": np.round(res["raw_oof_preds"]["M1_plus_directional"], 5),
        "directional_prob_cal": np.round(res["cal_oof_preds"]["M1_plus_directional"], 5),
        "speed_early_prob_raw": np.round(res["raw_oof_preds"]["M1_plus_speed_early"], 5),
        "speed_early_prob_cal": np.round(res["cal_oof_preds"]["M1_plus_speed_early"], 5),
        "radial_only_prob_raw": np.round(res["raw_oof_preds"]["M1_plus_radial_only"], 5),
        "tangential_only_prob_raw": np.round(res["raw_oof_preds"]["M1_plus_tangential_only"], 5),
        "both_detected_pre10": df_merged["both_detected_pre10"].values,
        "shooter_speed_rel": np.round(df_merged["shooter_speed_rel"].values, 3),
        "shooter_speed_early": np.round(df_merged["shooter_speed_early"].values, 3),
        "shooter_vel_radial": np.round(df_merged["shooter_vel_radial"].values, 3),
        "shooter_vel_tangential": np.round(df_merged["shooter_vel_tangential"].values, 3),
    })
    oof_path = out_dir / "phase3a_oof_predictions.csv"
    df_oof.to_csv(oof_path, index=False)
    logger.info(f"Saved Phase 3A OOF predictions to {oof_path}")

    # 2. Save Metrics JSON
    metrics_path = out_dir / "phase3a_metrics.json"
    clean_json = {
        "n_games": res["n_games"],
        "n_total_shots": res["n_total_shots"],
        "games_list": res["games_list"],
        "pooled_raw_metrics": res["pooled_raw_metrics"],
        "pooled_calibrated_metrics": res["pooled_calibrated_metrics"],
        "bootstrap_results": res["bootstrap_results"],
        "per_game_summary": res["per_game_summary"],
        "calibration_provenance": res["calibration_provenance"],
        "detection_sensitivity": res["detection_sensitivity"],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(clean_json, f, indent=2)
    logger.info(f"Saved Phase 3A metrics JSON to {metrics_path}")

    # 3. Generate Reliability Diagram
    png_path = out_dir / "calibration_reliability.png"
    plot_phase3a_reliability(
        y_true=df_merged["outcome"].astype(int).values,
        raw_preds=res["raw_oof_preds"],
        cal_preds=res["cal_oof_preds"],
        output_png=png_path,
        n_bins=10,
    )

    # Print summary benchmark table
    print("\n" + "=" * 105)
    print(" CourtIQ Phase 3A Experiment: Nested Calibration & Directional Velocity Evaluation")
    print("=" * 105)
    print(f"{'Model Architecture':<26} | {'Raw LL':<9} | {'Delta LL':<9} | {'Raw Brier':<9} | {'Raw AUC':<9} | {'Raw ECE10':<9} | {'Nested Cal LL':<13} | {'Cal ECE10':<9}")
    print("-" * 105)

    base_ll = res["pooled_raw_metrics"]["M1_benchmark"]["log_loss"]
    for name, cfg in PHASE3A_MODELS.items():
        rm = res["pooled_raw_metrics"][name]
        cm = res["pooled_calibrated_metrics"][name]
        d_ll = rm["log_loss"] - base_ll
        auc_str = f"{rm['roc_auc']:.4f}" if rm['roc_auc'] else "N/A"
        print(f"{name:<26} | {rm['log_loss']:<9.6f} | {d_ll:<+9.6f} | {rm['brier_score']:<9.6f} | {auc_str:<9} | {rm['ece_10']:<9.6f} | {cm['log_loss']:<13.6f} | {cm['ece_10']:<9.6f}")
    print("-" * 105)


if __name__ == "__main__":
    main()
