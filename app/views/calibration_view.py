"""Page 4 — Calibration & Reliability: Visual reliability diagnostics, ECE bin sensitivity, and nested Platt scaling."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.data_loader import load_merged_shot_data
from courtiq.evaluation.metrics import calculate_expected_calibration_error


def render_calibration_page() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="color: #FFFFFF; font-size: 1.8rem; font-weight: 700; margin-bottom: 0.25rem;">
                🎯 Calibration & Reliability Diagnostics
            </h2>
            <p style="color: #9CA3AF; font-size: 1rem;">
                Evaluating probability reliability, sharpness tradeoffs, bin-count sensitivity, and nested cross-fitted shrinkage.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Core Scientific Principle
    st.markdown(
        """
        > **A fundamental principle of probabilistic sports analytics:** Discrimination (ROC-AUC) and calibration (ECE/reliability) measure distinct dimensions of model quality. A model that predicts a constant base rate ($\hat{p} = 0.4516$) achieves a near-zero Expected Calibration Error, but has zero discriminatory power.
        """
    )

    # Audited Static Plot Display
    st.subheader("1. Audited Publication Calibration Curves")
    st.markdown(
        "Below are the verified 10-bin reliability diagrams from the Phase 3A research report, comparing uncalibrated predictions against nested cross-fitted Platt calibrated models across all 1,322 outer-fold shots:"
    )

    png_path = Path("reports/phase3a/calibration_reliability.png")
    if png_path.exists():
        st.image(str(png_path), caption="Figure 1: Reliability curves for raw vs nested cross-fitted Platt scaling (10 equal-width bins).", use_container_width=True)
    else:
        st.warning("Calibration figure artifact not found at reports/phase3a/calibration_reliability.png.")

    # Interactive Bin Sensitivity Tool
    st.markdown("---")
    st.subheader("2. Interactive Calibration Curve & ECE Binning Sensitivity")
    st.markdown(
        "Expected Calibration Error (ECE) is mathematically sensitive to the number of probability bins. "
        "Use the selector below to interactively observe how ECE varies across bin granularities:"
    )

    k_bins = st.select_slider(
        "Select Number of Equal-Width Bins (K)",
        options=[5, 10, 15, 20],
        value=10,
    )

    try:
        df = load_merged_shot_data(".")
        y_true = df["observed_outcome"].values

        ece_m0 = calculate_expected_calibration_error(y_true, df["m0_prob"].values, n_bins=k_bins)
        ece_m1_raw = calculate_expected_calibration_error(y_true, df["m1_prob_raw"].values, n_bins=k_bins)
        ece_m1_cal = calculate_expected_calibration_error(y_true, df["m1_prob_cal"].values, n_bins=k_bins)
        ece_speed_raw = calculate_expected_calibration_error(y_true, df["speed_rel_prob_raw"].values, n_bins=k_bins)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"M0 Baseline ECE ({k_bins} bins)", f"{ece_m0:.6f}", help="Compressed around 0.45; minimal tail predictions")
        c2.metric(f"M1 Benchmark Raw ECE ({k_bins} bins)", f"{ece_m1_raw:.6f}", help="Higher tail confidence increases binomial bin variance")
        c3.metric(f"M1 Nested Calibrated ECE ({k_bins} bins)", f"{ece_m1_cal:.6f}", delta=f"{ece_m1_cal - ece_m1_raw:+.6f}")
        c4.metric(f"M1 + Speed Rel Raw ECE ({k_bins} bins)", f"{ece_speed_raw:.6f}")

        # Interactive Reliability Diagram Plot
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0E1117")
        ax.set_facecolor("#1E293B")
        bins = np.linspace(0.0, 1.0, k_bins + 1)

        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)", alpha=0.7, linewidth=1.5)

        models_to_plot = {
            "M0 Baseline": (df["m0_prob"].values, "#94A3B8"),
            "M1 Context (Raw)": (df["m1_prob_raw"].values, "#38BDF8"),
            "M1 Context (Calibrated)": (df["m1_prob_cal"].values, "#10B981"),
            "M1 + Speed Rel": (df["speed_rel_prob_raw"].values, "#F97316"),
        }

        for label, (probs, color) in models_to_plot.items():
            bin_idx = np.clip(np.digitize(probs, bins) - 1, 0, k_bins - 1)
            bin_accs = []
            bin_confs = []
            for b in range(k_bins):
                mask = bin_idx == b
                if np.sum(mask) > 0:
                    bin_accs.append(np.mean(y_true[mask]))
                    bin_confs.append(np.mean(probs[mask]))
            ax.plot(bin_confs, bin_accs, marker="o", markersize=5, label=f"{label}", color=color, linewidth=1.5)

        ax.set_xlabel("Mean Predicted Probability", color="#E2E8F0", fontsize=9)
        ax.set_ylabel("Observed Make Fraction", color="#E2E8F0", fontsize=9)
        ax.set_title(f"Reliability Curve Comparison ({k_bins} Equal-Width Bins, N=1,322 Shots)", color="#FFFFFF", fontsize=11, fontweight="bold")
        ax.tick_params(colors="#94A3B8")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.legend(loc="upper left", facecolor="#1E293B", edgecolor="#475569", labelcolor="#F1F5F9", fontsize=8)
        ax.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error computing interactive calibration: {e}")

    # Deep Methodological Dive on ECE Disparity
    st.markdown("---")
    st.subheader("3. Why Calibration is Not Universally Better for One Model")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
            ### The ECE & Sharpness Tradeoff
            - **Baseline M0 Shrinkage:**
              M0 predicts almost zero shots below $0.20$ ($n=14$) and zero shots above $0.80$ ($n=0$). By restricting predictions to the central range where empirical variance is lower, its absolute deviations $|\bar{y}_b - \bar{p}_b|$ appear small.
            - **Context Model M1 Dispersion:**
              M1 leverages defensive context to assign confident probabilities: $86$ shots below $0.20$ and $33$ shots above $0.80$. In finite sample bins with $N \in [30, 80]$, binomial sampling noise ($\text{SE} \approx \sqrt{p(1-p)/N} \approx 0.05$) inflates ECE.
            - **Proper Scoring Rules Trump ECE:**
              Proper scoring rules (Log Loss and Brier Score) penalize both miscalibration and lack of sharpness. M1 vastly outperforms M0 on both ($p < 0.001$).
            """
        )

    with col_b:
        st.markdown(
            """
            ### Nested Cross-Fitted Platt Calibration
            - **The Historical Flaw:**
              In early Phase 2 prototypes, resubstitution calibration fitted Platt scaling on the same training data used to train the base model. This learned an uninformative identity mapping ($a \approx 1.0$), incurring an out-of-fold generalization penalty ($+0.000857$ log loss).
            - **The Mathematical Solution:**
              Phase 3A implemented a **10-outer $\times$ 9-inner nested Leave-One-Game-Out cross-fitting architecture**.
            - **Learned Shrinkage Provenance:**
              The calibrator learns an average slope $a = 0.9110 \pm 0.0067 < 1.0$, systematically shrinking extreme log-odds toward the base rate ($45.16\%$) and reducing 10-bin ECE by **12.7%** ($0.0511 \to 0.0446$).
            """
        )
