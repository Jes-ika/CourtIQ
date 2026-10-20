"""Page 3 — Model Lab: Comprehensive evaluation across M0, M1, Speed Rel, and Directional models."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.data_loader import AUTHORITATIVE_METRICS, load_merged_shot_data, load_per_game_metrics


def render_model_lab_page() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="color: #FFFFFF; font-size: 1.8rem; font-weight: 700; margin-bottom: 0.25rem;">
                🔬 Model Lab & Statistical Evaluation
            </h2>
            <p style="color: #9CA3AF; font-size: 1rem;">
                Authoritative Leave-One-Game-Out cross-validation benchmarks and paired uncertainty quantification.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Core Benchmark Comparison Table
    st.subheader("1. Master Model Comparison (Matched N=1,322 Shots, 10 LOGO Folds)")
    st.markdown(
        "All candidates were evaluated across identical shot rows and identical Leave-One-Game-Out outer folds. "
        "**Lower values are strictly superior** for Log Loss and Brier Score; **higher values are superior** for ROC-AUC."
    )

    models_data = [
        {
            "Model Specification": "M0 Baseline (Spatial/Mechanics)",
            "Raw": 9,
            "Design ($d$)*": 21,
            "Outer Log Loss": f"{AUTHORITATIVE_METRICS['M0_baseline']['log_loss']:.6f}",
            "Δ vs M1": f"+{AUTHORITATIVE_METRICS['M0_baseline']['delta_vs_m1']:.6f}",
            "95% Cluster Bootstrap CI": "—",
            "Brier Score": f"{AUTHORITATIVE_METRICS['M0_baseline']['brier_score']:.6f}",
            "ROC-AUC": f"{AUTHORITATIVE_METRICS['M0_baseline']['roc_auc']:.6f}",
            "10-bin ECE": f"{AUTHORITATIVE_METRICS['M0_baseline']['ece_10']:.6f}",
            "Win Rate (vs M1)": "—",
            "Scientific Status": "Deprecated Baseline",
        },
        {
            "Model Specification": "M1 Benchmark (Frozen Official)",
            "Raw": 12,
            "Design ($d$)*": 29,
            "Outer Log Loss": f"{AUTHORITATIVE_METRICS['M1_benchmark']['log_loss']:.6f}",
            "Δ vs M1": "—",
            "95% Cluster Bootstrap CI": "—",
            "Brier Score": f"{AUTHORITATIVE_METRICS['M1_benchmark']['brier_score']:.6f}",
            "ROC-AUC": f"{AUTHORITATIVE_METRICS['M1_benchmark']['roc_auc']:.6f}",
            "10-bin ECE": f"{AUTHORITATIVE_METRICS['M1_benchmark']['ece_10']:.6f}",
            "Win Rate (vs M1)": "—",
            "Scientific Status": "Official Benchmark",
        },
        {
            "Model Specification": "M1 + Speed Rel [t-0.4s, t]",
            "Raw": 13,
            "Design ($d$)*": 30,
            "Outer Log Loss": f"{AUTHORITATIVE_METRICS['M1_plus_speed_rel']['log_loss']:.6f}",
            "Δ vs M1": f"{AUTHORITATIVE_METRICS['M1_plus_speed_rel']['delta_vs_m1']:.6f}",
            "95% Cluster Bootstrap CI": "[-0.002177, +0.000854]",
            "Brier Score": f"{AUTHORITATIVE_METRICS['M1_plus_speed_rel']['brier_score']:.6f}",
            "ROC-AUC": f"{AUTHORITATIVE_METRICS['M1_plus_speed_rel']['roc_auc']:.6f}",
            "10-bin ECE": f"{AUTHORITATIVE_METRICS['M1_plus_speed_rel']['ece_10']:.6f}",
            "Win Rate (vs M1)": "7 / 10 Games",
            "Scientific Status": "Exploratory (Unconfirmed)",
        },
        {
            "Model Specification": "M1 + Speed Early [t-0.8s, t-0.4s]",
            "Raw": 13,
            "Design ($d$)*": 30,
            "Outer Log Loss": f"{AUTHORITATIVE_METRICS['M1_plus_speed_early']['log_loss']:.6f}",
            "Δ vs M1": f"{AUTHORITATIVE_METRICS['M1_plus_speed_early']['delta_vs_m1']:.6f}",
            "95% Cluster Bootstrap CI": "[-0.001136, +0.000901]",
            "Brier Score": f"{AUTHORITATIVE_METRICS['M1_plus_speed_early']['brier_score']:.6f}",
            "ROC-AUC": f"{AUTHORITATIVE_METRICS['M1_plus_speed_early']['roc_auc']:.6f}",
            "10-bin ECE": f"{AUTHORITATIVE_METRICS['M1_plus_speed_early']['ece_10']:.6f}",
            "Win Rate (vs M1)": "5 / 10 Games",
            "Scientific Status": "Signal Decayed",
        },
        {
            "Model Specification": "M1 + Directional (v_rad + v_tan)",
            "Raw": 14,
            "Design ($d$)*": 31,
            "Outer Log Loss": f"{AUTHORITATIVE_METRICS['M1_plus_directional']['log_loss']:.6f}",
            "Δ vs M1": f"+{AUTHORITATIVE_METRICS['M1_plus_directional']['delta_vs_m1']:.6f}",
            "95% Cluster Bootstrap CI": "[+0.000211, +0.002238]",
            "Brier Score": f"{AUTHORITATIVE_METRICS['M1_plus_directional']['brier_score']:.6f}",
            "ROC-AUC": f"{AUTHORITATIVE_METRICS['M1_plus_directional']['roc_auc']:.6f}",
            "10-bin ECE": f"{AUTHORITATIVE_METRICS['M1_plus_directional']['ece_10']:.6f}",
            "Win Rate (vs M1)": "1 / 10 Games",
            "Scientific Status": "Statistically Worse",
        },
    ]

    st.dataframe(pd.DataFrame(models_data), use_container_width=True, hide_index=True)

    st.caption(
        "*\*Design column counts reflect 9 regular-season regulation training folds. In Fold 6 (held-out game 178442), "
        "the training fold excludes overtime period 5, resulting in $d - 1$ columns ($d=28$ for M1; 29 parameters with intercept).*"
    )

    # Detailed Comparison Cards: M0 vs M1 vs Speed Rel vs Directional
    st.markdown("---")
    st.subheader("2. Rigorous Statistical Audit of Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div style="background: #111827; border: 1px solid #1E3A8A; padding: 1.25rem; border-radius: 8px;">
                <h4 style="color: #60A5FA; margin-top: 0;">M1 Context Benchmark: Confirmed Superiority</h4>
                <p style="color: #E2E8F0; font-size: 0.95rem; line-height: 1.5;">
                    Adding defensive proximity and contest level improves held-out log loss from 
                    <b>0.655161 to 0.617270</b> (<span style="color: #4ADE80;">-0.037891 reduction</span>) and ROC-AUC from 
                    <b>0.641026 to 0.694100</b> (<span style="color: #4ADE80;">+5.31 points</span>).
                </p>
                <div style="background: #1E293B; padding: 0.75rem; border-radius: 6px; font-size: 0.85rem; color: #94A3B8;">
                    <b>Why M1 has a higher 10-bin ECE (0.051075 vs 0.030552):</b><br>
                    M0 shrinks predictions tightly around the marginal base rate (0.45), having minimal tail sharpness. M1 correctly assigns high-confidence predictions (&lt;0.20 and &gt;0.80), where finite binomial sampling variance naturally inflates ECE.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="background: #111827; border: 1px solid #C2410C; padding: 1.25rem; border-radius: 8px;">
                <h4 style="color: #FB923C; margin-top: 0;">Shooter Speed: Exploratory, NOT Confirmed</h4>
                <p style="color: #E2E8F0; font-size: 0.95rem; line-height: 1.5;">
                    Release-lookback shooter speed yields a modest log-loss delta of <b>-0.000758</b>. However, 
                    paired game-cluster bootstrap resampling (B=2,000) yields a 95% CI of 
                    <b>[-0.002177, +0.000854]</b>, which spans zero.
                </p>
                <div style="background: #1E293B; padding: 0.75rem; border-radius: 6px; font-size: 0.85rem; color: #94A3B8;">
                    <b>Benchmark Governance Decision:</b><br>
                    Because the speed improvement fails to meet the pre-specified practical threshold of <b>0.0020</b> and is not statistically significant on 10 match clusters, M1 remains the official frozen benchmark.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Per-Game Fold Breakdown
    st.markdown("---")
    st.subheader("3. Per-Game Fold Breakdown across 10 LOGO Matches")
    st.markdown(
        "Inspect held-out game metrics derived deterministically from [`reports/phase1_5_per_game_metrics.csv`](reports/phase1_5_per_game_metrics.csv):"
    )

    try:
        df_games = load_per_game_metrics(".")
        
        # Display chart of per-game delta
        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0E1117")
        ax.set_facecolor("#1E293B")

        x = np.arange(len(df_games))
        deltas = df_games["delta_log_loss"].values
        colors = ["#10B981" if d < 0 else "#EF4444" for d in deltas]

        ax.bar(x, deltas, color=colors, alpha=0.9, width=0.55)
        ax.axhline(0, color="#94A3B8", linestyle="--", linewidth=1.0)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Game {gid}\n(N={n})" for gid, n in zip(df_games["game_id"], df_games["shots"])], color="#E2E8F0", fontsize=8)
        ax.set_ylabel("Log Loss Delta (M1 - M0)", color="#E2E8F0", fontsize=9)
        ax.set_title("Held-Out Log Loss Improvement by Game (Negative = M1 Superior)", color="#FFFFFF", fontsize=11, fontweight="bold")
        ax.tick_params(colors="#94A3B8")
        ax.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

        st.caption(
            "Notice Game 114086 is the only fold where M1 degraded (+0.0070). "
            "An audited forensic analysis revealed that in Game 114086, shooting teams converted heavily contested shots at over 50% "
            "(vs. the league average 41%), penalizing M1 for recognizing defensive pressure."
        )

        # Tabular display
        game_table = df_games[["fold", "game_id", "shots", "m0_log_loss", "m1_log_loss", "delta_log_loss", "m0_brier", "m1_brier", "delta_brier", "m0_auc", "m1_auc", "delta_auc"]].copy()
        game_table.columns = ["Fold", "Game ID", "Shots", "M0 LogLoss", "M1 LogLoss", "Δ LogLoss", "M0 Brier", "M1 Brier", "Δ Brier", "M0 AUC", "M1 AUC", "Δ AUC"]
        st.dataframe(game_table, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error rendering per-game breakdown: {e}")
