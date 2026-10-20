"""Page 2 — Shot Explorer: Tabular and statistical exploration of 1,322 eligible field-goal attempts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.data_loader import load_merged_shot_data


def render_shot_explorer_page() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="color: #FFFFFF; font-size: 1.8rem; font-weight: 700; margin-bottom: 0.25rem;">
                🏀 Shot Explorer & Kinematic Diagnostics
            </h2>
            <p style="color: #9CA3AF; font-size: 1rem;">
                Explore out-of-fold predictions and defensive tracking measurements across 1,322 matched shot attempts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Note regarding coordinate availability per audit instructions
    st.info(
        "ℹ️ **Artifact Scope Notice:** Spatial $(x, y)$ court coordinates were not included in the processed "
        "feature CSVs to adhere to data minimization principles and prevent raw tracking redistribution. "
        "In strict compliance with CourtIQ audit standards, this explorer presents verified tabular kinematics, "
        "defensive contest measurements, and out-of-fold model predictions **without fabricating artificial court coordinates**."
    )

    # Load verified merged dataset
    try:
        df = load_merged_shot_data(".")
    except Exception as e:
        st.error(f"Error loading shot dataset: {e}")
        return

    # Sidebar / Filter Panel
    st.markdown("### 🔍 Filter Shot Cohort")

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    with col_f1:
        game_options = ["All Games (10)"] + sorted(df["game_id"].astype(str).unique().tolist())
        selected_game = st.selectbox("Game ID Filter", game_options)

    with col_f2:
        outcome_filter = st.selectbox("Observed Outcome", ["All Attempts (1,322)", "Makes Only (597)", "Misses Only (725)"])

    with col_f3:
        tracking_options = ["All Tracking Strata", "Directly Detected Only (N=1,207)", "Extrapolated Only (N=115)"]
        selected_tracking = st.selectbox("Tracking Quality", tracking_options)

    with col_f4:
        divergence_options = ["All Shots", "High Divergence (|M1 - M0| > 0.10)", "M1 Much Higher (M1 - M0 > 0.10)", "M1 Much Lower (M0 - M1 > 0.10)"]
        selected_divergence = st.selectbox("Model Divergence", divergence_options)

    # Slider filters
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        min_def, max_def = float(df["dist_release"].min()), float(df["dist_release"].max())
        def_range = st.slider(
            "Closest Defender Distance at Release (ft)",
            min_value=round(min_def, 1),
            max_value=round(max_def, 1),
            value=(round(min_def, 1), round(max_def, 1)),
            step=0.5,
        )

    with s_col2:
        min_spd, max_spd = float(df["shooter_speed_rel"].min()), float(df["shooter_speed_rel"].max())
        spd_range = st.slider(
            "Shooter Speed [t-0.4s, t] (ft/s)",
            min_value=round(min_spd, 1),
            max_value=round(max_spd, 1),
            value=(round(min_spd, 1), round(max_spd, 1)),
            step=0.5,
        )

    # Apply filters
    filtered_df = df.copy()

    if selected_game != "All Games (10)":
        filtered_df = filtered_df[filtered_df["game_id"].astype(str) == selected_game]

    if outcome_filter == "Makes Only (597)":
        filtered_df = filtered_df[filtered_df["observed_outcome"] == 1]
    elif outcome_filter == "Misses Only (725)":
        filtered_df = filtered_df[filtered_df["observed_outcome"] == 0]

    if selected_tracking == "Directly Detected Only (N=1,207)":
        filtered_df = filtered_df[filtered_df["both_detected_pre10"] == True]
    elif selected_tracking == "Extrapolated Only (N=115)":
        filtered_df = filtered_df[filtered_df["both_detected_pre10"] == False]

    if selected_divergence == "High Divergence (|M1 - M0| > 0.10)":
        filtered_df = filtered_df[filtered_df["prob_diff_m1_m0"].abs() > 0.10]
    elif selected_divergence == "M1 Much Higher (M1 - M0 > 0.10)":
        filtered_df = filtered_df[filtered_df["prob_diff_m1_m0"] > 0.10]
    elif selected_divergence == "M1 Much Lower (M0 - M1 > 0.10)":
        filtered_df = filtered_df[filtered_df["prob_diff_m1_m0"] < -0.10]

    filtered_df = filtered_df[
        (filtered_df["dist_release"] >= def_range[0])
        & (filtered_df["dist_release"] <= def_range[1])
        & (filtered_df["shooter_speed_rel"] >= spd_range[0])
        & (filtered_df["shooter_speed_rel"] <= spd_range[1])
    ]

    # Filtered Summary Metrics
    st.markdown("---")
    st.subheader("Filtered Cohort Summary")

    n_filtered = len(filtered_df)
    if n_filtered == 0:
        st.warning("No shot attempts match the selected filter combination. Please expand filter boundaries.")
        return

    n_makes = int(filtered_df["observed_outcome"].sum())
    obs_make_rate = n_makes / n_filtered
    mean_m0_prob = float(filtered_df["m0_prob"].mean())
    mean_m1_prob = float(filtered_df["m1_prob_raw"].mean())
    mean_def_dist = float(filtered_df["dist_release"].mean())
    mean_speed = float(filtered_df["shooter_speed_rel"].mean())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Filtered Attempts (N)", f"{n_filtered:,}")
    c2.metric(
        "Observed Make Rate*",
        f"{obs_make_rate*100:.1f}%",
        help="*Empirical sample frequency in this filtered cohort. This is observed history, NOT a model prediction.",
    )
    c3.metric("Mean M0 Prob", f"{mean_m0_prob:.3f}")
    c4.metric("Mean M1 Prob", f"{mean_m1_prob:.3f}", delta=f"{mean_m1_prob - mean_m0_prob:+.3f}")
    c5.metric("Mean Nearest Def Dist", f"{mean_def_dist:.1f} ft")

    st.caption("*\*Important: Observed make rate represents empirical historical conversion for these attempts. It must not be confused with a model's predicted probability.*")

    # Interactive Analytical Charts
    st.markdown("---")
    st.subheader("Kinematic & Pressure Diagnostics")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Defender Distance vs. Observed Make Rate**")
        # Bin defender distance
        dist_bins = [0, 3.0, 5.0, 8.0, 15.0, 30.0]
        dist_labels = ["Heavily Contested (<3ft)", "Contested (3-5ft)", "Light (5-8ft)", "Open (8-15ft)", "Wide Open (>15ft)"]
        
        binned_df = filtered_df.copy()
        binned_df["dist_bucket"] = pd.cut(binned_df["dist_release"], bins=dist_bins, labels=dist_labels, include_lowest=True)
        bucket_stats = binned_df.groupby("dist_bucket", observed=False).agg(
            n_shots=("observed_outcome", "count"),
            make_rate=("observed_outcome", "mean"),
            m1_pred=("m1_prob_raw", "mean"),
        ).reset_index()

        fig1, ax1 = plt.subplots(figsize=(6, 3.5), facecolor="#0E1117")
        ax1.set_facecolor("#1E293B")
        x = np.arange(len(bucket_stats))
        width = 0.35

        ax1.bar(x - width/2, bucket_stats["make_rate"], width, label="Observed Make Rate", color="#F97316", alpha=0.9)
        ax1.bar(x + width/2, bucket_stats["m1_pred"], width, label="Mean M1 Prediction", color="#38BDF8", alpha=0.9)

        ax1.set_xticks(x)
        ax1.set_xticklabels([b.split(" ")[0] for b in dist_labels], color="#E2E8F0", fontsize=8)
        ax1.set_ylabel("Probability / Rate", color="#E2E8F0", fontsize=9)
        ax1.tick_params(colors="#94A3B8")
        ax1.set_ylim(0, 1.0)
        ax1.legend(loc="upper left", facecolor="#1E293B", edgecolor="#475569", labelcolor="#F1F5F9", fontsize=8)
        ax1.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig1)

    with chart_col2:
        st.markdown("**Shooter Speed Distribution by Outcome (Makes vs. Misses)**")
        fig2, ax2 = plt.subplots(figsize=(6, 3.5), facecolor="#0E1117")
        ax2.set_facecolor("#1E293B")

        makes_spd = filtered_df[filtered_df["observed_outcome"] == 1]["shooter_speed_rel"]
        misses_spd = filtered_df[filtered_df["observed_outcome"] == 0]["shooter_speed_rel"]

        bins = np.linspace(0, 25, 20)
        if len(makes_spd) > 0:
            ax2.hist(makes_spd, bins=bins, alpha=0.6, label=f"Makes (N={len(makes_spd)})", color="#10B981", density=True)
        if len(misses_spd) > 0:
            ax2.hist(misses_spd, bins=bins, alpha=0.6, label=f"Misses (N={len(misses_spd)})", color="#EF4444", density=True)

        ax2.set_xlabel("Shooter Speed at Release (ft/s)", color="#E2E8F0", fontsize=9)
        ax2.set_ylabel("Density", color="#E2E8F0", fontsize=9)
        ax2.tick_params(colors="#94A3B8")
        ax2.legend(loc="upper right", facecolor="#1E293B", edgecolor="#475569", labelcolor="#F1F5F9", fontsize=8)
        ax2.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig2)

    # Detailed Tabular Data Explorer
    st.markdown("---")
    st.subheader(f"Detailed Shot Records ({n_filtered:,} Attempts)")
    st.markdown("Inspect individual shot predictions and kinematic tracking measurements:")

    display_cols = [
        "shot_id",
        "game_id",
        "outcome_label",
        "m0_prob",
        "m1_prob_raw",
        "speed_rel_prob_raw",
        "dist_release",
        "contest_angle_deg",
        "shooter_speed_rel",
        "tracking_status",
    ]

    rename_map = {
        "shot_id": "Shot ID",
        "game_id": "Game ID",
        "outcome_label": "Outcome",
        "m0_prob": "P(M0 Baseline)",
        "m1_prob_raw": "P(M1 Context)",
        "speed_rel_prob_raw": "P(M1 + Speed)",
        "dist_release": "Def Dist (ft)",
        "contest_angle_deg": "Contest Angle (°)",
        "shooter_speed_rel": "Shooter Speed (ft/s)",
        "tracking_status": "Tracking Mode",
    }

    styled_df = filtered_df[display_cols].rename(columns=rename_map)

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
    )

    # CSV Export Button
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Filtered Cohort ({n_filtered:,} Rows CSV)",
        data=csv_data,
        file_name=f"courtiq_filtered_shots_{n_filtered}.csv",
        mime="text/csv",
    )
