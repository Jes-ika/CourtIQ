"""Page 1 — Overview: Executive research briefing, dataset funnel, and core benchmarks."""

from __future__ import annotations

import streamlit as st
from app.data_loader import AUTHORITATIVE_METRICS, FUNNEL_STATS


def render_overview_page() -> None:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #111827 0%, #1F2937 100%); padding: 1.5rem 2rem; border-radius: 12px; border-left: 6px solid #F97316; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
            <h1 style="color: #FFFFFF; margin: 0; font-size: 2.2rem; font-weight: 700; letter-spacing: -0.5px;">
                CourtIQ: Pressure-Aware Shot Quality
            </h1>
            <p style="color: #9CA3AF; margin-top: 0.5rem; font-size: 1.05rem; line-height: 1.5;">
                Audited Sports Analytics Benchmark for 25 Hz Optical Tracking Data (Spanish Liga ACB, 2025–26)
            </p>
            <div style="display: flex; gap: 0.75rem; margin-top: 1rem; flex-wrap: wrap;">
                <span style="background: #374151; color: #F3F4F6; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.85rem; font-weight: 600;">10 Matches</span>
                <span style="background: #374151; color: #F3F4F6; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.85rem; font-weight: 600;">1,322 Eligible FGA</span>
                <span style="background: #1E3A8A; color: #93C5FD; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.85rem; font-weight: 600;">Leave-One-Game-Out CV</span>
                <span style="background: #C2410C; color: #FFEDD5; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.85rem; font-weight: 600;">Frozen M1 Benchmark</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Plain English Research Question
    st.subheader("1. The Research Question")
    st.markdown(
        """
        > **"Does adding defensive-contest features and shooter kinematics measured at shot release improve probabilistic held-out-game prediction of basketball field-goal outcomes beyond a baseline using shot location and basic mechanics?"**

        In modern basketball, shot location alone cannot fully describe the true difficulty of an attempt. A 22-foot corner jumper taken with a defender draped over the shooter involves vastly different physics and make probability than an uncontested catch-and-shoot look from the exact same spatial coordinates. 
        
        CourtIQ investigates whether high-frequency (25 Hz) tracking data capturing **nearest defender distance**, **contest categorization**, and **instantaneous shooter speed** at the release frame provides statistically generalizable improvements on unseen games.
        """
    )

    # Audited Dataset Funnel
    st.markdown("---")
    st.subheader("2. Audited Dataset Accounting Funnel")
    st.markdown(
        "To strictly conform with FIBA official box-score scoring conventions, raw SkillCorner event logs undergo a four-stage sequential filtering pipeline:"
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Raw Events Ingested",
            value=f"{FUNNEL_STATS['total_raw_events']:,}",
            help="Total shot events recorded across all 10 Liga ACB match directories",
        )
    with col2:
        st.metric(
            label="Unlocated Excluded",
            value=f"-{FUNNEL_STATS['unlocated_excluded']}",
            delta="-1.78%",
            delta_color="inverse",
            help="Shots missing tracking coordinates or hoop distance (26 records, 0 were fouled)",
        )
    with col3:
        st.metric(
            label="Fouled Misses Excluded",
            value=f"-{FUNNEL_STATS['fouled_misses_excluded']}",
            delta="-7.61%",
            delta_color="inverse",
            help="Fouled misses awarded free throws; FIBA box-score rules treat these as non-FGAs (111 records)",
        )
    with col4:
        st.metric(
            label="Retained Eligible FGA",
            value=f"{FUNNEL_STATS['eligible_shots_retained']:,}",
            delta="100.0% Matched",
            help="Clean live-play field-goal attempts evaluated row-for-row across all models",
        )

    # Class balance breakdown
    st.info(
        f"**Retained Cohort Composition ($N = 1,322$):** Exactly **{FUNNEL_STATS['observed_makes']} Makes** "
        f"({FUNNEL_STATS['observed_make_rate']*100:.2f}%) and **{FUNNEL_STATS['observed_misses']} Misses** (54.84%). "
        f"Exactly **{FUNNEL_STATS['fouled_makes_retained']} fouled makes (and-ones)** were retained in the dataset as legitimate box-score field goals. "
        f"All 111 excluded fouled misses have 100% complete linkages to official foul event IDs (`foulId`)."
    )

    # Authoritative M0 vs M1 Benchmark
    st.markdown("---")
    st.subheader("3. Primary Head-to-Head Benchmark: M0 vs. M1")
    st.markdown(
        "All models are evaluated via **Leave-One-Game-Out cross-validation across 10 outer match clusters**, "
        "ensuring that training partitions share zero match, venue, opponent, or tactical overlap with the held-out test game."
    )

    m0 = AUTHORITATIVE_METRICS["M0_baseline"]
    m1 = AUTHORITATIVE_METRICS["M1_benchmark"]

    b_col1, b_col2, b_col3, b_col4 = st.columns(4)

    with b_col1:
        st.markdown(
            f"""
            <div style="background: #1E293B; padding: 1.25rem; border-radius: 8px; border-top: 4px solid #38BDF8;">
                <div style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase; font-weight: 600;">Out-of-Fold Log Loss</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #FFFFFF; margin: 0.4rem 0;">{m1['log_loss']:.6f}</div>
                <div style="color: #4ADE80; font-size: 0.9rem; font-weight: 600;">-0.037891 vs M0 ({m0['log_loss']:.6f})</div>
                <div style="color: #64748B; font-size: 0.75rem; margin-top: 0.25rem;">Lower is better (Cross-entropy)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b_col2:
        st.markdown(
            f"""
            <div style="background: #1E293B; padding: 1.25rem; border-radius: 8px; border-top: 4px solid #10B981;">
                <div style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase; font-weight: 600;">Brier Score</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #FFFFFF; margin: 0.4rem 0;">{m1['brier_score']:.6f}</div>
                <div style="color: #4ADE80; font-size: 0.9rem; font-weight: 600;">-0.015514 vs M0 ({m0['brier_score']:.6f})</div>
                <div style="color: #64748B; font-size: 0.75rem; margin-top: 0.25rem;">Lower is better (Mean squared error)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b_col3:
        st.markdown(
            f"""
            <div style="background: #1E293B; padding: 1.25rem; border-radius: 8px; border-top: 4px solid #F59E0B;">
                <div style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase; font-weight: 600;">ROC-AUC Discrimination</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #FFFFFF; margin: 0.4rem 0;">{m1['roc_auc']:.6f}</div>
                <div style="color: #4ADE80; font-size: 0.9rem; font-weight: 600;">+0.053074 vs M0 ({m0['roc_auc']:.6f})</div>
                <div style="color: #64748B; font-size: 0.75rem; margin-top: 0.25rem;">Higher is better (Rank-ordering)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b_col4:
        st.markdown(
            f"""
            <div style="background: #1E293B; padding: 1.25rem; border-radius: 8px; border-top: 4px solid #EF4444;">
                <div style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase; font-weight: 600;">10-Bin ECE (Calibration)</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #FFFFFF; margin: 0.4rem 0;">{m1['ece_10']:.6f}</div>
                <div style="color: #F87171; font-size: 0.9rem; font-weight: 600;">+0.020523 vs M0 ({m0['ece_10']:.6f})</div>
                <div style="color: #64748B; font-size: 0.75rem; margin-top: 0.25rem;">M1 has higher ECE (Sharpness tradeoff)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Concise Scientific Findings & Critical Limitations
    st.markdown("---")
    st.subheader("4. Core Takeaways & Epistemological Boundaries")

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown(
            """
            ### What the Empirical Evidence Establishes
            1. **Defensive Context Adds Large, Defensible Value:**
               Adding nearest defender distance and contest categorization (M1) reduces outer-fold log loss by **-0.037891** (paired t-test $p < 0.001$), improving held-out accuracy and discrimination across 9 of 10 matches.
            2. **Shooter Kinematics is Exploratory Only:**
               Release-lookback shooter speed (`M1 + speed_rel`) produces a minor exploratory log-loss reduction ($\Delta = -0.000758$). However, its paired game-cluster bootstrap 95% confidence interval is `[-0.002177, +0.000854]` (spans zero) and it fails the pre-specified practical target of $0.0020$.
            3. **Directional Velocity Degrades Performance:**
               Decomposing velocity into radial and tangential components degrades log loss ($\Delta = +0.001080$, 95% CI $[+0.000211, +0.002238]$ strictly worse), confirming that adding directional parameters on a 10-game sample introduces harmful estimator variance.
            """
        )

    with right_col:
        st.markdown(
            """
            ### Mandatory Scientific & Methodological Limitations
            1. **Release-Time Outcome, NOT a Decision Audit:**
               Predictions are generated strictly at shot release (`startFrame`). This system predicts outcome probability **given that a shot was taken**. It is **not** a counterfactual shot-selection model and cannot determine whether a player *should* have passed or driven.
            2. **Small-Cluster Regime ($K = 10$ Games):**
               The benchmark evaluates 10 matches from the Spanish Liga ACB (2025–26). Statistical power in a 10-cluster design is limited; results do not establish external validity across the NBA, EuroLeague, or NCAA.
            3. **ECE Does Not Mean M1 is Miscalibrated:**
               M0 appears superficially better in ECE because its predictions are compressed tightly around the base rate ($0.45$). M1 pushes predictions into the high-confidence tails ($<0.20$ and $>0.80$), where binomial sampling variance in smaller bins inflates ECE despite superior log loss and Brier score.
            """
        )
