"""Page 5 — Research Audit: Methodological governance, feature specifications, tracking quality, and limitations."""

from __future__ import annotations

from pathlib import Path
import streamlit as st


def render_research_audit_page() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="color: #FFFFFF; font-size: 1.8rem; font-weight: 700; margin-bottom: 0.25rem;">
                📋 Research Audit & Methodological Governance
            </h2>
            <p style="color: #9CA3AF; font-size: 1rem;">
                Exhaustive architectural accounting of validation protocols, feature dimensionalities, and scientific boundaries.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Validation Protocol
    st.subheader("1. Leave-One-Game-Out (LOGO) Cross-Validation Architecture")
    st.markdown(
        """
        To guarantee zero data contamination across matches, CourtIQ implements a double-loop grouped cross-validation scheme:
        - **Outer Loop (10 Matches):** Exactly one game ($N \in [114, 145]$ shots) is quarantined as the outer test set. The remaining 9 games form the outer training set.
        - **Inner Loop (9 Inner LOGO Folds):** Within each outer training fold, 9 inner models generate out-of-fold probability predictions strictly for calibrator fitting.
        - **Zero Test Leakage:** Scalers (`StandardScaler`), median imputers, one-hot encoders (`OneHotEncoder(handle_unknown="ignore")`), and Platt calibrators are fit **strictly within training partitions**. Outer test sets are transformed using frozen training statistics.
        - **Quarantined Labels:** Test labels are accessed exclusively after prediction arrays are sealed to calculate held-out loss and discrimination metrics.
        """
    )

    # 2. Verified Feature Inventory
    st.markdown("---")
    st.subheader("2. Verified Feature Specification & Dimensionality Analysis")
    st.markdown(
        "Tracing executable code in `src/courtiq/evaluation/phase3a_cv.py` and `src/courtiq/features/pipeline.py`:"
    )

    f_col1, f_col2 = st.columns(2)

    with f_col1:
        st.markdown(
            """
            #### M1 Benchmark Raw Inputs (12 Variables)
            - **Continuous (5):**
              1. `distance`: Euclidean distance from shooter to attacking hoop (ft)
              2. `shotClock`: Seconds remaining on 24-second possession clock
              3. `releaseTime`: Elapsed possession seconds prior to release
              4. `dribblesBefore`: Number of dribbles executed on touch
              5. `closestDefDist`: Euclidean distance to nearest defender at release (ft)
            - **Categorical (7):**
              6. `shotType`: Floater, heave, jumper, layup, post (5 levels)
              7. `three`: Boolean flag for 3-point attempt (2 levels)
              8. `catchAndShoot`: Boolean or null/untracked (3 levels)
              9. `createdFromPaint`: Drive creation flag (2 levels)
              10. `period`: Game period 1–4, and OT (5 levels)
              11. `contestLevel`: Open, light, average, plus, blocked (5 levels)
              12. `contested`: Binary indicator for defender within 5.0 ft (2 levels)
            """
        )

    with f_col2:
        st.markdown(
            """
            #### Transformed Design Matrix ($d$)
            - **One-Hot Encoding Expansion:**
              Categoricals expand into dummy indicators:
              $5 + 2 + 3 + 2 + 5 + 5 + 2 = 24$ dummy columns.
            - **Regulation Folds (Folds 1–5, 7–10):**
              $5 \text{ continuous} + 24 \text{ OHE} = \mathbf{29 \text{ design columns}}$ ($d = 29$). With intercept: 30 parameters.
            - **Overtime Fold 6 (Game 178442):**
              Game 178442 contains the only overtime period (`period == 5`, 4 shots). When held out, the 9 training games have no OT shots, fitting 23 dummy columns ($d = 28$). At test time, `handle_unknown="ignore"` maps period 5 to all zeros without leakage.
            - **Excluded Features:**
              `shot angle`, `contest angle`, and `closing velocity` are **strictly absent** from M1 (they were tested in exploratory Phase 2 ablations).
            """
        )

    # 3. Tracking Quality Sensitivity Audit
    st.markdown("---")
    st.subheader("3. Optical Computer Vision Quality & Tracking Sensitivity")
    st.markdown(
        "SkillCorner tracking data explicitly flags whether player locations are directly detected by arena cameras or smoothed/extrapolated across occlusions:"
    )

    t_col1, t_col2 = st.columns(2)

    with t_col1:
        st.markdown(
            """
            <div style="background: #111827; border: 1px solid #10B981; padding: 1.25rem; border-radius: 8px;">
                <h4 style="color: #34D399; margin-top: 0;">Directly Detected Subset (N = 1,207 | 91.30%)</h4>
                <p style="color: #E2E8F0; font-size: 0.95rem;">
                    Shots where the shooter position was directly detected by optical cameras at release frame ($t$):
                </p>
                <ul style="color: #94A3B8; font-size: 0.9rem;">
                    <li>M1 Benchmark Log Loss: <b>0.617401</b></li>
                    <li>M1 + Speed Rel Log Loss: <b>0.616768</b></li>
                    <li>Log Loss Delta: <b style="color: #4ADE80;">-0.000633</b></li>
                    <li>Conclusion: Exploratory speed signal is genuine to optical tracking and not an artifact of extrapolation.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t_col2:
        st.markdown(
            """
            <div style="background: #111827; border: 1px solid #F59E0B; padding: 1.25rem; border-radius: 8px;">
                <h4 style="color: #FBBF24; margin-top: 0;">Extrapolated Subset (N = 115 | 8.70%)</h4>
                <p style="color: #E2E8F0; font-size: 0.95rem;">
                    Shots where the shooter position was interpolated across optical occlusions or arena boundary constraints:
                </p>
                <ul style="color: #94A3B8; font-size: 0.9rem;">
                    <li>M1 Benchmark Log Loss: <b>0.615891</b></li>
                    <li>M1 + Speed Rel Log Loss: <b>0.613823</b></li>
                    <li>Log Loss Delta: <b style="color: #4ADE80;">-0.002068</b> (3× larger)</li>
                    <li>Domain Finding: High-tempo fast-break sprints and corner boundary scrambles elevate velocity dynamics where occlusion occurs.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Critical Scientific Boundaries
    st.markdown("---")
    st.subheader("4. Mandatory Epistemological & Scientific Boundaries")
    st.markdown(
        """
        1. **Observational Prediction vs. Counterfactual Decision:**
           CourtIQ predicts field-goal conversion conditional on a shot release occurring ($P(\text{Made} = 1 \mid X_t)$). It is **not** an audit of decision quality (e.g. whether a player should have passed to a cutting teammate or driven to the rim). A shot with $p = 0.35$ may still be the optimal tactical decision late in a shot clock.
        2. **Planar 2D Coordinate Limitation:**
           Tracking data is planar $(x, y)$. Release height, shooter vertical jump apex, arm extension angle, and shot trajectory arc are unmeasured.
        3. **10-Game Cluster Regimes:**
           Evaluated strictly across 10 Spanish Liga ACB matches. Findings cannot be generalized to the NBA, EuroLeague, or NCAA without multi-venue validation.
        4. **Zero Unjustified Significance Claims:**
           The paired cluster bootstrap 95% confidence interval for shooter speed (`[-0.002177, +0.000854]`) crosses zero. The hypothesis remains exploratory.
        """
    )

    # 5. Access to Full Audited Reports
    st.markdown("---")
    st.subheader("5. Audited Documentation Deliverables")
    st.markdown("All research phases were systematically documented and audited in the repository:")

    doc_files = [
        ("reports/phase3b/final_research_report.md", "Phase 3B Final Research Report (Master Scientific Deliverable)"),
        ("reports/phase3a/research_report.md", "Phase 3A Velocity Experiment & Nested Calibration Report"),
        ("reports/phase3a/calibration_comparison.md", "Phase 3A Calibration Parameter Stability & Provenance Audit"),
        ("reports/phase3a/tracking_quality_sensitivity.md", "Phase 3A Tracking Quality & Lookback Sensitivity Report"),
        ("reports/phase3b/independent_methodological_audit.md", "Phase 3B Independent External Methodological Audit"),
        ("reports/phase3c/final_audit_findings.md", "Phase 3C Forensic Code & Statistical Audit Log"),
        ("reports/phase3d/documentation_correction_log.md", "Phase 3D Documentation Correction & Dimensionality Log"),
        ("reports/phase3e/final_consistency_audit.md", "Phase 3E Repository Consistency Audit"),
    ]

    for rel_path, title in doc_files:
        p = Path(rel_path)
        if p.exists():
            with st.expander(f"📄 {title} (`{rel_path}`)"):
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read()
                st.markdown(content[:3000] + ("\n\n*(Content truncated for preview. Full report available in repository)*" if len(content) > 3000 else ""))
                st.download_button(
                    label=f"Download {p.name}",
                    data=content.encode("utf-8"),
                    file_name=p.name,
                    mime="text/markdown",
                    key=f"dl_{rel_path}",
                )
