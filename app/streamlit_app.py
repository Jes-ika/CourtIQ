"""CourtIQ Research Dashboard — Judge-Facing Presentation & Exploration Interface.

Streamlit entry point for hackathon evaluation and interactive research exploration.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root directory is on Python path for imports
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import streamlit as st

from app.views.calibration_view import render_calibration_page
from app.views.model_lab import render_model_lab_page
from app.views.overview import render_overview_page
from app.views.research_audit import render_research_audit_page
from app.views.shot_explorer import render_shot_explorer_page

# Set page configuration
st.set_page_config(
    page_title="CourtIQ — Pressure-Aware Shot Quality Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Sports Analytics CSS (Dark Navy / Charcoal with Orange Accents)
st.markdown(
    """
    <style>
        /* Base page and font settings */
        .stApp {
            background-color: #0B0F19;
            color: #F3F4F6;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #1F2937;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background-color: #1F2937;
            border: 1px solid #374151;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }

        div[data-testid="stMetricLabel"] {
            color: #9CA3AF !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
        }

        div[data-testid="stMetricValue"] {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }

        /* Dataframe styling */
        div[data-testid="stDataFrame"] {
            border: 1px solid #374151;
            border-radius: 8px;
            background-color: #111827;
        }

        /* Radio navigation buttons */
        div[data-testid="stRadio"] > label {
            color: #F97316 !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            margin-bottom: 0.5rem;
        }

        /* Expander headers */
        div[data-testid="stExpander"] {
            background-color: #111827;
            border: 1px solid #374151;
            border-radius: 8px;
            margin-bottom: 0.5rem;
        }

        /* Tooltip and help text */
        .stTooltipIcon {
            color: #F97316 !important;
        }

        /* Primary buttons */
        div.stButton > button:first-child {
            background-color: #EA580C;
            color: #FFFFFF;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease-in-out;
        }

        div.stButton > button:first-child:hover {
            background-color: #C2410C;
            color: #FFFFFF;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    # Sidebar Header & Navigation
    st.sidebar.markdown(
        """
        <div style="padding: 0.5rem 0 1rem 0; text-align: center;">
            <div style="font-size: 2.2rem; margin-bottom: 0.25rem;">🏀</div>
            <h2 style="color: #FFFFFF; margin: 0; font-size: 1.5rem; font-weight: 700; letter-spacing: -0.5px;">
                Court<span style="color: #F97316;">IQ</span>
            </h2>
            <p style="color: #9CA3AF; font-size: 0.8rem; margin-top: 0.25rem;">
                Research & Evaluation Dashboard
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        options=[
            "1. Executive Overview",
            "2. Shot Explorer & Kinematics",
            "3. Model Lab & Evaluation",
            "4. Calibration & Reliability",
            "5. Research Audit & Methodology",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-size: 0.8rem; color: #9CA3AF; line-height: 1.6;">
            <div style="color: #F3F4F6; font-weight: 600; margin-bottom: 0.4rem;">Audited Benchmark Status:</div>
            <div>• <b>Benchmark:</b> M1 (Frozen)</div>
            <div>• <b>Data:</b> 10 ACB Matches (25 Hz)</div>
            <div>• <b>Sample:</b> 1,322 Matched FGA</div>
            <div>• <b>Validation:</b> 10-Fold LOGO CV</div>
            <div>• <b>Calibration:</b> Nested Platt Scaling</div>
            <div>• <b>Inference:</b> Cluster Bootstrap</div>
            <div>• <b>Tests:</b> 49 / 49 Passing</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "CourtIQ Hackathon Presentation Interface | © 2026 SkillCorner Open Data Research"
    )

    # Route to selected page
    if page == "1. Executive Overview":
        render_overview_page()
    elif page == "2. Shot Explorer & Kinematics":
        render_shot_explorer_page()
    elif page == "3. Model Lab & Evaluation":
        render_model_lab_page()
    elif page == "4. Calibration & Reliability":
        render_calibration_page()
    elif page == "5. Research Audit & Methodology":
        render_research_audit_page()


if __name__ == "__main__":
    main()
