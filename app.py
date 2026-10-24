"""DriftGuard AI - Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from ui import state
from ui.sidebar import render_sidebar

st.set_page_config(
    page_title="DriftGuard AI - Data Drift Monitoring",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

state.init_state()

pages = st.navigation(
    {
        "Monitor": [
            st.Page("views/overview.py", title="Overview", icon=":material/home:", default=True),
            st.Page("views/comparison.py", title="Dataset Comparison", icon=":material/compare_arrows:"),
            st.Page("views/drift_analysis.py", title="Drift Analysis", icon=":material/query_stats:"),
            st.Page("views/quality.py", title="Data Quality", icon=":material/rule:"),
        ],
        "Results": [
            st.Page("views/reports.py", title="Reports", icon=":material/description:"),
            st.Page("views/history.py", title="Analysis History", icon=":material/history:"),
        ],
    }
)

st.session_state["config"] = render_sidebar()
pages.run()
