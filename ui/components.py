"""Reusable presentation helpers."""

from __future__ import annotations

import streamlit as st

from driftguard.drift import DRIFT, INSUFFICIENT, NO_DRIFT
from driftguard.pipeline import AnalysisResult
from driftguard.quality import CRITICAL, INFO, WARNING
from driftguard.reports import format_p
from driftguard.schema import NUMERICAL

from . import state

STATUS_BADGE = {DRIFT: ":red-badge[Drift Detected]", NO_DRIFT: ":green-badge[No Significant Drift]", INSUFFICIENT: ":gray-badge[Insufficient Data]"}
SEVERITY_BADGE = {CRITICAL: ":red-badge[Critical]", WARNING: ":orange-badge[Warning]", INFO: ":blue-badge[Info]"}


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def dataset_context(result: AnalysisResult) -> None:
    ref, cur = result.reference_meta, result.current_meta
    synthetic = " :violet-badge[Synthetic data]" if ref.synthetic or cur.synthetic else ""
    st.markdown(
        f"**Reference:** `{ref.name}` ({ref.n_rows:,} x {ref.n_columns}) &nbsp;·&nbsp; "
        f"**Current:** `{cur.name}` ({cur.n_rows:,} x {cur.n_columns}) &nbsp;·&nbsp; "
        f"Analysis `{result.analysis_id}` at {result.timestamp.replace('T', ' ')} UTC{synthetic}"
    )


def require_result() -> AnalysisResult | None:
    result = state.get_result()
    if result is not None:
        return result
    with st.container(border=True):
        st.subheader("No analysis yet")
        if state.has_datasets():
            st.write("Datasets are loaded. Click **Run analysis** in the sidebar to compute results.")
        else:
            st.write(
                "Load a reference and a current dataset from the sidebar, or open **Overview** "
                "and run the one-click demo."
            )
        st.page_link("views/overview.py", label="Go to Overview", icon=":material/home:")
    return None


def p_text(p: float | None) -> str:
    return format_p(p)


def stat_label(feature_type: str) -> str:
    return "KS D" if feature_type == NUMERICAL else "Cramer's V"
