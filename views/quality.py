import pandas as pd
import streamlit as st

from driftguard.quality import CRITICAL, INFO, WARNING
from driftguard.visualization import missing_comparison
from ui.components import dataset_context, page_header, require_result

page_header(
    "Data Quality",
    "Integrity checks reported separately from distribution drift. Quality issues never change a drift status.",
)

result = require_result()
if result is None:
    st.stop()

dataset_context(result)
issues = result.quality_issues
counts = result.summary()["quality_counts"]

m = st.columns(4)
m[0].metric("Critical", counts[CRITICAL], border=True)
m[1].metric("Warnings", counts[WARNING], border=True)
m[2].metric("Info", counts[INFO], border=True)
m[3].metric("Duplicate rows (cur)", f"{result.current_profile.n_duplicate_rows:,}", border=True,
            help=f"Reference: {result.reference_profile.n_duplicate_rows:,}")

with st.container(border=True):
    st.markdown("#### Issues")
    if not issues:
        st.success("No data quality issues found.", icon=":material/check_circle:")
    else:
        severities = st.segmented_control(
            "Severity", [CRITICAL, WARNING, INFO], selection_mode="multi",
            default=[CRITICAL, WARNING, INFO], format_func=str.title,
        )
        df = pd.DataFrame([i.to_dict() for i in issues])
        df = df[df["severity"].isin(severities or [])]
        st.dataframe(
            df[["severity", "check", "column", "dataset", "message"]],
            hide_index=True,
            column_config={"message": st.column_config.TextColumn("details", width="large")},
        )

common = result.schema.common_columns
missing_cols = [
    c for c in common
    if result.reference_profile.columns[c].pct_missing > 0 or result.current_profile.columns[c].pct_missing > 0
]
with st.container(border=True):
    if missing_cols:
        st.plotly_chart(missing_comparison(result.reference_profile, result.current_profile, missing_cols), width="stretch")
        st.caption("Only columns with at least one missing value are shown. Missing values are excluded from drift tests.")
    else:
        st.markdown("#### Missing values")
        st.write("No missing values in any shared column.")

if result.excluded_features:
    with st.container(border=True):
        st.markdown("#### Columns excluded from drift testing")
        for col, reason in result.excluded_features.items():
            st.markdown(f"- `{col}`: {reason}")
