import pandas as pd
import streamlit as st

from driftguard.methodology import DRIFT_VS_DEGRADATION, LIMITATIONS, METHODOLOGY, threshold_rows
from driftguard.reports import html_report, results_csv, summary_json
from ui.components import dataset_context, page_header, require_result

page_header("Reports", "Export results and review the methodology, thresholds and limitations behind them.")

result = require_result()
if result is None:
    st.stop()

dataset_context(result)
base = f"driftguard_{result.analysis_id}"

with st.container(border=True):
    st.markdown("#### Download")
    st.caption("Exports contain aggregate statistics only - no raw rows.")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Feature results (CSV)", results_csv(result), f"{base}_features.csv", "text/csv",
                       icon=":material/table_view:", width="stretch", on_click="ignore")
    c2.download_button("Full summary (JSON)", summary_json(result), f"{base}_summary.json", "application/json",
                       icon=":material/data_object:", width="stretch", on_click="ignore")
    c3.download_button("Report (HTML)", html_report(result), f"{base}_report.html", "text/html",
                       icon=":material/description:", width="stretch", on_click="ignore", type="primary")

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown("#### Analysis metadata")
        meta = pd.DataFrame(
            [
                ("Analysis ID", result.analysis_id),
                ("Timestamp (UTC)", result.timestamp),
                ("Reference", f"{result.reference_meta.name} ({result.reference_meta.n_rows:,} rows)"),
                ("Current", f"{result.current_meta.name} ({result.current_meta.n_rows:,} rows)"),
                ("Synthetic data", "Yes" if result.reference_meta.synthetic or result.current_meta.synthetic else "No"),
            ],
            columns=["field", "value"],
        )
        st.dataframe(meta, hide_index=True)
    with st.container(border=True):
        st.markdown("#### Thresholds used")
        st.dataframe(pd.DataFrame(threshold_rows(result.config), columns=["setting", "value"]), hide_index=True)

with right:
    with st.container(border=True):
        st.markdown("#### Methodology")
        for title, text in METHODOLOGY:
            st.markdown(f"**{title}.** {text}")

with st.container(border=True):
    st.markdown("#### Data drift is not model performance degradation")
    st.dataframe(pd.DataFrame(DRIFT_VS_DEGRADATION, columns=["concept", "meaning"]), hide_index=True)

with st.container(border=True):
    st.markdown("#### Limitations")
    for item in LIMITATIONS:
        st.markdown(f"- {item}")
