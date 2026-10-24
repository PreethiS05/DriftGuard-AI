import json

import pandas as pd
import streamlit as st

from ui import state
from ui.components import page_header

page_header(
    "Analysis History",
    "Summaries of past analyses stored in a local SQLite database. Raw data is never stored.",
)

store = state.history_store()
records = store.list(limit=200)

if not records:
    with st.container(border=True):
        st.subheader("No saved analyses")
        st.write("Run an analysis with **Save analyses to local history** enabled (Analysis settings in the sidebar).")
    st.stop()

df = pd.DataFrame(records)
df["drifted_features"] = df["drifted_features"].map(lambda f: ", ".join(f) or "-")
st.dataframe(
    df,
    hide_index=True,
    column_config={
        "analysis_id": "ID",
        "timestamp_utc": "timestamp (UTC)",
        "reference_name": "reference",
        "current_name": "current",
        "reference_rows": st.column_config.NumberColumn("ref rows", format="localized"),
        "current_rows": st.column_config.NumberColumn("cur rows", format="localized"),
        "n_features": "features",
        "n_drifted": "drifted",
        "n_insufficient": "insufficient",
        "n_quality_issues": "quality issues",
        "drifted_features": st.column_config.TextColumn("drifted features", width="large"),
    },
)

ids = df["analysis_id"].tolist()
labels = {r["analysis_id"]: f"{r['analysis_id']} - {r['timestamp_utc']} - {r['current_name']}" for r in records}

with st.container(border=True):
    st.markdown("#### Compare two analyses")
    if len(ids) < 2:
        st.write("Save at least two analyses to compare them.")
    else:
        c1, c2 = st.columns(2)
        a_id = c1.selectbox("Analysis A", ids, index=1, format_func=labels.get)
        b_id = c2.selectbox("Analysis B", ids, index=0, format_func=labels.get)
        a, b = store.get(a_id), store.get(b_id)
        if a and b:
            keys = ["n_features_analyzed", "n_drifted", "n_insufficient", "n_quality_issues", "share_drifted"]
            cmp = pd.DataFrame({"metric": keys, "A": [a["summary"].get(k) for k in keys], "B": [b["summary"].get(k) for k in keys]})
            st.dataframe(cmp, hide_index=True)
            fa = {f["feature"]: f["status"] for f in a["features"]}
            fb = {f["feature"]: f["status"] for f in b["features"]}
            changes = pd.DataFrame(
                [{"feature": f, "A": fa.get(f, "-"), "B": fb.get(f, "-")} for f in sorted(set(fa) | set(fb))]
            )
            changes["changed"] = changes["A"] != changes["B"]
            st.markdown("**Feature status by analysis**")
            st.dataframe(changes, hide_index=True)

with st.container(border=True):
    st.markdown("#### Inspect or delete")
    selected = st.selectbox("Analysis", ids, format_func=labels.get, key="hist_selected")
    c1, c2, c3 = st.columns(3)
    record = store.get(selected)
    if record:
        c1.download_button("Download JSON", json.dumps(record, indent=2), f"driftguard_{selected}_summary.json",
                           "application/json", width="stretch", icon=":material/download:", on_click="ignore")
    if c2.button("Delete this analysis", width="stretch", icon=":material/delete:"):
        store.delete(selected)
        st.rerun()
    confirm = c3.checkbox("Confirm clearing all history")
    if c3.button("Clear all history", width="stretch", disabled=not confirm, icon=":material/delete_sweep:"):
        store.clear()
        st.rerun()
    if record:
        with st.expander("Stored summary"):
            st.json(record["summary"])
