import pandas as pd
import streamlit as st

from driftguard.schema import NUMERICAL
from ui import state
from ui.components import dataset_context, page_header, require_result

page_header("Dataset Comparison", "Shape, schema and per-column profiles of the reference and current datasets.")

result = require_result()
if result is None:
    st.stop()

dataset_context(result)
rp, cp = result.reference_profile, result.current_profile


def profile_card(title: str, meta, prof) -> None:
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.caption(f"`{meta.name}` - source: {meta.source}{' (synthetic)' if meta.synthetic else ''}")
        a, b, c, d = st.columns(4)
        a.metric("Rows", f"{prof.n_rows:,}")
        b.metric("Columns", prof.n_columns)
        c.metric("Missing cells", f"{prof.pct_missing_cells:.1%}")
        d.metric("Duplicate rows", f"{prof.n_duplicate_rows:,}")
        st.caption(f"In-memory size: {prof.memory_mb:.2f} MB")


c1, c2 = st.columns(2)
with c1:
    profile_card("Reference", result.reference_meta, rp)
with c2:
    profile_card("Current", result.current_meta, cp)

s = result.schema
with st.container(border=True):
    st.markdown("#### Schema")
    a, b, c, d = st.columns(4)
    a.metric("Common columns", len(s.common_columns))
    b.metric("Missing in current", len(s.missing_in_current))
    c.metric("New in current", len(s.new_in_current))
    d.metric("Type mismatches", len(s.type_mismatches))
    if s.missing_in_current:
        st.warning("Missing from current: " + ", ".join(f"`{c}`" for c in s.missing_in_current))
    if s.new_in_current:
        st.info("New in current: " + ", ".join(f"`{c}`" for c in s.new_in_current))
    if s.type_mismatches:
        st.error("Type mismatches: " + ", ".join(
            f"`{c}` ({s.reference_types[c]} -> {s.current_types[c]})" for c in s.type_mismatches))

def center(p) -> str | None:
    if p is None:
        return None
    if p.feature_type == NUMERICAL:
        mean = p.stats.get("mean")
        return None if mean is None else f"{mean:,.4g}"
    return p.stats.get("mode")


all_cols = list(dict.fromkeys(list(rp.columns) + list(cp.columns)))
rows = []
for col in all_cols:
    r, c = rp.columns.get(col), cp.columns.get(col)
    rows.append({
        "column": col,
        "type (ref)": r.feature_type if r else "-",
        "type (cur)": c.feature_type if c else "-",
        "dtype (ref)": r.dtype if r else "-",
        "dtype (cur)": c.dtype if c else "-",
        "missing % (ref)": r.pct_missing * 100 if r else None,
        "missing % (cur)": c.pct_missing * 100 if c else None,
        "unique (ref)": r.n_unique if r else None,
        "unique (cur)": c.n_unique if c else None,
        "mean / mode (ref)": center(r),
        "mean / mode (cur)": center(c),
        "status": "excluded" if col in result.excluded_features else ("tested" if col in result.feature_map else "not comparable"),
    })

with st.container(border=True):
    st.markdown("#### Column profiles")
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        column_config={
            "missing % (ref)": st.column_config.NumberColumn(format="%.2f%%"),
            "missing % (cur)": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )

with st.container(border=True):
    st.markdown("#### Feature statistics side by side")
    options = [c for c in s.common_columns]
    feature = st.selectbox("Column", options, key="cmp_feature")
    if feature:
        r, c = rp.columns[feature], cp.columns[feature]
        if r.feature_type == NUMERICAL and c.feature_type == NUMERICAL:
            keys = ["count", "mean", "std", "min", "q25", "median", "q75", "max"]
            df = pd.DataFrame({"statistic": keys, "reference": [r.stats.get(k) for k in keys],
                               "current": [c.stats.get(k) for k in keys]})
            df["difference"] = [
                (cv - rv) if isinstance(cv, (int, float)) and isinstance(rv, (int, float)) else None
                for rv, cv in zip(df["reference"], df["current"])
            ]
            st.dataframe(df, hide_index=True)
        else:
            ref_top = pd.DataFrame(r.stats.get("top", [])).rename(columns={"count": "ref count", "proportion": "ref share"})
            cur_top = pd.DataFrame(c.stats.get("top", [])).rename(columns={"count": "cur count", "proportion": "cur share"})
            merged = ref_top.merge(cur_top, on="category", how="outer") if not ref_top.empty and not cur_top.empty else (ref_top if not ref_top.empty else cur_top)
            st.caption(f"Top categories (ref unique: {r.n_unique}, cur unique: {c.n_unique})")
            st.dataframe(merged, hide_index=True, column_config={
                "ref share": st.column_config.NumberColumn(format="percent"),
                "cur share": st.column_config.NumberColumn(format="percent"),
            })

with st.expander("Preview rows", icon=":material/table_rows:"):
    if state.has_datasets():
        t1, t2 = st.tabs(["Reference", "Current"])
        t1.dataframe(st.session_state[state.REF].head(50))
        t2.dataframe(st.session_state[state.CUR].head(50))
