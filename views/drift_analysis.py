import streamlit as st

from driftguard.drift import DRIFT, STATUSES
from driftguard.reports import results_dataframe
from driftguard.schema import CATEGORICAL, NUMERICAL
from driftguard.visualization import categorical_bar, ecdf_figure, numeric_histogram
from ui import state
from ui.components import STATUS_BADGE, dataset_context, p_text, page_header, require_result

page_header(
    "Drift Analysis",
    "Per-feature hypothesis tests, multiple-testing-adjusted p-values and effect sizes.",
)

result = require_result()
if result is None:
    st.stop()

dataset_context(result)
cfg = result.config

f1, f2, f3 = st.columns([2, 2, 3])
statuses = f1.multiselect("Status", STATUSES, default=list(STATUSES))
types = f2.multiselect("Feature type", [NUMERICAL, CATEGORICAL], default=[NUMERICAL, CATEGORICAL])
query = f3.text_input("Search features", placeholder="e.g. income").strip().lower()

df = results_dataframe(result)
mask = df["status"].isin(statuses) & df["feature_type"].isin(types)
if query:
    mask &= df["feature"].str.lower().str.contains(query, regex=False)
view = df[mask].sort_values(["status", "effect_size"], ascending=[True, False])

st.dataframe(
    view[["feature", "feature_type", "status", "test", "n_reference", "n_current", "p_value",
          "adjusted_p_value", "effect_size_name", "effect_size", "psi", "explanation"]],
    hide_index=True,
    column_config={
        "feature_type": "type",
        "n_reference": st.column_config.NumberColumn("n ref", format="localized"),
        "n_current": st.column_config.NumberColumn("n cur", format="localized"),
        "p_value": st.column_config.NumberColumn("p-value", format="%.2e"),
        "adjusted_p_value": st.column_config.NumberColumn("adjusted p", format="%.2e",
                                                          help=f"Adjusted with {cfg.correction}; compared with alpha = {cfg.alpha:g}."),
        "effect_size_name": "effect measure",
        "effect_size": st.column_config.ProgressColumn("effect size", min_value=0.0, max_value=1.0, format="%.3f"),
        "psi": st.column_config.NumberColumn("PSI", format="%.3f"),
        "explanation": st.column_config.TextColumn("explanation", width="large"),
    },
)
st.caption(f"{len(view)} of {len(df)} features shown. Excluded columns: "
           + (", ".join(f"`{c}`" for c in result.excluded_features) or "none") + ".")

st.divider()
st.subheader("Feature detail")
features = view["feature"].tolist() or df["feature"].tolist()
if not features:
    st.info("No features to display.")
    st.stop()
default = next((i for i, f in enumerate(features) if result.feature_map[f].status == DRIFT), 0)
name = st.selectbox("Feature", features, index=default)
r = result.feature_map[name]

st.markdown(f"### {r.feature} {STATUS_BADGE[r.status]}")
st.write(r.explanation)

m = st.columns(5)
m[0].metric("Test", r.test.split(" (")[0] if len(r.test) > 28 else r.test, help=r.test, border=True)
m[1].metric("p-value", p_text(r.p_value), border=True)
m[2].metric("Adjusted p", p_text(r.adjusted_p_value), help=f"alpha = {cfg.alpha:g}", border=True)
thr = cfg.ks_effect_threshold if r.feature_type == NUMERICAL else cfg.cramers_v_threshold
m[3].metric(r.effect_size_name or "Effect size", "n/a" if r.effect_size is None else f"{r.effect_size:.3f}",
            help=f"Practical threshold: {thr:g}", border=True)
m[4].metric("PSI", "n/a" if r.psi is None else f"{r.psi:.3f}", help=f"Threshold: {cfg.psi_threshold:g}", border=True)

if r.p_value is not None:
    c1, c2 = st.columns(2)
    c1.markdown(
        ("**Statistically significant** :green-badge[yes]" if r.statistically_significant else
         "**Statistically significant** :gray-badge[no]")
        + f"  \nAdjusted p {'<' if r.statistically_significant else '>='} alpha ({cfg.alpha:g})."
    )
    c2.markdown(
        ("**Practically significant** :green-badge[yes]" if r.practically_significant else
         "**Practically significant** :gray-badge[no]")
        + f"  \nEffect size >= {thr:g} or PSI >= {cfg.psi_threshold:g}."
    )

ref_df, cur_df = st.session_state.get(state.REF), st.session_state.get(state.CUR)
charts_available = ref_df is not None and cur_df is not None and name in ref_df and name in cur_df
if r.feature_type == NUMERICAL and charts_available:
    a, b = st.columns(2)
    with a, st.container(border=True):
        st.plotly_chart(numeric_histogram(ref_df[name], cur_df[name], name), width="stretch")
    with b, st.container(border=True):
        st.plotly_chart(ecdf_figure(ref_df[name], cur_df[name], name), width="stretch")
elif r.feature_type == CATEGORICAL:
    with st.container(border=True):
        st.plotly_chart(categorical_bar(r), width="stretch")
else:
    st.info("Raw data for charts is no longer loaded in this session.")

with st.expander("Raw test details", icon=":material/data_object:"):
    st.json({"reference_stats": r.reference_stats, "current_stats": r.current_stats,
             "details": {k: v for k, v in r.details.items() if k != "proportions"}})
