import streamlit as st

from driftguard.config import AnalysisConfig
from driftguard.drift import DRIFT, INSUFFICIENT
from driftguard.methodology import DRIFT_VS_DEGRADATION, METHODOLOGY
from driftguard.synthetic import SCENARIOS
from driftguard.visualization import psi_overview
from ui import state
from ui.components import STATUS_BADGE, dataset_context, page_header

page_header(
    "Overview",
    "Compare a reference dataset with a current dataset to detect statistically significant, "
    "practically meaningful changes in feature distributions.",
)

config: AnalysisConfig = st.session_state.get("config") or AnalysisConfig()
result = state.get_result()


def demo_panel() -> None:
    with st.container(border=True):
        st.subheader("Try it in one click")
        st.write(
            "Generate a reproducible **synthetic** e-commerce dataset with a known, injected change, "
            "then run the full analysis. Every scenario documents which features were changed so you can "
            "check that the detector finds exactly those."
        )
        cols = st.columns(4)
        for col, key in zip(cols, ("combined", "mean_shift", "category_shift", "stable")):
            if col.button(SCENARIOS[key].label, key=f"demo_{key}", width="stretch",
                          type="primary" if key == "combined" else "secondary"):
                with st.spinner("Generating data and running drift tests..."):
                    state.run_demo(key, config)
                st.rerun()
        st.caption("More scenarios, custom seeds and sample sizes are available in the sidebar. Or upload your own CSVs.")


if result is None:
    demo_panel()
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("#### How it works")
        for title, text in METHODOLOGY[:4]:
            st.markdown(f"**{title}.** {text}")
    with c2, st.container(border=True):
        st.markdown("#### What this tool measures")
        for title, text in DRIFT_VS_DEGRADATION:
            st.markdown(f"**{title}** - {text}")
    st.stop()

dataset_context(result)
for note in result.notes:
    st.info(note, icon=":material/science:")

summary = result.summary()
q = summary["quality_counts"]
if summary["n_drifted"]:
    st.error(summary["headline"], icon=":material/warning:")
elif summary["n_features_analyzed"] > summary["n_insufficient"]:
    st.success(summary["headline"], icon=":material/check_circle:")
else:
    st.warning(summary["headline"], icon=":material/info:")

m = st.columns(5)
m[0].metric("Features analysed", summary["n_features_analyzed"], border=True,
            help=f"{summary['n_excluded']} column(s) excluded (identifiers or type mismatches).")
m[1].metric("Drift detected", summary["n_drifted"], border=True,
            help="Adjusted p-value below alpha and effect size above threshold.")
m[2].metric("Insufficient data", summary["n_insufficient"], border=True)
m[3].metric("Quality issues", summary["n_quality_issues"], border=True,
            help=f"{q['critical']} critical, {q['warning']} warning, {q['info']} info. Reported separately from drift.")
m[4].metric("Rows (ref / cur)", f"{result.reference_meta.n_rows:,} / {result.current_meta.n_rows:,}", border=True)

left, right = st.columns([3, 2])
with left, st.container(border=True):
    st.plotly_chart(psi_overview(result.features, config=result.config), width="stretch")
    st.caption("Bar colour shows final status (red = drift, green = no significant drift, grey = insufficient data).")

with right:
    with st.container(border=True):
        st.markdown("#### Features with detected drift")
        drifted = [f for f in result.features if f.status == DRIFT]
        if not drifted:
            st.write("None. No feature passed both the significance and effect-size criteria.")
        for f in sorted(drifted, key=lambda r: -(r.effect_size or 0)):
            st.markdown(f"**{f.feature}** {STATUS_BADGE[f.status]}  \n{f.explanation}")
        insufficient = [f.feature for f in result.features if f.status == INSUFFICIENT]
        if insufficient:
            st.caption("Insufficient data: " + ", ".join(insufficient))
    with st.container(border=True):
        st.markdown("#### Data quality at a glance")
        if not result.quality_issues:
            st.write("No data quality issues found.")
        for issue in result.quality_issues[:5]:
            st.markdown(f"- **{issue.severity.title()}** - {issue.message}")
        if len(result.quality_issues) > 5:
            st.page_link("views/quality.py", label=f"See all {len(result.quality_issues)} issues", icon=":material/arrow_forward:")

st.info(
    "Data drift describes a change in input distributions. It does not by itself mean model accuracy "
    "has dropped; no model was evaluated in this analysis.",
    icon=":material/info:",
)
