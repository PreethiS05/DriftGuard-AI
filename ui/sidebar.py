"""Sidebar: data source selection and analysis settings."""

from __future__ import annotations

import streamlit as st

from driftguard.config import CORRECTION_LABELS, CORRECTION_METHODS, AnalysisConfig
from driftguard.data_loader import MAX_FILE_SIZE_MB, DatasetValidationError, list_sample_files

from driftguard.synthetic import SCENARIOS

from . import state

SOURCES = ("Synthetic demo", "Bundled samples", "Upload CSVs")


def settings_config() -> AnalysisConfig | None:
    with st.sidebar.expander("Analysis settings", icon=":material/tune:"):
        alpha = st.number_input(
            "Significance level (alpha)", 0.001, 0.2, 0.05, 0.005, format="%.3f",
            help="Compared with adjusted p-values. Lower = fewer false alarms but less sensitivity.",
        )
        correction = st.selectbox(
            "Multiple-testing correction", CORRECTION_METHODS, format_func=CORRECTION_LABELS.get,
            help="Benjamini-Hochberg controls the false discovery rate; Bonferroni is stricter.",
        )
        require_effect = st.toggle(
            "Require practical effect size", True,
            help="Only flag drift when the effect size or PSI also exceeds its threshold.",
        )
        ks_thr = st.slider("KS D threshold (numerical)", 0.01, 0.5, 0.10, 0.01)
        v_thr = st.slider("Cramer's V threshold (categorical)", 0.01, 0.5, 0.10, 0.01)
        psi_thr = st.slider("PSI threshold", 0.05, 0.5, 0.20, 0.01, help="Rule of thumb: >0.2 significant shift.")
        psi_bins = st.slider("PSI bins (numerical)", 5, 20, 10)
        min_samples = st.number_input("Minimum non-missing samples", 5, 1000, 30, 5)
        missing_thr = st.slider("Missing-rate increase alert (pp)", 1, 50, 5) / 100
        st.checkbox("Save analyses to local history", key=state.SAVE_HISTORY)
    try:
        return AnalysisConfig(
            alpha=float(alpha),
            correction=correction,
            require_practical_effect=require_effect,
            ks_effect_threshold=ks_thr,
            cramers_v_threshold=v_thr,
            psi_threshold=psi_thr,
            psi_bins=int(psi_bins),
            min_samples=int(min_samples),
            missing_increase_threshold=missing_thr,
        )
    except ValueError as exc:
        st.sidebar.error(f"Invalid settings: {exc}")
        return None


def _synthetic_source() -> bool:
    scenario = st.selectbox("Scenario", list(SCENARIOS), format_func=lambda k: SCENARIOS[k].label)
    st.caption(SCENARIOS[scenario].description)
    c1, c2 = st.columns(2)
    n_ref = c1.number_input("Reference rows", 100, 100_000, 5000, 500)
    n_cur = c2.number_input("Current rows", 100, 100_000, 3000, 500)
    seed = st.number_input("Random seed", 0, 2**31 - 1, 42, help="Same seed + scenario = identical data.")
    if st.button("Generate synthetic data", type="secondary", width="stretch", icon=":material/science:"):
        state.load_synthetic(scenario, int(seed), int(n_ref), int(n_cur))
        return True
    return False


def _sample_source() -> bool:
    files = list_sample_files()
    if not files:
        st.warning("No sample files found in data/samples. Run `python scripts/generate_samples.py`.")
        return False
    ref_default = files.index("reference_customers.csv") if "reference_customers.csv" in files else 0
    ref_name = st.selectbox("Reference file", files, index=ref_default)
    cur_options = [f for f in files if f != ref_name] or files
    cur_name = st.selectbox("Current file", cur_options)
    if st.button("Load samples", width="stretch", icon=":material/folder_open:"):
        state.load_samples(ref_name, cur_name)
        return True
    return False


def _upload_source() -> bool:
    ref_file = st.file_uploader("Reference dataset (CSV)", type=["csv"], key="ref_upload")
    cur_file = st.file_uploader("Current dataset (CSV)", type=["csv"], key="cur_upload")
    st.caption(f"UTF-8 CSV with a header row, up to {MAX_FILE_SIZE_MB} MB each.")
    if st.button("Load uploaded files", width="stretch", disabled=not (ref_file and cur_file), icon=":material/upload:"):
        state.load_uploaded(ref_file, cur_file)
        return True
    return False


def render_sidebar() -> AnalysisConfig | None:
    with st.sidebar:
        st.markdown("### :material/shield: DriftGuard AI")
        st.caption("Data drift detection & monitoring")
        source = st.segmented_control("Data source", SOURCES, default=SOURCES[0], required=True)
        loaded = False
        try:
            if source == "Synthetic demo":
                loaded = _synthetic_source()
            elif source == "Bundled samples":
                loaded = _sample_source()
            else:
                loaded = _upload_source()
        except DatasetValidationError as exc:
            st.error(str(exc), icon=":material/error:")
        except ValueError as exc:
            st.error(f"Could not load data: {exc}", icon=":material/error:")

        if loaded:
            st.success("Datasets loaded.", icon=":material/check_circle:")
        for w in st.session_state.get(state.WARNINGS, []):
            st.warning(w, icon=":material/warning:")

        if state.has_datasets():
            ref_meta, cur_meta = st.session_state[state.REF_META], st.session_state[state.CUR_META]
            st.caption(
                f"Loaded: **{ref_meta.name}** ({ref_meta.n_rows:,} rows) vs **{cur_meta.name}** ({cur_meta.n_rows:,} rows)"
            )

    config = settings_config()

    with st.sidebar:
        run = st.button(
            "Run analysis", type="primary", width="stretch", icon=":material/play_arrow:",
            disabled=not state.has_datasets() or config is None,
        )
        if run and config is not None:
            try:
                with st.spinner("Running drift tests..."):
                    state.run_current_analysis(config)
                st.toast("Analysis complete.", icon=":material/check_circle:")
            except DatasetValidationError as exc:
                st.error(str(exc), icon=":material/error:")
        elif state.has_datasets() and state.get_result() is None:
            st.info("Datasets loaded. Click **Run analysis**.", icon=":material/info:")
    return config
