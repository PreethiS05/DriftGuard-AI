"""Session-state helpers. All statistics come from the driftguard package."""

from __future__ import annotations

import streamlit as st

from driftguard.config import AnalysisConfig
from driftguard.data_loader import LoadedDataset, load_csv_bytes, load_sample
from driftguard.history import HistoryStore
from driftguard.pipeline import AnalysisResult, DatasetMeta, run_analysis
from driftguard.synthetic import SCENARIOS, SYNTHETIC_LABEL, generate_scenario

REF = "ref_df"
CUR = "cur_df"
REF_META = "ref_meta"
CUR_META = "cur_meta"
NOTES = "dataset_notes"
WARNINGS = "load_warnings"
RESULT = "analysis_result"
SAVE_HISTORY = "save_history"


def init_state() -> None:
    for key, default in ((RESULT, None), (REF, None), (CUR, None), (NOTES, []), (WARNINGS, []), (SAVE_HISTORY, True)):
        st.session_state.setdefault(key, default)


def get_result() -> AnalysisResult | None:
    return st.session_state.get(RESULT)


def has_datasets() -> bool:
    return st.session_state.get(REF) is not None and st.session_state.get(CUR) is not None


@st.cache_resource
def history_store() -> HistoryStore:
    return HistoryStore()


def _set_datasets(ref_df, cur_df, ref_meta: DatasetMeta, cur_meta: DatasetMeta, notes, warnings) -> None:
    st.session_state[REF] = ref_df
    st.session_state[CUR] = cur_df
    st.session_state[REF_META] = ref_meta
    st.session_state[CUR_META] = cur_meta
    st.session_state[NOTES] = list(notes)
    st.session_state[WARNINGS] = list(warnings)
    st.session_state[RESULT] = None


def load_synthetic(scenario: str, seed: int, n_reference: int, n_current: int) -> None:
    data = generate_scenario(scenario, n_reference, n_current, seed)
    label = SCENARIOS[scenario].label
    notes = [SYNTHETIC_LABEL, f"Scenario: {label} (seed {seed}). {data.scenario.description}"]
    _set_datasets(
        data.reference,
        data.current,
        DatasetMeta(f"synthetic_reference_seed{seed}.csv", *data.reference.shape, source="synthetic generator", synthetic=True),
        DatasetMeta(f"synthetic_current_{scenario}_seed{seed}.csv", *data.current.shape, source="synthetic generator", synthetic=True),
        notes,
        [],
    )


def _meta(ds: LoadedDataset, source: str) -> DatasetMeta:
    synthetic = source == "bundled sample"
    return DatasetMeta(ds.name, *ds.frame.shape, source=source, size_bytes=ds.size_bytes, synthetic=synthetic)


def load_uploaded(ref_file, cur_file) -> None:
    ref = load_csv_bytes(ref_file.getvalue(), ref_file.name)
    cur = load_csv_bytes(cur_file.getvalue(), cur_file.name)
    warnings = [f"Reference: {w}" for w in ref.warnings] + [f"Current: {w}" for w in cur.warnings]
    _set_datasets(ref.frame, cur.frame, _meta(ref, "upload"), _meta(cur, "upload"), [], warnings)


def load_samples(ref_name: str, cur_name: str) -> None:
    ref = load_sample(ref_name)
    cur = load_sample(cur_name)
    warnings = [f"Reference: {w}" for w in ref.warnings] + [f"Current: {w}" for w in cur.warnings]
    _set_datasets(
        ref.frame, cur.frame, _meta(ref, "bundled sample"), _meta(cur, "bundled sample"), [SYNTHETIC_LABEL], warnings
    )


def run_current_analysis(config: AnalysisConfig) -> AnalysisResult:
    result = run_analysis(
        st.session_state[REF],
        st.session_state[CUR],
        config,
        st.session_state[REF_META],
        st.session_state[CUR_META],
        st.session_state[NOTES],
    )
    st.session_state[RESULT] = result
    if st.session_state.get(SAVE_HISTORY, True):
        try:
            history_store().save(result)
        except OSError as exc:
            st.toast(f"Analysis complete, but history could not be saved: {exc}")
    return result


def run_demo(scenario: str, config: AnalysisConfig) -> AnalysisResult:
    load_synthetic(scenario, 42, 5000, 3000)
    return run_current_analysis(config)
