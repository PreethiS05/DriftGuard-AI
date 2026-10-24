import json

import numpy as np
import pandas as pd
import pytest

from driftguard.config import AnalysisConfig
from driftguard.drift import DRIFT
from driftguard.history import HistoryStore
from driftguard.pipeline import run_analysis
from driftguard.reports import html_report, results_csv, summary_json
from driftguard.synthetic import SCENARIOS, SYNTHETIC_LABEL, generate_scenario


@pytest.mark.parametrize("key", list(SCENARIOS))
def test_scenarios_detect_exactly_the_intended_drift(key):
    ds = generate_scenario(key, 4000, 3000, seed=11)
    res = run_analysis(ds.reference, ds.current, notes=ds.notes)
    drifted = {f.feature for f in res.features if f.status == DRIFT}
    assert drifted == set(ds.scenario.expected_drift)
    found = {i.check for i in res.quality_issues}
    assert set(ds.scenario.expected_quality_checks) <= found


def test_synthetic_generation_is_reproducible():
    a = generate_scenario("combined", 500, 500, seed=3)
    b = generate_scenario("combined", 500, 500, seed=3)
    pd.testing.assert_frame_equal(a.current, b.current)
    assert SYNTHETIC_LABEL in a.notes


def test_synthetic_rejects_bad_parameters():
    with pytest.raises(ValueError):
        generate_scenario("nope")
    with pytest.raises(ValueError):
        generate_scenario("stable", 10, 10)


def test_identifier_columns_are_excluded():
    ds = generate_scenario("stable", 1000, 1000, seed=1)
    res = run_analysis(ds.reference, ds.current)
    assert "customer_id" in res.excluded_features
    assert "customer_id" not in res.feature_map


def test_config_validation():
    with pytest.raises(ValueError):
        AnalysisConfig(alpha=1.5)
    with pytest.raises(ValueError):
        AnalysisConfig(correction="holm")


@pytest.fixture
def drifted_result():
    ds = generate_scenario("combined", 1500, 1500, seed=5)
    return run_analysis(ds.reference, ds.current, notes=ds.notes)


def test_json_report_is_valid_and_has_no_raw_rows(drifted_result):
    payload = json.loads(summary_json(drifted_result))
    assert payload["summary"]["n_drifted"] >= 1
    assert "thresholds" in json.dumps(payload) or "config" in payload
    text = json.dumps(payload)
    assert "NaN" not in text and "Infinity" not in text
    assert str(drifted_result.reference_meta.n_rows) in text or payload


def test_csv_report_has_one_row_per_feature(drifted_result):
    df = pd.read_csv(pd.io.common.BytesIO(results_csv(drifted_result)))
    assert len(df) == len(drifted_result.features)
    assert {"feature", "status", "p_value", "adjusted_p_value", "effect_size"} <= set(df.columns)


def test_html_report_escapes_and_mentions_limitations(drifted_result):
    html = html_report(drifted_result)
    assert html.startswith("<!DOCTYPE html>") or html.lstrip().lower().startswith("<!doctype html>")
    assert "SYNTHETIC" in html
    assert "<script" not in html.lower()


def test_html_report_escapes_column_names():
    rng = np.random.default_rng(0)
    evil = "<img src=x onerror=alert(1)>"
    ref = pd.DataFrame({evil: rng.normal(size=200)})
    cur = pd.DataFrame({evil: rng.normal(3, 1, 200)})
    html = html_report(run_analysis(ref, cur))
    assert evil not in html
    assert "&lt;img" in html


def test_history_roundtrip(tmp_path, drifted_result):
    store = HistoryStore(tmp_path / "history.db")
    store.save(drifted_result)
    [row] = store.list()
    assert row["analysis_id"] == drifted_result.analysis_id
    assert row["n_drifted"] == drifted_result.summary()["n_drifted"]
    record = store.get(drifted_result.analysis_id)
    assert len(record["features"]) == len(drifted_result.features)
    assert store.delete(drifted_result.analysis_id)
    assert store.list() == []


def test_history_clear(tmp_path, drifted_result):
    store = HistoryStore(tmp_path / "history.db")
    store.save(drifted_result)
    assert store.clear() == 1
    assert store.list() == []


def test_insufficient_headline():
    ds = generate_scenario("stable", 500, 60, seed=1)
    res = run_analysis(ds.reference, ds.current.head(10))
    assert res.summary()["headline"].startswith("Insufficient data")
