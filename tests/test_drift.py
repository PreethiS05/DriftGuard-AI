import numpy as np
import pandas as pd
import pytest

from driftguard.config import AnalysisConfig
from driftguard.drift import (
    DRIFT,
    INSUFFICIENT,
    NO_DRIFT,
    adjust_p_values,
    analyze_categorical,
    analyze_numerical,
    detect_drift,
    psi_categorical,
    psi_numeric,
)


def test_identical_numeric_distribution_has_no_drift(config, rng):
    data = rng.normal(10, 2, 2000)
    r = analyze_numerical("x", pd.Series(data), pd.Series(data.copy()), config)
    assert r.p_value == pytest.approx(1.0)
    assert r.effect_size == pytest.approx(0.0)
    assert r.psi == pytest.approx(0.0, abs=1e-9)


def test_numeric_mean_shift_is_detected(config, rng):
    ref = pd.Series(rng.normal(0, 1, 2000))
    cur = pd.Series(rng.normal(0.8, 1, 2000))
    [r] = detect_drift(pd.DataFrame({"x": ref}), pd.DataFrame({"x": cur}), {"x": "numerical"}, config)
    assert r.status == DRIFT
    assert r.statistically_significant and r.practically_significant
    assert "SD)" in r.explanation
    assert "%" not in r.explanation


def test_numeric_variance_shift_is_detected(config, rng):
    ref = pd.DataFrame({"x": rng.normal(0, 1, 3000)})
    cur = pd.DataFrame({"x": rng.normal(0, 2.5, 3000)})
    [r] = detect_drift(ref, cur, {"x": "numerical"}, config)
    assert r.status == DRIFT


def test_tiny_but_significant_shift_needs_practical_effect(rng):
    n = 200_000
    ref = pd.DataFrame({"x": rng.normal(0, 1, n)})
    cur = pd.DataFrame({"x": rng.normal(0.03, 1, n)})
    strict = detect_drift(ref, cur, {"x": "numerical"}, AnalysisConfig())[0]
    assert strict.statistically_significant
    assert not strict.practically_significant
    assert strict.status == NO_DRIFT
    loose = detect_drift(ref, cur, {"x": "numerical"}, AnalysisConfig(require_practical_effect=False))[0]
    assert loose.status == DRIFT


def test_insufficient_samples(config, rng):
    r = analyze_numerical("x", pd.Series(rng.normal(size=500)), pd.Series(rng.normal(size=10)), config)
    assert r.status == INSUFFICIENT
    assert r.p_value is None
    assert "at least 30" in r.explanation


def test_missing_and_infinite_values_are_ignored(config, rng):
    cur = pd.Series(np.r_[rng.normal(size=500), [np.nan] * 50, [np.inf, -np.inf]])
    ref = pd.DataFrame({"x": rng.normal(size=500)})
    [r] = detect_drift(ref, pd.DataFrame({"x": cur}), {"x": "numerical"}, config)
    assert r.n_current == 500
    assert r.status != INSUFFICIENT


def test_constant_columns_are_handled(config):
    frame = pd.DataFrame({"x": [1.0] * 100})
    [r] = detect_drift(frame, frame.copy(), {"x": "numerical"}, config)
    assert r.status == NO_DRIFT
    assert r.effect_size == pytest.approx(0.0)
    assert r.psi == pytest.approx(0.0, abs=1e-9)


def test_categorical_shift_is_detected(config, rng):
    ref = pd.Series(rng.choice(["a", "b", "c"], 3000, p=[0.6, 0.3, 0.1]))
    cur = pd.Series(rng.choice(["a", "b", "c"], 3000, p=[0.3, 0.3, 0.4]))
    [r] = detect_drift(pd.DataFrame({"c": ref}), pd.DataFrame({"c": cur}), {"c": "categorical"}, config)
    assert r.status == DRIFT
    assert r.effect_size_name == "Cramer's V"


def test_categorical_stable_has_no_drift(config, small_frames):
    ref, cur = small_frames
    [r] = detect_drift(ref, cur, {"cat": "categorical"}, config)
    assert r.status == NO_DRIFT


def test_new_category_is_reported(config, rng):
    ref = pd.Series(rng.choice(["a", "b"], 2000))
    cur = pd.Series(rng.choice(["a", "b", "z"], 2000, p=[0.4, 0.4, 0.2]))
    [r] = detect_drift(pd.DataFrame({"c": ref}), pd.DataFrame({"c": cur}), {"c": "categorical"}, config)
    assert r.status == DRIFT
    assert "z" in r.explanation
    assert r.p_value is not None and r.p_value < 0.05


def test_sparse_categories_use_exact_fallback(config, rng):
    ref = pd.Series(["a"] * 40 + ["b"] * 2)
    cur = pd.Series(["a"] * 38 + ["b"] * 1 + ["c"] * 1)
    r = analyze_categorical("c", ref, cur, config)
    assert r.p_value is not None
    assert 0 <= r.p_value <= 1


def test_single_category_both_sides(config):
    frame = pd.DataFrame({"c": ["a"] * 100})
    [r] = detect_drift(frame, frame.copy(), {"c": "categorical"}, config)
    assert r.status == NO_DRIFT


def test_benjamini_hochberg_matches_reference_values():
    p = [0.01, 0.04, 0.03, 0.005]
    adj = adjust_p_values(p, "fdr_bh")
    assert adj == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_bonferroni_is_capped_at_one():
    assert adjust_p_values([0.3, 0.01], "bonferroni") == pytest.approx([0.6, 0.02])
    assert adjust_p_values([0.9, 0.8], "bonferroni") == pytest.approx([1.0, 1.0])


def test_no_correction_returns_raw():
    assert adjust_p_values([0.2, 0.01], "none") == pytest.approx([0.2, 0.01])


def test_multiple_testing_reduces_false_positives():
    rng = np.random.default_rng(123)
    n_features = 40
    ref = pd.DataFrame({f"f{i}": rng.normal(size=400) for i in range(n_features)})
    cur = pd.DataFrame({f"f{i}": rng.normal(size=400) for i in range(n_features)})
    features = {c: "numerical" for c in ref}
    raw = detect_drift(ref, cur, features, AnalysisConfig(correction="none", require_practical_effect=False))
    bh = detect_drift(ref, cur, features, AnalysisConfig(correction="fdr_bh", require_practical_effect=False))
    assert sum(r.status == DRIFT for r in bh) <= sum(r.status == DRIFT for r in raw)
    assert sum(r.status == DRIFT for r in bh) == 0


def test_psi_is_non_negative_and_grows_with_shift(rng):
    ref = rng.normal(0, 1, 5000)
    small = psi_numeric(ref, rng.normal(0.1, 1, 5000))
    large = psi_numeric(ref, rng.normal(1.0, 1, 5000))
    assert 0 <= small < large
    assert large > 0.2


def test_psi_categorical_handles_unseen_categories():
    ref = pd.Series({"a": 50, "b": 50})
    cur = pd.Series({"a": 40, "b": 40, "c": 20})
    value = psi_categorical(ref, cur)
    assert np.isfinite(value) and value > 0


def test_results_are_json_safe_numbers(config, rng):
    r = analyze_numerical("x", pd.Series(rng.normal(size=300)), pd.Series(rng.normal(size=300)), config)
    d = r.to_dict()
    for key in ("p_value", "effect_size", "psi"):
        assert d[key] is None or np.isfinite(d[key])
