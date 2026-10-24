import numpy as np
import pandas as pd

from driftguard.config import AnalysisConfig
from driftguard.pipeline import run_analysis


def checks(result):
    return {i.check for i in result.quality_issues}


def base(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.normal(50, 5, n),
        "cat": rng.choice(["a", "b"], n),
    })


def test_clean_data_has_no_warnings():
    res = run_analysis(base(seed=1), base(seed=2))
    assert not [i for i in res.quality_issues if i.severity != "info"]


def test_missing_value_increase_is_flagged():
    cur = base(seed=2)
    cur.loc[:150, "x"] = np.nan
    res = run_analysis(base(seed=1), cur)
    assert "missing_increase" in checks(res)


def test_duplicate_rows_are_flagged():
    cur = base(seed=2)
    cur = pd.concat([cur, cur.head(100)], ignore_index=True)
    assert "duplicate_rows" in checks(run_analysis(base(seed=1), cur))


def test_unseen_categories_are_flagged():
    cur = base(seed=2)
    cur.loc[:60, "cat"] = "new"
    assert "unseen_categories" in checks(run_analysis(base(seed=1), cur))


def test_out_of_range_values_are_flagged():
    cur = base(seed=2)
    cur.loc[:40, "x"] = 10_000
    assert "out_of_range" in checks(run_analysis(base(seed=1), cur, AnalysisConfig()))


def test_constant_column_is_flagged():
    cur = base(seed=2)
    cur["cat"] = "a"
    assert "constant_column" in checks(run_analysis(base(seed=1), cur))


def test_schema_issues_are_flagged():
    ref = base(seed=1).assign(extra=1.0)
    cur = base(seed=2).assign(new_col="z")
    cur["x"] = cur["x"].astype(str) + "kg"
    found = checks(run_analysis(ref, cur))
    assert {"missing_column", "new_column", "type_mismatch"} <= found


def test_quality_issues_do_not_change_drift_status():
    ref, cur = base(seed=1), base(seed=2)
    clean = run_analysis(ref, cur)
    dup = run_analysis(ref, pd.concat([cur, cur], ignore_index=True))
    assert [f.feature for f in clean.features] == [f.feature for f in dup.features]
