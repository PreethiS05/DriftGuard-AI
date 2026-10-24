import pandas as pd
import pytest

from driftguard.data_loader import (
    DatasetValidationError,
    list_sample_files,
    load_csv_bytes,
    load_sample,
    safe_display_name,
)
from driftguard.schema import CATEGORICAL, NUMERICAL, compare_schemas, ensure_compatible, infer_feature_type


def test_loads_valid_csv():
    ds = load_csv_bytes(b"a,b\n1,x\n2,y\n", "data.csv")
    assert ds.frame.shape == (2, 2)


@pytest.mark.parametrize(
    "data,name,match",
    [
        (b"", "empty.csv", "empty"),
        (b"a,b\n1,2\n", "data.txt", "not a .csv"),
        (b"a,a\n1,2\n", "dup.csv", "[Dd]uplicate"),
        (b"\x00\x01\x02binary", "bin.csv", None),
    ],
)
def test_rejects_invalid_files(data, name, match):
    with pytest.raises(DatasetValidationError, match=match):
        load_csv_bytes(data, name)


def test_rejects_oversized_file():
    with pytest.raises(DatasetValidationError):
        load_csv_bytes(b"a\n" + b"1\n" * 1000, "big.csv", max_bytes=100)


def test_header_only_file_is_rejected():
    with pytest.raises(DatasetValidationError):
        load_csv_bytes(b"a,b\n", "header.csv")


def test_latin1_fallback_warns():
    ds = load_csv_bytes("name,v\ncaf\xe9,1\n".encode("latin-1"), "l1.csv")
    assert ds.frame.iloc[0, 0] == "caf\xe9"
    assert ds.warnings


def test_display_name_is_sanitised():
    assert "/" not in safe_display_name("../../etc/passwd.csv")
    assert safe_display_name("C:\\Users\\me\\data.csv") == "data.csv"


def test_sample_files_are_listed_and_load():
    files = list_sample_files()
    assert "reference_customers.csv" in files
    assert load_sample("reference_customers.csv").frame.shape[0] > 1000


def test_sample_loader_rejects_path_traversal():
    with pytest.raises(DatasetValidationError):
        load_sample("../README.md")


def test_feature_type_inference():
    assert infer_feature_type(pd.Series([1.0, 2.5, 3.0])) == NUMERICAL
    assert infer_feature_type(pd.Series(["a", "b"])) == CATEGORICAL
    assert infer_feature_type(pd.Series([True, False])) == CATEGORICAL


def test_schema_comparison_detects_changes():
    ref = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [1.0, 2.0]})
    cur = pd.DataFrame({"a": [1, 2], "b": [1, 2], "d": ["z", "w"]})
    s = compare_schemas(ref, cur)
    assert s.missing_in_current == ["c"]
    assert s.new_in_current == ["d"]
    assert s.type_mismatches == ["b"]
    assert set(s.common_columns) == {"a", "b"}


def test_no_common_columns_is_incompatible():
    s = compare_schemas(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}))
    with pytest.raises(DatasetValidationError):
        ensure_compatible(s)
