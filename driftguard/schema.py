"""Feature type inference and schema comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd
from pandas.api import types as ptypes

from .data_loader import DatasetValidationError

NUMERICAL = "numerical"
CATEGORICAL = "categorical"


def infer_feature_type(series: pd.Series) -> str:
    if ptypes.is_bool_dtype(series):
        return CATEGORICAL
    if ptypes.is_numeric_dtype(series):
        return NUMERICAL
    return CATEGORICAL


def is_identifier_like(series: pd.Series, unique_ratio: float, min_rows: int = 50) -> bool:
    """Categorical columns where almost every value is unique (IDs, free text)."""
    values = series.dropna()
    if len(values) < min_rows:
        return False
    return values.nunique() / len(values) >= unique_ratio


@dataclass
class SchemaComparison:
    common_columns: list[str]
    missing_in_current: list[str]
    new_in_current: list[str]
    reference_types: dict[str, str]
    current_types: dict[str, str]
    reference_dtypes: dict[str, str]
    current_dtypes: dict[str, str]
    type_mismatches: list[str] = field(default_factory=list)

    @property
    def has_differences(self) -> bool:
        return bool(self.missing_in_current or self.new_in_current or self.type_mismatches)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_schemas(reference: pd.DataFrame, current: pd.DataFrame) -> SchemaComparison:
    ref_cols = list(reference.columns)
    cur_cols = list(current.columns)
    cur_set = set(cur_cols)
    ref_set = set(ref_cols)
    common = [c for c in ref_cols if c in cur_set]
    ref_types = {c: infer_feature_type(reference[c]) for c in ref_cols}
    cur_types = {c: infer_feature_type(current[c]) for c in cur_cols}
    mismatches = [
        c
        for c in common
        if ref_types[c] != cur_types[c]
        and not (reference[c].isna().all() or current[c].isna().all())
    ]
    return SchemaComparison(
        common_columns=common,
        missing_in_current=[c for c in ref_cols if c not in cur_set],
        new_in_current=[c for c in cur_cols if c not in ref_set],
        reference_types=ref_types,
        current_types=cur_types,
        reference_dtypes={c: str(reference[c].dtype) for c in ref_cols},
        current_dtypes={c: str(current[c].dtype) for c in cur_cols},
        type_mismatches=mismatches,
    )


def ensure_compatible(schema: SchemaComparison) -> None:
    if not schema.common_columns:
        raise DatasetValidationError(
            "The reference and current datasets share no column names, so they "
            "cannot be compared. Check that both files describe the same features."
        )
