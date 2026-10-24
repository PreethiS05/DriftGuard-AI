"""Descriptive profiling of a single dataset."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .schema import NUMERICAL, infer_feature_type


def clean_number(value: Any) -> float | int | None:
    """Convert numpy scalars to JSON-safe Python numbers (NaN/inf -> None)."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def numeric_summary(values: pd.Series) -> dict[str, Any]:
    v = pd.to_numeric(values, errors="coerce").dropna()
    v = v[np.isfinite(v.to_numpy(dtype=float))]
    if v.empty:
        return {k: None for k in ("count", "mean", "median", "std", "min", "q25", "q75", "max")} | {"count": 0}
    arr = v.to_numpy(dtype=float)
    return {
        "count": int(arr.size),
        "mean": clean_number(arr.mean()),
        "median": clean_number(np.median(arr)),
        "std": clean_number(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": clean_number(arr.min()),
        "q25": clean_number(np.quantile(arr, 0.25)),
        "q75": clean_number(np.quantile(arr, 0.75)),
        "max": clean_number(arr.max()),
    }


def categorical_summary(values: pd.Series, top_n: int = 10) -> dict[str, Any]:
    v = values.dropna().astype(str)
    counts = v.value_counts()
    total = int(counts.sum())
    top = [
        {"category": str(cat), "count": int(cnt), "proportion": cnt / total if total else 0.0}
        for cat, cnt in counts.head(top_n).items()
    ]
    return {
        "count": total,
        "n_unique": int(counts.size),
        "mode": str(counts.index[0]) if total else None,
        "top": top,
    }


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    feature_type: str
    n_missing: int
    pct_missing: float
    n_unique: int
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetProfile:
    n_rows: int
    n_columns: int
    n_duplicate_rows: int
    pct_duplicate_rows: float
    n_missing_cells: int
    pct_missing_cells: float
    memory_mb: float
    columns: dict[str, ColumnProfile]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_dataset(df: pd.DataFrame, feature_types: dict[str, str] | None = None) -> DatasetProfile:
    n_rows, n_cols = df.shape
    columns: dict[str, ColumnProfile] = {}
    for col in df.columns:
        series = df[col]
        ftype = (feature_types or {}).get(col) or infer_feature_type(series)
        n_missing = int(series.isna().sum())
        stats = numeric_summary(series) if ftype == NUMERICAL else categorical_summary(series)
        columns[col] = ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            feature_type=ftype,
            n_missing=n_missing,
            pct_missing=n_missing / n_rows if n_rows else 0.0,
            n_unique=int(series.nunique(dropna=True)),
            stats=stats,
        )
    n_dup = int(df.duplicated().sum()) if n_rows else 0
    total_cells = n_rows * n_cols
    n_missing_cells = int(df.isna().sum().sum())
    return DatasetProfile(
        n_rows=n_rows,
        n_columns=n_cols,
        n_duplicate_rows=n_dup,
        pct_duplicate_rows=n_dup / n_rows if n_rows else 0.0,
        n_missing_cells=n_missing_cells,
        pct_missing_cells=n_missing_cells / total_cells if total_cells else 0.0,
        memory_mb=float(df.memory_usage(deep=True).sum()) / 1_048_576,
        columns=columns,
    )
