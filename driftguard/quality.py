"""Data quality checks, kept separate from distribution drift.

Quality issues describe *integrity* problems (missing data, duplicates, type
inconsistencies, schema changes, unseen categories, out-of-range values).
They never change a feature's drift status.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .schema import CATEGORICAL, NUMERICAL, SchemaComparison

CRITICAL = "critical"
WARNING = "warning"
INFO = "info"
SEVERITY_ORDER = {CRITICAL: 0, WARNING: 1, INFO: 2}


@dataclass
class QualityIssue:
    check: str
    severity: str
    message: str
    column: str | None = None
    dataset: str = "current"
    value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def numeric_parse_share(series: pd.Series) -> float:
    """Share of non-missing values that parse as numbers."""
    values = series.dropna()
    if values.empty:
        return 0.0
    parsed = pd.to_numeric(values.astype(str).str.strip(), errors="coerce")
    return float(parsed.notna().mean())


def check_schema(schema: SchemaComparison) -> list[QualityIssue]:
    issues = []
    for col in schema.missing_in_current:
        issues.append(
            QualityIssue(
                "missing_column",
                WARNING,
                f"Column '{col}' exists in the reference data but is missing from the current data; it cannot be monitored.",
                col,
            )
        )
    for col in schema.new_in_current:
        issues.append(
            QualityIssue(
                "new_column",
                INFO,
                f"Column '{col}' is new in the current data and has no reference baseline.",
                col,
            )
        )
    for col in schema.type_mismatches:
        issues.append(
            QualityIssue(
                "type_mismatch",
                CRITICAL,
                f"Column '{col}' is {schema.reference_types[col]} ({schema.reference_dtypes[col]}) in reference "
                f"but {schema.current_types[col]} ({schema.current_dtypes[col]}) in current; drift was not tested.",
                col,
                "both",
            )
        )
    return issues


def check_mixed_types(df: pd.DataFrame, dataset: str, types: dict[str, str]) -> list[QualityIssue]:
    issues = []
    for col in df.columns:
        if types.get(col) != CATEGORICAL:
            continue
        share = numeric_parse_share(df[col])
        if 0.05 <= share <= 0.95:
            issues.append(
                QualityIssue(
                    "inconsistent_type",
                    WARNING,
                    f"Column '{col}' mixes numeric-looking and non-numeric values ({share:.0%} parse as numbers).",
                    col,
                    dataset,
                    share,
                )
            )
    return issues


def check_missing(
    reference: pd.DataFrame, current: pd.DataFrame, common: list[str], config: AnalysisConfig
) -> list[QualityIssue]:
    issues = []
    for col in common:
        ref_pct = float(reference[col].isna().mean())
        cur_pct = float(current[col].isna().mean())
        if cur_pct == 1.0:
            issues.append(
                QualityIssue("empty_column", CRITICAL, f"Column '{col}' has no values in the current data.", col, value=cur_pct)
            )
            continue
        increase = cur_pct - ref_pct
        if increase > config.missing_increase_threshold:
            severity = CRITICAL if cur_pct >= 0.5 else WARNING
            issues.append(
                QualityIssue(
                    "missing_increase",
                    severity,
                    f"Missing values in '{col}' rose from {ref_pct:.1%} to {cur_pct:.1%} ({increase * 100:+.1f} pp).",
                    col,
                    value=increase,
                )
            )
    return issues


def check_duplicates(df: pd.DataFrame, dataset: str) -> list[QualityIssue]:
    n_dup = int(df.duplicated().sum())
    if n_dup == 0:
        return []
    share = n_dup / len(df)
    return [
        QualityIssue(
            "duplicate_rows",
            WARNING if share >= 0.01 else INFO,
            f"{n_dup:,} fully duplicated row(s) in the {dataset} data ({share:.2%}). Rows were kept as uploaded.",
            None,
            dataset,
            share,
        )
    ]


def check_unseen_categories(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]
) -> list[QualityIssue]:
    issues = []
    for col in columns:
        ref_vals = set(reference[col].dropna().astype(str).unique())
        cur = current[col].dropna().astype(str)
        if cur.empty:
            continue
        mask = ~cur.isin(ref_vals)
        if not mask.any():
            continue
        unseen = cur[mask].value_counts()
        share = float(mask.mean())
        examples = ", ".join(f"'{v}'" for v in unseen.index[:3])
        issues.append(
            QualityIssue(
                "unseen_categories",
                WARNING if share >= 0.01 else INFO,
                f"'{col}' has {unseen.size} category(ies) not present in reference ({examples}), "
                f"covering {share:.1%} of current non-missing rows.",
                col,
                value=share,
            )
        )
    return issues


def check_ranges(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], config: AnalysisConfig
) -> list[QualityIssue]:
    issues = []
    for col in columns:
        ref = pd.to_numeric(reference[col], errors="coerce").to_numpy(float)
        cur = pd.to_numeric(current[col], errors="coerce").to_numpy(float)
        ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
        if ref.size == 0 or cur.size == 0:
            continue
        lo, hi = ref.min(), ref.max()
        outside = (cur < lo) | (cur > hi)
        share = float(outside.mean())
        if share > config.range_violation_threshold:
            issues.append(
                QualityIssue(
                    "out_of_range",
                    WARNING,
                    f"{share:.1%} of current '{col}' values fall outside the reference range "
                    f"[{lo:,.4g}, {hi:,.4g}] (current range [{cur.min():,.4g}, {cur.max():,.4g}]).",
                    col,
                    value=share,
                )
            )
    return issues


def check_constant_columns(df: pd.DataFrame, dataset: str, columns: list[str]) -> list[QualityIssue]:
    issues = []
    for col in columns:
        values = df[col].dropna()
        if not values.empty and values.nunique() == 1:
            issues.append(
                QualityIssue(
                    "constant_column",
                    INFO,
                    f"Column '{col}' has a single distinct value in the {dataset} data.",
                    col,
                    dataset,
                )
            )
    return issues


def run_quality_checks(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    schema: SchemaComparison,
    tested_features: dict[str, str],
    excluded: dict[str, str],
    config: AnalysisConfig,
) -> list[QualityIssue]:
    numeric_cols = [c for c, t in tested_features.items() if t == NUMERICAL]
    categorical_cols = [c for c, t in tested_features.items() if t == CATEGORICAL]
    issues: list[QualityIssue] = []
    issues += check_schema(schema)
    issues += check_mixed_types(reference, "reference", schema.reference_types)
    issues += check_mixed_types(current, "current", schema.current_types)
    issues += check_missing(reference, current, schema.common_columns, config)
    issues += check_duplicates(reference, "reference")
    issues += check_duplicates(current, "current")
    issues += check_unseen_categories(reference, current, categorical_cols)
    issues += check_ranges(reference, current, numeric_cols, config)
    issues += check_constant_columns(current, "current", list(tested_features))
    for col, reason in excluded.items():
        if reason.startswith("Identifier"):
            issues.append(QualityIssue("identifier_column", INFO, f"'{col}': {reason}", col, "both"))
    issues.sort(key=lambda i: (SEVERITY_ORDER[i.severity], i.check, i.column or ""))
    return issues


def summarize_issues(issues: list[QualityIssue]) -> dict[str, int]:
    counts = {CRITICAL: 0, WARNING: 0, INFO: 0}
    for issue in issues:
        counts[issue.severity] += 1
    return counts
