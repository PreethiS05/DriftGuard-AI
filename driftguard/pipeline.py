"""End-to-end analysis pipeline: schema -> profiling -> drift -> quality."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .config import AnalysisConfig
from .drift import DRIFT, INSUFFICIENT, NO_DRIFT, FeatureDriftResult, detect_drift
from .profiling import DatasetProfile, profile_dataset
from .quality import QualityIssue, run_quality_checks, summarize_issues
from .schema import NUMERICAL, SchemaComparison, compare_schemas, ensure_compatible, is_identifier_like

APP_VERSION = "1.0.0"


@dataclass
class DatasetMeta:
    name: str
    n_rows: int
    n_columns: int
    source: str = "upload"
    size_bytes: int | None = None
    synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class AnalysisResult:
    analysis_id: str
    timestamp: str
    config: AnalysisConfig
    reference_meta: DatasetMeta
    current_meta: DatasetMeta
    schema: SchemaComparison
    reference_profile: DatasetProfile
    current_profile: DatasetProfile
    features: list[FeatureDriftResult]
    excluded_features: dict[str, str]
    quality_issues: list[QualityIssue]
    notes: list[str] = field(default_factory=list)

    @property
    def feature_map(self) -> dict[str, FeatureDriftResult]:
        return {f.feature: f for f in self.features}

    def summary(self) -> dict[str, Any]:
        n = len(self.features)
        drifted = [f.feature for f in self.features if f.status == DRIFT]
        counts = {
            DRIFT: len(drifted),
            NO_DRIFT: sum(f.status == NO_DRIFT for f in self.features),
            INSUFFICIENT: sum(f.status == INSUFFICIENT for f in self.features),
        }
        quality = summarize_issues(self.quality_issues)
        tested = n - counts[INSUFFICIENT]
        if n == 0:
            headline = "No features could be analysed."
        elif tested == 0:
            headline = f"Insufficient data: none of the {n} features had enough non-missing values to test."
        elif drifted:
            headline = f"Drift detected in {len(drifted)} of {tested} tested features."
        else:
            headline = f"No significant drift detected across {tested} tested features."
        if n and counts[INSUFFICIENT] and tested:
            headline += f" {counts[INSUFFICIENT]} feature(s) had insufficient data."
        return {
            "headline": headline,
            "n_features_analyzed": n,
            "n_drifted": counts[DRIFT],
            "n_no_drift": counts[NO_DRIFT],
            "n_insufficient": counts[INSUFFICIENT],
            "drifted_features": drifted,
            "share_drifted": len(drifted) / n if n else 0.0,
            "n_excluded": len(self.excluded_features),
            "quality_counts": quality,
            "n_quality_issues": len(self.quality_issues),
        }


def select_features(
    reference: pd.DataFrame, current: pd.DataFrame, schema: SchemaComparison, config: AnalysisConfig
) -> tuple[dict[str, str], dict[str, str]]:
    """Return (features to test -> type, excluded columns -> reason)."""
    features: dict[str, str] = {}
    excluded: dict[str, str] = {}
    for col in schema.common_columns:
        if col in schema.type_mismatches:
            excluded[col] = "Type mismatch between datasets; see Data Quality."
            continue
        ftype = schema.reference_types[col]
        ref_col = reference[col]
        looks_like_id = (
            is_identifier_like(ref_col, config.identifier_unique_ratio)
            if ftype != NUMERICAL
            else (
                (col.lower().endswith("id") or col.lower() in {"index", "unnamed: 0"})
                and is_identifier_like(ref_col, config.identifier_unique_ratio)
            )
        )
        if looks_like_id:
            excluded[col] = "Identifier-like column (almost every value is unique); not a meaningful feature for drift."
            continue
        features[col] = ftype
    return features, excluded


def run_analysis(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: AnalysisConfig | None = None,
    reference_meta: DatasetMeta | None = None,
    current_meta: DatasetMeta | None = None,
    notes: list[str] | None = None,
) -> AnalysisResult:
    config = config or AnalysisConfig()
    schema = compare_schemas(reference, current)
    ensure_compatible(schema)

    features, excluded = select_features(reference, current, schema, config)
    drift_results = detect_drift(reference, current, features, config)
    quality = run_quality_checks(reference, current, schema, features, excluded, config)

    return AnalysisResult(
        analysis_id=uuid.uuid4().hex[:12],
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config=config,
        reference_meta=reference_meta or DatasetMeta("reference", *reference.shape),
        current_meta=current_meta or DatasetMeta("current", *current.shape),
        schema=schema,
        reference_profile=profile_dataset(reference, schema.reference_types),
        current_profile=profile_dataset(current, schema.current_types),
        features=drift_results,
        excluded_features=excluded,
        quality_issues=quality,
        notes=list(notes or []),
    )
