"""Shared methodology and limitation text used by the UI and exported reports."""

from __future__ import annotations

from .config import CORRECTION_LABELS, AnalysisConfig

METHODOLOGY: list[tuple[str, str]] = [
    (
        "Numerical features",
        "Two-sample Kolmogorov-Smirnov (KS) test on non-missing, finite values. The KS statistic D "
        "is the largest vertical gap between the two empirical CDFs and is used as the effect size. "
        "The Population Stability Index (PSI) is computed on bins defined by reference-data quantiles only.",
    ),
    (
        "Categorical features",
        "Chi-square test of homogeneity on a 2 x K contingency table. Categories whose expected count "
        "is below the minimum are merged into one bucket. If the assumption still fails (very small "
        "samples), a Monte Carlo permutation test of the chi-square statistic is used instead. "
        "Effect size is Cramer's V; PSI and total variation distance are also reported.",
    ),
    (
        "Multiple testing",
        "Every tested feature produces one p-value. P-values are adjusted across all features "
        "(Benjamini-Hochberg by default) before comparison with alpha, which limits false alarms "
        "when many features are monitored at once.",
    ),
    (
        "Drift decision",
        "A feature is flagged 'Drift Detected' when its adjusted p-value is below alpha and, by default, "
        "its effect size or PSI also exceeds the practical threshold. Features with too few non-missing "
        "values are reported as 'Insufficient Data'.",
    ),
    (
        "Data quality",
        "Missing values, duplicates, type inconsistencies, schema changes, unseen categories and "
        "out-of-range values are checked separately and never change a feature's drift status. "
        "Missing values are excluded from drift tests; changes in missing rates appear as quality issues.",
    ),
]

LIMITATIONS: list[str] = [
    "Data drift is not model performance degradation. No model is trained or evaluated here; "
    "a drifted feature may or may not affect predictions, and accuracy can drop without any input drift (concept drift).",
    "Statistical significance is not practical importance. With large samples, tiny changes become significant; "
    "with small samples, large changes may not be.",
    "PSI thresholds (0.1 / 0.2) are industry rules of thumb, not statistically derived cut-offs.",
    "No value shown is a calibrated probability of drift. P-values describe how surprising the data would be "
    "if both datasets came from the same distribution.",
    "KS p-values are conservative for discrete or heavily tied numerical data.",
    "Each feature is tested on its own (univariate). Changes in correlations between features are not detected.",
    "Rows are assumed to be independent samples. Time-series autocorrelation inflates false positives.",
]

DRIFT_VS_DEGRADATION: list[tuple[str, str]] = [
    ("Data drift (covariate shift)", "The distribution of input features P(X) changes. This is what DriftGuard AI measures."),
    ("Concept drift", "The relationship between inputs and the target P(y | X) changes, even if inputs look the same. Requires labels to detect."),
    ("Model performance degradation", "Measured accuracy, error or business KPIs get worse. Requires predictions and ground-truth labels."),
]


def threshold_rows(config: AnalysisConfig) -> list[tuple[str, str]]:
    return [
        ("Significance level (alpha)", f"{config.alpha:g}"),
        ("Multiple-testing correction", CORRECTION_LABELS[config.correction]),
        ("Require practical effect size", "Yes" if config.require_practical_effect else "No"),
        ("KS D threshold (numerical)", f"{config.ks_effect_threshold:g}"),
        ("Cramer's V threshold (categorical)", f"{config.cramers_v_threshold:g}"),
        ("PSI threshold", f"{config.psi_threshold:g}"),
        ("PSI bins (numerical)", str(config.psi_bins)),
        ("Minimum non-missing samples", str(config.min_samples)),
        ("Chi-square minimum expected count", f"{config.chi2_min_expected:g}"),
        ("Max categories before grouping", str(config.max_categories)),
        ("Permutation resamples", str(config.permutation_iterations)),
        ("Missing-rate increase threshold", f"{config.missing_increase_threshold:.0%}"),
        ("Out-of-range share threshold", f"{config.range_violation_threshold:.0%}"),
    ]
