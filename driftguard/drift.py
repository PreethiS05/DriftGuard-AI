"""Statistical drift detection for numerical and categorical features.

Design decisions (see README "Drift detection methods"):

* Missing values are excluded from distribution tests. Changes in the
  missing-value rate are reported by the data-quality module instead, so
  distribution drift and data quality stay separate.
* A feature is flagged "Drift Detected" only when its *adjusted* p-value is
  below ``alpha`` AND (by default) an effect-size measure exceeds its
  practical threshold. With large samples, tiny and irrelevant differences
  become statistically significant; the effect-size gate guards against that.
* No score produced here is a probability of drift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .config import AnalysisConfig
from .profiling import categorical_summary, clean_number, numeric_summary
from .schema import CATEGORICAL, NUMERICAL

DRIFT = "Drift Detected"
NO_DRIFT = "No Significant Drift"
INSUFFICIENT = "Insufficient Data"
STATUSES: tuple[str, ...] = (DRIFT, NO_DRIFT, INSUFFICIENT)

PSI_EPSILON = 1e-4
RARE_BUCKET = "(rare categories merged)"
OTHER_BUCKET = "(other categories)"


@dataclass
class FeatureDriftResult:
    feature: str
    feature_type: str
    status: str
    test: str
    n_reference: int
    n_current: int
    p_value: float | None = None
    adjusted_p_value: float | None = None
    statistic: float | None = None
    statistic_name: str | None = None
    effect_size: float | None = None
    effect_size_name: str | None = None
    psi: float | None = None
    statistically_significant: bool = False
    practically_significant: bool = False
    reference_stats: dict[str, Any] = field(default_factory=dict)
    current_stats: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Multiple-testing correction
# --------------------------------------------------------------------------


def adjust_p_values(p_values: list[float], method: str = "fdr_bh") -> list[float]:
    """Return adjusted p-values in the original order.

    * ``fdr_bh`` - Benjamini-Hochberg step-up procedure; controls the expected
      proportion of false discoveries among flagged features.
    * ``bonferroni`` - multiplies by the number of tests; controls the
      probability of *any* false flag (more conservative).
    * ``none`` - returns the raw p-values.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return []
    if np.any((p < 0) | (p > 1) | ~np.isfinite(p)):
        raise ValueError("p-values must be finite and within [0, 1].")
    if method == "none":
        return p.tolist()
    if method == "bonferroni":
        return np.minimum(p * m, 1.0).tolist()
    if method == "fdr_bh":
        order = np.argsort(p)
        ranked = p[order] * m / np.arange(1, m + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        adjusted = np.empty(m)
        adjusted[order] = np.minimum(ranked, 1.0)
        return adjusted.tolist()
    raise ValueError(f"Unknown correction method: {method}")


# --------------------------------------------------------------------------
# Distance measures
# --------------------------------------------------------------------------


def _psi_from_proportions(ref_p: np.ndarray, cur_p: np.ndarray) -> float:
    ref_p = np.clip(ref_p, PSI_EPSILON, None)
    cur_p = np.clip(cur_p, PSI_EPSILON, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def numeric_bin_edges(reference: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior bin edges from reference quantiles (outer bins are open-ended).

    Using only the reference to define bins avoids leaking information from
    the current dataset into the baseline.
    """
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.unique(np.quantile(reference, qs))


def psi_numeric(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    edges = numeric_bin_edges(reference, n_bins)
    ref_idx = np.searchsorted(edges, reference, side="left")
    cur_idx = np.searchsorted(edges, current, side="left")
    k = edges.size + 1
    ref_p = np.bincount(ref_idx, minlength=k) / reference.size
    cur_p = np.bincount(cur_idx, minlength=k) / current.size
    return _psi_from_proportions(ref_p, cur_p)


def psi_categorical(ref_counts: pd.Series, cur_counts: pd.Series) -> float:
    cats = ref_counts.index.union(cur_counts.index)
    ref_p = ref_counts.reindex(cats, fill_value=0).to_numpy(float)
    cur_p = cur_counts.reindex(cats, fill_value=0).to_numpy(float)
    return _psi_from_proportions(ref_p / ref_p.sum(), cur_p / cur_p.sum())


# --------------------------------------------------------------------------
# Numerical features
# --------------------------------------------------------------------------


def _finite_values(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    return f"{value:.{digits}g}"


def analyze_numerical(
    feature: str, reference: pd.Series, current: pd.Series, config: AnalysisConfig
) -> FeatureDriftResult:
    ref = _finite_values(reference)
    cur = _finite_values(current)
    result = FeatureDriftResult(
        feature=feature,
        feature_type=NUMERICAL,
        status=INSUFFICIENT,
        test="Kolmogorov-Smirnov (two-sample)",
        n_reference=int(ref.size),
        n_current=int(cur.size),
        statistic_name="KS D",
        effect_size_name="KS D",
        reference_stats=numeric_summary(reference),
        current_stats=numeric_summary(current),
    )
    if ref.size < config.min_samples or cur.size < config.min_samples:
        result.test = "Not tested"
        result.explanation = (
            f"Only {ref.size} reference and {cur.size} current non-missing values; "
            f"at least {config.min_samples} are required in each dataset."
        )
        return result

    ref_constant = np.ptp(ref) == 0
    cur_constant = np.ptp(cur) == 0
    if ref_constant and cur_constant and ref[0] == cur[0]:
        result.test = "Constant in both datasets"
        result.statistic = 0.0
        result.effect_size = 0.0
        result.psi = 0.0
        result.p_value = 1.0
        result.details = {"constant_value": clean_number(ref[0])}
        result.explanation = f"Feature is constant ({_fmt(float(ref[0]))}) in both datasets."
        return result

    ks = stats.ks_2samp(ref, cur, alternative="two-sided", method="auto")
    result.statistic = float(ks.statistic)
    result.effect_size = float(ks.statistic)
    result.p_value = float(ks.pvalue)
    result.psi = psi_numeric(ref, cur, config.psi_bins)

    ref_std = float(ref.std(ddof=1)) if ref.size > 1 else 0.0
    wasserstein = float(stats.wasserstein_distance(ref, cur))
    ref_mean, cur_mean = float(ref.mean()), float(cur.mean())
    result.details = {
        "wasserstein_distance": wasserstein,
        "wasserstein_normalized": wasserstein / ref_std if ref_std > 0 else None,
        "mean_difference": cur_mean - ref_mean,
        "std_ratio": float(cur.std(ddof=1)) / ref_std if ref_std > 0 else None,
        "reference_constant": bool(ref_constant),
        "current_constant": bool(cur_constant),
        "discrete_values": bool(np.unique(ref).size <= 20),
    }
    return result


def _numeric_explanation(r: FeatureDriftResult) -> str:
    ref, cur, d = r.reference_stats, r.current_stats, r.details
    parts = []
    if ref.get("mean") is not None and cur.get("mean") is not None:
        change = ""
        delta = cur["mean"] - ref["mean"]
        ref_std = ref.get("std") or 0
        # A percentage of a near-zero mean is meaningless; express it in reference SDs instead.
        if ref["mean"] != 0 and abs(ref["mean"]) >= 0.5 * ref_std:
            change = f" ({delta / abs(ref['mean']):+.1%})"
        elif ref_std > 0:
            change = f" ({delta / ref_std:+.2f} SD)"
        parts.append(f"Mean {_fmt(ref['mean'])} -> {_fmt(cur['mean'])}{change}")
    if d.get("std_ratio") is not None:
        parts.append(f"std ratio {d['std_ratio']:.2f}")
    parts.append(f"KS D = {r.statistic:.3f}, PSI = {r.psi:.3f}")
    text = "; ".join(parts) + "."
    if d.get("reference_constant") or d.get("current_constant"):
        text += " One dataset is constant for this feature."
    if d.get("discrete_values"):
        text += " Few distinct values: KS p-values are conservative for discrete data."
    return text


# --------------------------------------------------------------------------
# Categorical features
# --------------------------------------------------------------------------


def _cap_categories(
    ref_counts: pd.Series, cur_counts: pd.Series, max_categories: int
) -> tuple[pd.Series, pd.Series, bool]:
    """Keep the most frequent reference categories; group the rest."""
    cats = ref_counts.index.union(cur_counts.index)
    if cats.size <= max_categories:
        return ref_counts, cur_counts, False
    keep = list(ref_counts.sort_values(ascending=False).index[: max_categories - 1])

    def regroup(counts: pd.Series) -> pd.Series:
        kept = counts.reindex(keep, fill_value=0)
        other = counts.drop(labels=[c for c in keep if c in counts.index]).sum()
        return pd.concat([kept, pd.Series({OTHER_BUCKET: other})])

    return regroup(ref_counts), regroup(cur_counts), True


def _merge_sparse(table: np.ndarray, labels: list[str], min_expected: float):
    """Merge categories whose expected count is below ``min_expected``."""
    expected = stats.contingency.expected_freq(table)
    sparse = (expected < min_expected).any(axis=0)
    if not sparse.any():
        return table, labels, []
    merged_labels = [lab for lab, s in zip(labels, sparse) if s]
    kept = table[:, ~sparse]
    rare = table[:, sparse].sum(axis=1, keepdims=True)
    new_table = np.hstack([kept, rare])
    new_labels = [lab for lab, s in zip(labels, sparse) if not s] + [RARE_BUCKET]
    return new_table, new_labels, merged_labels


def _chi2_statistic(table: np.ndarray) -> float:
    expected = stats.contingency.expected_freq(table)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (table - expected) ** 2 / expected, 0.0)
    return float(terms.sum())


def permutation_chi2_test(
    ref_codes: np.ndarray, cur_codes: np.ndarray, n_categories: int, iterations: int, seed: int
) -> tuple[float, float]:
    """Monte Carlo permutation test using the chi-square statistic.

    Valid without the large-expected-count assumption of the asymptotic
    chi-square distribution. The p-value uses the (b + 1) / (B + 1) estimator,
    so it is never exactly zero.
    """
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([ref_codes, cur_codes])
    n_ref = ref_codes.size

    def stat(codes: np.ndarray) -> float:
        table = np.vstack(
            [
                np.bincount(codes[:n_ref], minlength=n_categories),
                np.bincount(codes[n_ref:], minlength=n_categories),
            ]
        )
        return _chi2_statistic(table)

    observed = stat(pooled)
    exceed = sum(stat(rng.permutation(pooled)) >= observed - 1e-12 for _ in range(iterations))
    return observed, (exceed + 1) / (iterations + 1)


def analyze_categorical(
    feature: str, reference: pd.Series, current: pd.Series, config: AnalysisConfig
) -> FeatureDriftResult:
    ref = reference.dropna().astype(str)
    cur = current.dropna().astype(str)
    result = FeatureDriftResult(
        feature=feature,
        feature_type=CATEGORICAL,
        status=INSUFFICIENT,
        test="Chi-square test of homogeneity",
        n_reference=int(ref.size),
        n_current=int(cur.size),
        statistic_name="Chi-square",
        effect_size_name="Cramer's V",
        reference_stats=categorical_summary(reference),
        current_stats=categorical_summary(current),
    )
    ref_counts_full = ref.value_counts()
    cur_counts_full = cur.value_counts()
    unseen = sorted(set(cur_counts_full.index) - set(ref_counts_full.index))
    disappeared = sorted(set(ref_counts_full.index) - set(cur_counts_full.index))
    result.details = {
        "unseen_categories": unseen[:20],
        "n_unseen_categories": len(unseen),
        "current_share_unseen": float(cur_counts_full[unseen].sum() / cur.size) if unseen and cur.size else 0.0,
        "disappeared_categories": disappeared[:20],
        "n_disappeared_categories": len(disappeared),
    }

    if ref.size < config.min_samples or cur.size < config.min_samples:
        result.test = "Not tested"
        result.explanation = (
            f"Only {ref.size} reference and {cur.size} current non-missing values; "
            f"at least {config.min_samples} are required in each dataset."
        )
        return result

    ref_counts, cur_counts, capped = _cap_categories(
        ref_counts_full, cur_counts_full, config.max_categories
    )
    labels = sorted(ref_counts.index.union(cur_counts.index), key=str)
    table = np.vstack(
        [
            ref_counts.reindex(labels, fill_value=0).to_numpy(),
            cur_counts.reindex(labels, fill_value=0).to_numpy(),
        ]
    ).astype(float)
    result.details["categories_capped"] = capped

    ref_p = table[0] / table[0].sum()
    cur_p = table[1] / table[1].sum()
    result.psi = _psi_from_proportions(ref_p, cur_p)
    result.details["total_variation_distance"] = float(0.5 * np.abs(ref_p - cur_p).sum())
    diffs = cur_p - ref_p
    top = int(np.argmax(np.abs(diffs)))
    result.details["largest_change"] = {
        "category": labels[top],
        "reference_proportion": float(ref_p[top]),
        "current_proportion": float(cur_p[top]),
    }
    result.details["proportions"] = {
        lab: {"reference": float(rp), "current": float(cp)}
        for lab, rp, cp in zip(labels, ref_p, cur_p)
    }

    if len(labels) == 1:
        result.test = "Constant in both datasets"
        result.statistic = 0.0
        result.effect_size = 0.0
        result.p_value = 1.0
        result.explanation = f"Feature has a single category ('{labels[0]}') in both datasets."
        return result

    test_table, test_labels, merged = _merge_sparse(table, labels, config.chi2_min_expected)
    result.details["merged_sparse_categories"] = merged[:20]
    expected = stats.contingency.expected_freq(test_table) if test_table.shape[1] >= 2 else None
    chi2_valid = expected is not None and bool((expected >= config.chi2_min_expected).all())

    n_total = float(table.sum())
    if chi2_valid:
        chi2, p, dof, _ = stats.chi2_contingency(test_table, correction=False)
        result.statistic = float(chi2)
        result.p_value = float(p)
        result.details["degrees_of_freedom"] = int(dof)
        if merged:
            result.test = "Chi-square (sparse categories merged)"
        stat_for_v = float(chi2)
    else:
        codes = {lab: i for i, lab in enumerate(labels)}
        ref_mapped = ref.map(lambda v: v if v in codes else OTHER_BUCKET)
        cur_mapped = cur.map(lambda v: v if v in codes else OTHER_BUCKET)
        ref_codes = ref_mapped.map(codes).to_numpy(dtype=int)
        cur_codes = cur_mapped.map(codes).to_numpy(dtype=int)
        observed, p = permutation_chi2_test(
            ref_codes, cur_codes, len(labels), config.permutation_iterations, config.random_seed
        )
        result.test = f"Permutation chi-square ({config.permutation_iterations} resamples)"
        result.statistic = observed
        result.p_value = float(p)
        stat_for_v = observed
        result.details["reason_for_permutation"] = (
            f"Expected counts below {config.chi2_min_expected:g} even after merging "
            "sparse categories, so the asymptotic chi-square approximation is unreliable."
        )

    # For a 2 x K table, min(r, c) - 1 = 1, so V = sqrt(chi2 / N).
    result.effect_size = float(np.sqrt(stat_for_v / n_total)) if n_total > 0 else 0.0
    return result


def _categorical_explanation(r: FeatureDriftResult) -> str:
    d = r.details
    parts = []
    lc = d.get("largest_change")
    if lc:
        delta = (lc["current_proportion"] - lc["reference_proportion"]) * 100
        parts.append(
            f"Largest change: '{lc['category']}' {lc['reference_proportion']:.1%} -> "
            f"{lc['current_proportion']:.1%} ({delta:+.1f} pp)"
        )
    if r.effect_size is not None:
        parts.append(f"Cramer's V = {r.effect_size:.3f}, PSI = {r.psi:.3f}")
    text = "; ".join(parts) + "."
    if d.get("n_unseen_categories"):
        names = ", ".join(f"'{c}'" for c in d["unseen_categories"][:3])
        text += (
            f" {d['n_unseen_categories']} category(ies) not seen in reference ({names}), "
            f"covering {d['current_share_unseen']:.1%} of current rows."
        )
    if d.get("merged_sparse_categories"):
        text += f" {len(d['merged_sparse_categories'])} sparse category(ies) were merged for the test."
    return text


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _effect_threshold(r: FeatureDriftResult, config: AnalysisConfig) -> float:
    return config.ks_effect_threshold if r.feature_type == NUMERICAL else config.cramers_v_threshold


def finalize_results(results: list[FeatureDriftResult], config: AnalysisConfig) -> list[FeatureDriftResult]:
    """Apply the multiple-testing correction and assign final statuses."""
    tested = [r for r in results if r.p_value is not None]
    adjusted = adjust_p_values([r.p_value for r in tested], config.correction)
    for r, adj in zip(tested, adjusted):
        r.adjusted_p_value = float(adj)
        r.statistically_significant = adj < config.alpha
        effect_ok = (r.effect_size or 0.0) >= _effect_threshold(r, config)
        psi_ok = (r.psi or 0.0) >= config.psi_threshold
        r.practically_significant = bool(effect_ok or psi_ok)
        drift = r.statistically_significant and (
            r.practically_significant or not config.require_practical_effect
        )
        r.status = DRIFT if drift else NO_DRIFT

        if r.test.startswith("Constant"):
            continue
        base = _numeric_explanation(r) if r.feature_type == NUMERICAL else _categorical_explanation(r)
        if r.status == DRIFT:
            note = "Distribution change is statistically significant and exceeds the effect-size threshold."
        elif r.statistically_significant:
            note = (
                "Statistically significant but below the practical effect-size thresholds; "
                "likely a small change detected because of large sample sizes."
            )
        elif r.practically_significant:
            note = (
                "Effect size exceeds a threshold but the change is not statistically "
                "significant after correction; may be noise from small samples."
            )
        else:
            note = "No statistically significant change after correction."
        r.explanation = f"{base} {note}"
    return results


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: dict[str, str],
    config: AnalysisConfig,
) -> list[FeatureDriftResult]:
    """Run drift tests for ``features`` (mapping of column -> feature type)."""
    results = []
    for col, ftype in features.items():
        if ftype == NUMERICAL:
            results.append(analyze_numerical(col, reference[col], current[col], config))
        else:
            results.append(analyze_categorical(col, reference[col], current[col], config))
    return finalize_results(results, config)
