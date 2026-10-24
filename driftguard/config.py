"""Analysis configuration and threshold defaults.

Threshold rationale (documented in README "Drift detection methods"):

* ``alpha = 0.05`` - conventional significance level, applied to p-values
  *after* multiple-testing correction.
* ``psi_threshold = 0.2`` - common industry rule of thumb (PSI < 0.1 little
  change, 0.1-0.2 moderate, > 0.2 significant). It is a heuristic, not a
  statistically derived cut-off.
* ``ks_effect_threshold = 0.1`` - a KS statistic of 0.1 means the empirical
  CDFs differ by at least 10 percentage points somewhere.
* ``cramers_v_threshold = 0.1`` - Cohen's conventional "small" effect for a
  2 x K table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CORRECTION_METHODS: tuple[str, ...] = ("fdr_bh", "bonferroni", "none")

CORRECTION_LABELS: dict[str, str] = {
    "fdr_bh": "Benjamini-Hochberg (FDR)",
    "bonferroni": "Bonferroni (FWER)",
    "none": "None (raw p-values)",
}


@dataclass(frozen=True)
class AnalysisConfig:
    alpha: float = 0.05
    correction: str = "fdr_bh"
    psi_threshold: float = 0.2
    psi_bins: int = 10
    ks_effect_threshold: float = 0.1
    cramers_v_threshold: float = 0.1
    require_practical_effect: bool = True
    min_samples: int = 30
    chi2_min_expected: float = 5.0
    max_categories: int = 30
    permutation_iterations: int = 2000
    random_seed: int = 42
    missing_increase_threshold: float = 0.05
    range_violation_threshold: float = 0.01
    identifier_unique_ratio: float = 0.95

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1 (exclusive).")
        if self.correction not in CORRECTION_METHODS:
            raise ValueError(f"correction must be one of {CORRECTION_METHODS}.")
        if self.psi_threshold <= 0:
            raise ValueError("psi_threshold must be positive.")
        if not 2 <= self.psi_bins <= 50:
            raise ValueError("psi_bins must be between 2 and 50.")
        if not 0 < self.ks_effect_threshold < 1:
            raise ValueError("ks_effect_threshold must be between 0 and 1.")
        if not 0 < self.cramers_v_threshold < 1:
            raise ValueError("cramers_v_threshold must be between 0 and 1.")
        if self.min_samples < 2:
            raise ValueError("min_samples must be at least 2.")
        if self.chi2_min_expected <= 0:
            raise ValueError("chi2_min_expected must be positive.")
        if self.max_categories < 2:
            raise ValueError("max_categories must be at least 2.")
        if self.permutation_iterations < 100:
            raise ValueError("permutation_iterations must be at least 100.")
        if not 0 <= self.missing_increase_threshold <= 1:
            raise ValueError("missing_increase_threshold must be between 0 and 1.")
        if not 0 <= self.range_violation_threshold <= 1:
            raise ValueError("range_violation_threshold must be between 0 and 1.")
        if not 0 < self.identifier_unique_ratio <= 1:
            raise ValueError("identifier_unique_ratio must be in (0, 1].")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
