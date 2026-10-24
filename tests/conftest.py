import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from driftguard.config import AnalysisConfig  # noqa: E402


@pytest.fixture
def config() -> AnalysisConfig:
    return AnalysisConfig(permutation_iterations=500)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def small_frames(rng):
    n = 800
    ref = pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "cat": rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2]),
    })
    cur = pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "cat": rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2]),
    })
    return ref, cur
