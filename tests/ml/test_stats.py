import numpy as np
import pytest

from ml.stats import bootstrap_auc_diff


@pytest.fixture
def predictions():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=1000)
    strong = y + rng.normal(0, 0.5, size=1000)
    weak = y + rng.normal(0, 2.0, size=1000)
    return y, strong, weak


def test_detects_clearly_better_model(predictions):
    y, strong, weak = predictions
    low, high = bootstrap_auc_diff(y, strong, weak, n=500, seed=42)
    assert 0 < low < high


def test_identical_models_include_zero(predictions):
    y, strong, _ = predictions
    low, high = bootstrap_auc_diff(y, strong, strong, n=500, seed=42)
    assert low <= 0 <= high


def test_is_reproducible(predictions):
    first = bootstrap_auc_diff(*predictions, n=200, seed=42)
    second = bootstrap_auc_diff(*predictions, n=200, seed=42)
    assert first == second