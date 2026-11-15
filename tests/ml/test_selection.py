import numpy as np
import pytest

from ml.metrics import evaluate
from ml.selection import calibrated_brier, select_best


def test_select_best_takes_highest_pr_auc():
    a = {"id": "a", "pr_auc": 0.300, "brier_calibrated": 0.070}
    c = {"id": "c", "pr_auc": 0.250, "brier_calibrated": 0.010}
    assert select_best([a, c], tol=0.002)["id"] == "a"


def test_select_best_breaks_near_tie_with_brier():
    a = {"id": "a", "pr_auc": 0.300, "brier_calibrated": 0.070}
    b = {"id": "b", "pr_auc": 0.299, "brier_calibrated": 0.060}
    c = {"id": "c", "pr_auc": 0.250, "brier_calibrated": 0.010}
    assert select_best([a, b, c], tol=0.002)["id"] == "b"


@pytest.fixture
def overconfident():
    rng = np.random.default_rng(0)
    true_prob = rng.uniform(0, 0.3, size=5000)
    y = (rng.random(5000) < true_prob).astype(int)
    return y, np.clip(true_prob * 3, 0, 1)


def test_calibration_fixes_overconfident_predictions(overconfident):
    y, prob = overconfident
    assert calibrated_brier(y, prob) < evaluate(y, prob)["brier"]


def test_calibrated_brier_is_reproducible(overconfident):
    assert calibrated_brier(*overconfident, seed=42) == calibrated_brier(*overconfident, seed=42)