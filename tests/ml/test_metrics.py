import numpy as np
import pytest

from ml.metrics import evaluate


def test_perfect_model_gets_full_marks():
    m = evaluate(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["roc_auc"] == 1 and m["gini"] == 1 and m["ks"] == 1


def test_reversed_model_gets_negative_gini():
    m = evaluate(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1]))
    assert m["roc_auc"] == 0 and m["gini"] == -1


def test_brier_is_mean_squared_error():
    m = evaluate(np.array([0, 1]), np.array([0.2, 0.6]))
    assert m["brier"] == pytest.approx(0.1)


def test_returns_exact_metric_names():
    m = evaluate(np.array([0, 1, 0, 1]), np.array([0.3, 0.7, 0.4, 0.6]))
    assert set(m) == {"roc_auc", "gini", "ks", "pr_auc", "brier"}