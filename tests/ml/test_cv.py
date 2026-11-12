import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml.cv import cross_val_oof


@pytest.fixture
def xy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=400) > 1).astype(int))
    return X, y


class MemoryModel:
    """Eğitimde gördüğü satırlara 1, görmediklerine 0 verir."""

    def fit(self, X, y):
        self.seen = set(X.index)
        return self

    def predict_proba(self, X):
        p = np.array([1.0 if i in self.seen else 0.0 for i in X.index])
        return np.column_stack([1 - p, p])


def test_oof_covers_every_row(xy):
    result = cross_val_oof(LogisticRegression, *xy, n_splits=5, seed=42)
    assert len(result.oof) == len(xy[1]) and not np.isnan(result.oof).any()
    assert len(result.fold_metrics) == 5


def test_no_row_is_predicted_by_a_model_that_saw_it(xy):
    result = cross_val_oof(MemoryModel, *xy, n_splits=5, seed=42)
    assert (result.oof == 0).all()


def test_oof_is_reproducible(xy):
    first = cross_val_oof(LogisticRegression, *xy, n_splits=5, seed=42).oof
    second = cross_val_oof(LogisticRegression, *xy, n_splits=5, seed=42).oof
    assert np.array_equal(first, second)


def test_learns_real_signal(xy):
    result = cross_val_oof(LogisticRegression, *xy, n_splits=5, seed=42)
    assert np.mean([m["roc_auc"] for m in result.fold_metrics]) > 0.8