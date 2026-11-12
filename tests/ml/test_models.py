import numpy as np
import pandas as pd
import pytest
from credit_features.frame import align_frame, fit_spec

from ml.models import make_baseline, make_model


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 600
    raw = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": np.where(rng.random(n) < 0.3, np.nan, rng.normal(size=n)),
            "job": rng.choice(["x", "y", "z", None], size=n),
        }
    )
    signal = raw["a"] + (raw["job"] == "x") + rng.normal(scale=0.5, size=n)
    y = pd.Series((signal > 1).astype(int))
    spec = fit_spec(raw)
    return align_frame(raw, spec), y, spec


@pytest.mark.parametrize("name", ["lightgbm", "catboost"])
def test_boosting_handles_missing_and_categories(data, name):
    X, y, spec = data
    prob = make_model(name, "none", spec).fit(X, y).predict_proba(X)[:, 1]
    assert ((prob >= 0) & (prob <= 1)).all()


def test_baseline_handles_missing_and_categories(data):
    X, y, spec = data
    prob = make_baseline(spec).fit(X, y).predict_proba(X)[:, 1]
    assert ((prob >= 0) & (prob <= 1)).all()


@pytest.mark.parametrize("name", ["lightgbm", "catboost"])
def test_same_seed_same_predictions(data, name):
    X, y, spec = data
    first = make_model(name, "none", spec).fit(X, y).predict_proba(X)[:, 1]
    second = make_model(name, "none", spec).fit(X, y).predict_proba(X)[:, 1]
    assert np.allclose(first, second)


def test_unknown_model_is_rejected(data):
    with pytest.raises(ValueError):
        make_model("xgboost", "none", data[2])


def test_imbalance_arms_come_later(data):
    with pytest.raises(NotImplementedError):
        make_model("lightgbm", "class_weight", data[2])