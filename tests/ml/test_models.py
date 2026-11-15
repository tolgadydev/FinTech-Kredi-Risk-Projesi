import numpy as np
import pandas as pd
import pytest
from credit_features.frame import align_frame, fit_spec
from imblearn.over_sampling import SMOTENC

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


@pytest.mark.parametrize("name", ["lightgbm", "catboost"])
@pytest.mark.parametrize("arm", ["class_weight", "smotenc"])
def test_imbalance_arms_give_probabilities(data, name, arm):
    X, y, spec = data
    prob = make_model(name, arm, spec).fit(X, y).predict_proba(X)[:, 1]
    assert len(prob) == len(X)
    assert ((prob >= 0) & (prob <= 1)).all()


def test_smote_runs_once_on_fit_never_on_predict(data, mocker):
    X, y, spec = data
    spy = mocker.spy(SMOTENC, "fit_resample")
    pipe = make_model("lightgbm", "smotenc", spec).fit(X, y)
    pipe.predict_proba(X)
    assert spy.call_count == 1


def test_smote_imputer_learns_only_from_training_rows(data):
    X, y, spec = data
    train, val = X.iloc[:500], X.iloc[500:].copy()
    val["b"] = 1e12
    pipe = make_model("lightgbm", "smotenc", spec).fit(train, y.iloc[:500])
    imputer = pipe.named_steps["prep"].named_transformers_["num"]
    assert imputer.statistics_[spec.numeric.index("b")] < 1e12
    assert len(pipe.predict_proba(val)) == len(val)


def test_unknown_arm_is_rejected(data):
    with pytest.raises(ValueError):
        make_model("lightgbm", "oversample", data[2])