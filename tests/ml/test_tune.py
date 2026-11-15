import optuna
import pytest

from ml import tune as tune_module
from ml.models import make_model
from ml.tune import tune


@pytest.fixture
def small_space(monkeypatch):
    def space(trial):
        return {
            "depth": trial.suggest_int("depth", 4, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
            "iterations": trial.suggest_int("iterations", 20, 60, step=20),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        }

    monkeypatch.setitem(tune_module.SEARCH_SPACES, "catboost", space)


def test_tune_returns_params_and_score(data, small_space):
    X, y, spec = data
    params, score = tune("catboost", "none", X, y, spec, n_trials=2, n_splits=3, seed=42)
    assert set(params) == {"depth", "learning_rate", "iterations", "l2_leaf_reg"}
    assert 0 < score <= 1


def test_tune_is_reproducible(data, small_space):
    X, y, spec = data
    first, _ = tune("catboost", "none", X, y, spec, n_trials=2, n_splits=3, seed=42)
    second, _ = tune("catboost", "none", X, y, spec, n_trials=2, n_splits=3, seed=42)
    assert first == second


def test_tuned_params_work_in_make_model(data, small_space):
    X, y, spec = data
    params, _ = tune("catboost", "none", X, y, spec, n_trials=2, n_splits=3, seed=42)
    prob = make_model("catboost", "none", spec, params).fit(X, y).predict_proba(X)[:, 1]
    assert len(prob) == len(X)


def test_catboost_space_respects_bounds():
    seen = []
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    study.optimize(lambda t: seen.append(tune_module._catboost_space(t)) or 0.0, n_trials=50)
    for p in seen:
        assert 4 <= p["depth"] <= 8
        assert 0.02 <= p["learning_rate"] <= 0.15
        assert 300 <= p["iterations"] <= 1500 and p["iterations"] % 100 == 0
        assert 1.0 <= p["l2_leaf_reg"] <= 10.0


def test_only_catboost_has_a_search_space(data):
    X, y, spec = data
    with pytest.raises(ValueError):
        tune("lightgbm", "none", X, y, spec, n_trials=1)