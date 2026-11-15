import numpy as np
import optuna
import pandas as pd
from credit_features.frame import FeatureSpec

from ml.cv import cross_val_oof
from ml.models import make_model

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _catboost_space(trial: optuna.Trial) -> dict:
    return {
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "iterations": trial.suggest_int("iterations", 300, 1500, step=100),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
    }


SEARCH_SPACES = {"catboost": _catboost_space}


def tune(
    name: str,
    arm: str,
    X: pd.DataFrame,
    y: pd.Series,
    spec: FeatureSpec,
    n_trials: int = 30,
    n_splits: int = 3,
    seed: int = 42,
) -> tuple[dict, float]:
    if name not in SEARCH_SPACES:
        raise ValueError(f"{name} için arama alanı yok. Seçenekler: {list(SEARCH_SPACES)}")

    def objective(trial: optuna.Trial) -> float:
        params = SEARCH_SPACES[name](trial)
        cv = cross_val_oof(lambda: make_model(name, arm, spec, params), X, y, n_splits, seed)
        return float(np.mean([f["pr_auc"] for f in cv.fold_metrics]))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, float(study.best_value)