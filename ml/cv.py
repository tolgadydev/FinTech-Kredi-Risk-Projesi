from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ml.metrics import evaluate


@dataclass
class CVResult:
    oof: np.ndarray
    fold_metrics: list[dict[str, float]]


def cross_val_oof(
    make_model: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
) -> CVResult:
    oof = np.full(len(y), np.nan)
    fold_metrics = []
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, val_idx in folds.split(X, y):
        model = make_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = model.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = prob
        fold_metrics.append(evaluate(y.iloc[val_idx].to_numpy(), prob))

    return CVResult(oof=oof, fold_metrics=fold_metrics)