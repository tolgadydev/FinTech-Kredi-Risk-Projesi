import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold


def calibrated_brier(y, prob, n_splits: int = 5, seed: int = 42) -> float:
    y = np.asarray(y)
    prob = np.asarray(prob, dtype=float)
    calibrated = np.empty_like(prob)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fit_idx, eval_idx in folds.split(prob.reshape(-1, 1), y):
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        calibrator.fit(prob[fit_idx], y[fit_idx])
        calibrated[eval_idx] = calibrator.predict(prob[eval_idx])

    return float(brier_score_loss(y, calibrated))


def select_best(results: list[dict], tol: float = 0.002) -> dict:
    top = max(r["pr_auc"] for r in results)
    contenders = [r for r in results if top - r["pr_auc"] <= tol]
    return min(contenders, key=lambda r: r["brier_calibrated"])