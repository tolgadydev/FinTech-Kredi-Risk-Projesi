import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    roc_auc = roc_auc_score(y_true, y_prob)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_prob)
    return {
        "roc_auc": float(roc_auc),
        "gini": float(2 * roc_auc - 1),
        "ks": float(np.max(true_positive_rate - false_positive_rate)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }