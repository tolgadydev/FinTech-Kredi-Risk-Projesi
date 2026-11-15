import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_auc_diff(
    y, prob_a, prob_b, n: int = 2000, seed: int = 42, level: float = 0.95
) -> tuple[float, float]:
    y = np.asarray(y)
    prob_a = np.asarray(prob_a)
    prob_b = np.asarray(prob_b)
    rng = np.random.default_rng(seed)

    diffs = []
    while len(diffs) < n:
        idx = rng.integers(0, len(y), size=len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], prob_a[idx]) - roc_auc_score(y[idx], prob_b[idx]))

    tail = (1 - level) / 2
    low, high = np.quantile(diffs, [tail, 1 - tail])
    return float(low), float(high)