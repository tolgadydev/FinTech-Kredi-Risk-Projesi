import numpy as np
import pandas as pd
from credit_decision import Thresholds, assign_band
from fairlearn.metrics import MetricFrame

BANDS = ("approve", "manual_review", "reject")
DAYS_PER_YEAR = 365.25


def assign_bands(probs, thresholds: Thresholds) -> np.ndarray:
    return np.array([assign_band(float(p), thresholds) for p in probs])


def band_shares(probs, thresholds: Thresholds) -> dict[str, float]:
    bands = assign_bands(probs, thresholds)
    return {band: float(np.mean(bands == band)) for band in BANDS}


def _rate(values) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def fairness_report(y, bands, sensitive) -> pd.DataFrame:
    y = np.asarray(y)
    bands = np.asarray(bands)
    groups = np.asarray(sensitive)
    rejected = (bands == "reject").astype(int)

    frame = MetricFrame(
        metrics={
            "count": lambda y_true, y_pred: len(y_true),
            "default_rate": lambda y_true, y_pred: _rate(y_true),
            "reject_rate": lambda y_true, y_pred: _rate(y_pred),
            "tpr": lambda y_true, y_pred: _rate(y_pred[y_true == 1]),
            "fpr": lambda y_true, y_pred: _rate(y_pred[y_true == 0]),
        },
        y_true=y,
        y_pred=rejected,
        sensitive_features=groups,
    )
    report = frame.by_group
    report["approval_rate"] = pd.Series(bands == "approve").groupby(groups).mean()
    return report.rename_axis("group")


def age_groups(days_birth) -> pd.Series:
    years = -pd.Series(days_birth) / DAYS_PER_YEAR
    labels = ["<30", "30-50", "50+"]
    return pd.cut(years, bins=[-np.inf, 30, 50, np.inf], right=False, labels=labels).astype(str)