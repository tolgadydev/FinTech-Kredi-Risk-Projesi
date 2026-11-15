import numpy as np
import pandas as pd
import pytest
from credit_decision import thresholds_from

from ml.fairness import age_groups, band_shares, fairness_report


def test_band_shares():
    shares = band_shares(np.array([0.05, 0.20, 0.30, 0.10]), thresholds_from(0.15, 0.45))
    assert shares == pytest.approx({"approve": 0.5, "manual_review": 0.25, "reject": 0.25})


def test_fairness_report_rates_per_group():
    report = fairness_report(
        y=np.array([0, 1, 0, 1]),
        bands=np.array(["approve", "reject", "reject", "approve"]),
        sensitive=pd.Series(["F", "F", "M", "M"]),
    )
    assert set(report.index) == {"F", "M"}
    assert report.loc["F", "approval_rate"] == 0.5
    assert report.loc["F", "tpr"] == 1.0 and report.loc["F", "fpr"] == 0.0
    assert report.loc["M", "tpr"] == 0.0 and report.loc["M", "fpr"] == 1.0


def test_age_groups():
    days = pd.Series([-365.25 * 25, -365.25 * 30, -365.25 * 49.9, -365.25 * 60])
    assert list(age_groups(days)) == ["<30", "30-50", "30-50", "50+"]