from pathlib import Path

import pytest

from credit_decision import assign_band, load_thresholds, thresholds_from

CONFIG = Path(__file__).parents[2] / "config" / "decision.yaml"


def test_thresholds_from_spec():
    t = thresholds_from(0.15, 0.45)
    assert t.reject == pytest.approx(0.25)
    assert t.approve == pytest.approx(0.125)


def test_band_boundaries():
    t = thresholds_from(0.15, 0.45)
    assert assign_band(t.approve - 1e-6, t) == "approve"
    assert assign_band(t.approve, t) == "manual_review"
    assert assign_band(t.reject - 1e-6, t) == "manual_review"
    assert assign_band(t.reject, t) == "reject"


def test_config_file_matches_spec():
    assert load_thresholds(CONFIG) == thresholds_from(0.15, 0.45)


@pytest.mark.parametrize("margin, lgd", [(0, 0.45), (0.15, 0), (-0.1, 0.45)])
def test_invalid_assumptions_are_rejected(margin, lgd):
    with pytest.raises(ValueError):
        thresholds_from(margin, lgd)


@pytest.mark.parametrize("pd_value", [-0.01, 1.01])
def test_probability_outside_0_1_is_rejected(pd_value):
    with pytest.raises(ValueError):
        assign_band(pd_value, thresholds_from(0.15, 0.45))