import numpy as np
import pandas as pd
import pytest

from ml.split import make_holdout


@pytest.fixture
def labelled():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(1000, 2000),
            "TARGET": (rng.random(1000) < 0.08).astype(int),
        }
    )


def test_holdout_is_disjoint_and_20_percent(labelled):
    dev, test = make_holdout(labelled)
    assert not set(dev) & set(test)
    assert len(dev) + len(test) == len(labelled)
    assert len(test) / len(labelled) == pytest.approx(0.2, abs=0.01)


def test_holdout_keeps_target_rate(labelled):
    _, test = make_holdout(labelled)
    rate = labelled.set_index("SK_ID_CURR").loc[test, "TARGET"].mean()
    assert rate == pytest.approx(labelled["TARGET"].mean(), abs=0.01)


def test_holdout_is_reproducible(labelled):
    first = make_holdout(labelled, seed=42)[1]
    second = make_holdout(labelled, seed=42)[1]
    assert (first == second).all()