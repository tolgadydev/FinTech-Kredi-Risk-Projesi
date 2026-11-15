import numpy as np
import pandas as pd
import pytest

from ml.drift import psi, psi_table


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_identical_data_has_zero_psi(rng):
    values = pd.Series(rng.normal(size=1000))
    assert psi(values, values) == pytest.approx(0.0)


def test_shifted_numbers_have_large_psi(rng):
    before = pd.Series(rng.normal(size=5000))
    after = pd.Series(rng.normal(loc=1.0, size=5000))
    assert psi(before, after) > 0.25


def test_new_missing_values_are_detected(rng):
    before = pd.Series(rng.normal(size=1000))
    after = before.copy()
    after.iloc[:500] = np.nan
    assert psi(before, after) > 0.25


def test_category_shift_is_detected():
    before = pd.Series(["x"] * 90 + ["y"] * 10)
    after = pd.Series(["x"] * 50 + ["y"] * 50)
    assert psi(before, after) > 0.25


def test_psi_table_sorts_and_labels(rng):
    reference = pd.DataFrame({"stable": rng.normal(size=2000), "moved": rng.normal(size=2000)})
    current = pd.DataFrame(
        {"stable": rng.normal(size=2000), "moved": rng.normal(loc=1.0, size=2000)}
    )
    table = psi_table(reference, current)
    assert list(table.index) == ["moved", "stable"]
    assert table.loc["moved", "level"] == "önemli kayma"
    assert table.loc["stable", "level"] == "stabil"