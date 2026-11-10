import numpy as np
import pandas as pd
import pytest

from credit_features.bureau import G3_COLUMNS, build_g3


@pytest.fixture
def bureau():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "SK_ID_BUREAU": [10, 11, 20],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
            "AMT_CREDIT_MAX_OVERDUE": [0.0, 500.0, np.nan],
        }
    )


@pytest.fixture
def bureau_balance():
    return pd.DataFrame(
        {
            "SK_ID_BUREAU": [10, 10, 11, 11, 20, 99],
            "MONTHS_BALANCE": [-1, -2, -1, -2, -1, -1],
            "STATUS": ["0", "1", "C", "X", "0", "5"],
        }
    )


def test_bureau_summary(app_df, bureau, bureau_balance):
    g3 = build_g3(app_df, bureau, bureau_balance)
    assert g3.loc[1, "g3_bureau_count"] == 2
    assert g3.loc[1, "g3_bureau_active_share"] == pytest.approx(0.5)
    assert g3.loc[1, "g3_bureau_overdue_max"] == 500.0


def test_balance_reaches_customer_through_bureau_id(app_df, bureau, bureau_balance):
    assert build_g3(app_df, bureau, bureau_balance).loc[1, "g3_bb_dpd_share"] == pytest.approx(0.25)


def test_customer_without_bureau_is_kept(app_df, bureau, bureau_balance):
    g3 = build_g3(app_df, bureau, bureau_balance)
    assert np.isnan(g3.loc[3, "g3_bureau_count"])
    assert g3.loc[3, "EXT_SOURCE_2"] == 0.4


def test_unknown_bureau_id_is_ignored(app_df, bureau, bureau_balance):
    assert 99 not in build_g3(app_df, bureau, bureau_balance).index


def test_columns_are_exact(app_df, bureau, bureau_balance):
    assert list(build_g3(app_df, bureau, bureau_balance).columns) == G3_COLUMNS