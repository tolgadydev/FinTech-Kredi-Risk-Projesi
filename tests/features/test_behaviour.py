import numpy as np
import pandas as pd
import pytest

from credit_features.behaviour import G2_COLUMNS, build_g2


@pytest.fixture
def tables():
    installments = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2, 2],
            "DAYS_INSTALMENT": [-100.0, -500.0, -50.0, -60.0],
            "DAYS_ENTRY_PAYMENT": [-90.0, -500.0, -50.0, -60.0],
            "AMT_INSTALMENT": [100.0, 100.0, 0.0, 200.0],
            "AMT_PAYMENT": [100.0, 50.0, 0.0, 200.0],
        }
    )
    pos = pd.DataFrame({"SK_ID_CURR": [1, 3], "SK_DPD": [0.0, 7.0]})
    credit_card = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "AMT_BALANCE": [50.0, 10.0],
            "AMT_CREDIT_LIMIT_ACTUAL": [100.0, 0.0],
        }
    )
    previous = pd.DataFrame(
        {"SK_ID_CURR": [1, 1, 1], "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"]}
    )
    return installments, pos, credit_card, previous


def test_late_payment_stats(tables):
    g2 = build_g2(*tables)
    assert g2.loc[1, "g2_inst_late_share"] == pytest.approx(0.5)
    assert g2.loc[1, "g2_inst_late_days_mean"] == pytest.approx(5.0)


def test_recent_window_only_counts_last_year(tables):
    assert build_g2(*tables).loc[1, "g2_inst_late_share_last12"] == pytest.approx(1.0)


def test_underpayment_ratio(tables):
    assert build_g2(*tables).loc[1, "g2_inst_underpay_ratio_mean"] == pytest.approx(0.25)


def test_underpayment_ignores_zero_instalment(tables):
    assert np.isfinite(build_g2(*tables).loc[2, "g2_inst_underpay_ratio_mean"])


def test_zero_card_limit_gives_missing_not_inf(tables):
    assert np.isnan(build_g2(*tables).loc[2, "g2_cc_utilization_mean"])


def test_refused_share(tables):
    assert build_g2(*tables).loc[1, "g2_prev_refused_share"] == pytest.approx(1 / 3)


def test_customer_with_only_pos_row_is_present(tables):
    g2 = build_g2(*tables)
    assert np.isnan(g2.loc[3, "g2_inst_late_share"])
    assert g2.loc[3, "g2_pos_dpd_max"] == 7.0


def test_columns_are_exact(tables):
    assert list(build_g2(*tables).columns) == G2_COLUMNS