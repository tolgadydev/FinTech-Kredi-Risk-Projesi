import pandas as pd
import pytest

from ml.dataset import build_dataset, thin_file_mask


@pytest.fixture
def parquet_dir(tmp_path, app_df):
    tables = {
        "application_train": app_df,
        "installments_payments": pd.DataFrame(
            {
                "SK_ID_CURR": [1, 1],
                "DAYS_INSTALMENT": [-10.0, -20.0],
                "DAYS_ENTRY_PAYMENT": [-5.0, -20.0],
                "AMT_INSTALMENT": [100.0, 100.0],
                "AMT_PAYMENT": [100.0, 100.0],
            }
        ),
        "POS_CASH_balance": pd.DataFrame({"SK_ID_CURR": [1], "SK_DPD": [0.0]}),
        "credit_card_balance": pd.DataFrame(
            {"SK_ID_CURR": [2], "AMT_BALANCE": [50.0], "AMT_CREDIT_LIMIT_ACTUAL": [100.0]}
        ),
        "previous_application": pd.DataFrame(
            {"SK_ID_CURR": [1], "NAME_CONTRACT_STATUS": ["Approved"]}
        ),
        "bureau": pd.DataFrame(
            {
                "SK_ID_CURR": [1, 2],
                "SK_ID_BUREAU": [10, 20],
                "CREDIT_ACTIVE": ["Active", "Closed"],
                "AMT_CREDIT_MAX_OVERDUE": [0.0, 0.0],
            }
        ),
        "bureau_balance": pd.DataFrame(
            {"SK_ID_BUREAU": [10], "MONTHS_BALANCE": [-1], "STATUS": ["0"]}
        ),
    }
    for name, table in tables.items():
        table.to_parquet(tmp_path / f"{name}.parquet", index=False)
    return tmp_path


def _has_prefix(columns, prefixes):
    return [c for c in columns if c.startswith(prefixes)]


def test_g1_has_no_behaviour_or_bureau_columns(parquet_dir):
    X, _ = build_dataset("G1", parquet_dir)
    assert not _has_prefix(X.columns, ("g2_", "g3_", "EXT_"))


def test_g1g2_adds_behaviour_but_not_bureau(parquet_dir):
    X, _ = build_dataset("G1G2", parquet_dir)
    assert "g2_inst_late_share" in X.columns
    assert not _has_prefix(X.columns, ("g3_", "EXT_"))


def test_g1g2g3_adds_bureau(parquet_dir):
    X, _ = build_dataset("G1G2G3", parquet_dir)
    assert {"g3_bureau_count", "EXT_SOURCE_2"} <= set(X.columns)


def test_rows_follow_applications(parquet_dir, app_df):
    X, y = build_dataset("G1G2", parquet_dir)
    assert list(X.index) == list(app_df["SK_ID_CURR"])
    assert y.loc[2] == 1
    assert "TARGET" not in X.columns


def test_unknown_group_is_rejected(parquet_dir):
    with pytest.raises(ValueError):
        build_dataset("G2", parquet_dir)


def test_thin_file_mask_marks_people_without_bureau():
    mask = thin_file_mask(pd.Index([1, 2, 3, 4]), pd.DataFrame({"SK_ID_CURR": [1, 2, 2]}))
    assert list(mask) == [False, False, True, True]