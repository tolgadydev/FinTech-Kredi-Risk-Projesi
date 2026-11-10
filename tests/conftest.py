import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def app_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "TARGET": [0, 1, 0, 0],
            "AMT_INCOME_TOTAL": [100000.0, 200000.0, 150000.0, 90000.0],
            "AMT_CREDIT": [300000.0, 400000.0, 450000.0, 180000.0],
            "AMT_ANNUITY": [15000.0, 20000.0, 22000.0, np.nan],
            "AMT_GOODS_PRICE": [250000.0, 400000.0, 450000.0, 150000.0],
            "DAYS_EMPLOYED": [-1000, -3652, -500, -2000],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Cash loans", "Revolving loans", "Cash loans"],
            "NAME_INCOME_TYPE": ["Working", "Pensioner", "Working", "Working"],
            "OCCUPATION_TYPE": ["Laborers", None, "Drivers", "Laborers"],
            "ORGANIZATION_TYPE": ["School", "XNA", "Self-employed", "School"],
            "FLAG_OWN_CAR": ["N", "Y", "N", "N"],
            "FLAG_OWN_REALTY": ["Y", "Y", "N", "Y"],
            "REGION_RATING_CLIENT": [2, 1, 3, 2],
            "CODE_GENDER": ["F", "M", "F", "M"],
            "DAYS_BIRTH": [-12000, -20000, -9000, -15000],
            "NAME_FAMILY_STATUS": ["Married", "Widow", "Single", "Married"],
            "CNT_CHILDREN": [1, 0, 0, 2],
            "CNT_FAM_MEMBERS": [3.0, 1.0, 1.0, 4.0],
            "EXT_SOURCE_1": [0.5, np.nan, 0.3, 0.7],
            "EXT_SOURCE_2": [0.6, 0.2, 0.4, 0.8],
            "EXT_SOURCE_3": [np.nan, 0.1, 0.5, 0.9],
        }
    )


@pytest.fixture
def raw_csv_dir(tmp_path, app_df):
    raw = tmp_path / "raw"
    raw.mkdir()
    app_df.to_csv(raw / "application_train.csv", index=False)
    return raw