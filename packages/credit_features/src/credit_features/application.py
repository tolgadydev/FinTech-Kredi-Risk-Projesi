import numpy as np
import pandas as pd

EMPLOYED_SENTINEL = 365243
DAYS_PER_YEAR = 365.25

NUMERIC_G1_COLUMNS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
    "REGION_RATING_CLIENT",
]
CATEGORICAL_G1_COLUMNS = [
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
]
RAW_G1_COLUMNS = NUMERIC_G1_COLUMNS + CATEGORICAL_G1_COLUMNS
DEMOGRAPHIC_COLUMNS = [
    "CODE_GENDER",
    "DAYS_BIRTH",
    "NAME_FAMILY_STATUS",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
]


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator > 0, np.nan)


def build_g1(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reindex(columns=RAW_G1_COLUMNS).copy()
    for column in NUMERIC_G1_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].replace(EMPLOYED_SENTINEL, np.nan)

    out["credit_income_ratio"] = _ratio(out["AMT_CREDIT"], out["AMT_INCOME_TOTAL"])
    out["annuity_income_ratio"] = _ratio(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
    out["credit_goods_ratio"] = _ratio(out["AMT_CREDIT"], out["AMT_GOODS_PRICE"])
    out["employed_years"] = -out["DAYS_EMPLOYED"] / DAYS_PER_YEAR
    return out