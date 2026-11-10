import numpy as np

from credit_features.application import DEMOGRAPHIC_COLUMNS, build_g1
from credit_features.frame import align_frame, fit_spec


def test_employed_sentinel_is_missing(app_df):
    app_df.loc[0, "DAYS_EMPLOYED"] = 365243
    assert np.isnan(build_g1(app_df).loc[0, "employed_years"])


def test_zero_income_gives_nan_not_inf(app_df):
    app_df.loc[0, "AMT_INCOME_TOTAL"] = 0
    out = build_g1(app_df)
    assert np.isnan(out.loc[0, "credit_income_ratio"])
    assert not np.isinf(out.select_dtypes("number")).any().any()


def test_no_demographics(app_df):
    assert not set(build_g1(app_df).columns) & set(DEMOGRAPHIC_COLUMNS)


def test_align_maps_unseen_category_to_missing(app_df):
    g1 = build_g1(app_df)
    spec = fit_spec(g1)
    g1.loc[0, "OCCUPATION_TYPE"] = "YENI_MESLEK"
    assert align_frame(g1, spec).loc[0, "OCCUPATION_TYPE"] == "MISSING"