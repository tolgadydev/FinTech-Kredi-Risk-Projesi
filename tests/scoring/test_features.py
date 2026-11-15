import pandas as pd
import pytest
from credit_features.application import RAW_G1_COLUMNS, build_g1
from credit_features.frame import align_frame, fit_spec
from pydantic import ValidationError

from app.features import request_to_frame
from app.schemas import ApplicationIn, ScoreRequest


def _application(row: pd.Series) -> dict:
    return {column: (None if pd.isna(row[column]) else row[column]) for column in RAW_G1_COLUMNS}


def test_schema_fields_match_g1_columns():
    assert list(ApplicationIn.model_fields) == RAW_G1_COLUMNS


@pytest.mark.parametrize("row", range(4))
def test_serving_features_match_training(app_df, row):
    training = build_g1(app_df)
    spec = fit_spec(training)
    expected = align_frame(training.iloc[[row]], spec).reset_index(drop=True)
    request = ScoreRequest(application=_application(app_df.iloc[row]))
    pd.testing.assert_frame_equal(request_to_frame(request, spec), expected)


def test_behaviour_values_reach_the_model(app_df):
    spec = fit_spec(build_g1(app_df).assign(g2_inst_late_share=0.1))
    frame = request_to_frame(ScoreRequest(behaviour={"g2_inst_late_share": 0.5}), spec)
    assert frame.loc[0, "g2_inst_late_share"] == 0.5


def test_missing_behaviour_becomes_missing_value(app_df):
    spec = fit_spec(build_g1(app_df).assign(g2_inst_late_share=0.1))
    assert request_to_frame(ScoreRequest(), spec)["g2_inst_late_share"].isna().all()


@pytest.mark.parametrize(
    "payload",
    [
        {"application": {"CODE_GENDER": "F"}},
        {"application": {"AMT_CREDIT": "abc"}},
        {"behaviour": {"TARGET": 1.0}},
        {"model": "C"},
    ],
)
def test_invalid_requests_are_rejected(payload):
    with pytest.raises(ValidationError):
        ScoreRequest(**payload)