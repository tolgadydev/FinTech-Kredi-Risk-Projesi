from typing import cast
from app.schemas import ApplicationIn, ReasonOut, ScoreRequest, ScoreResponse

from credit.scoring_client import build_score_payload, parse_score_response
from credit.serializers import ApplicationFormSerializer


def test_django_request_matches_fastapi_schema():
    payload = build_score_payload(
        form={"AMT_CREDIT": 450000.0, "NAME_INCOME_TYPE": "Working"},
        behaviour={"g2_inst_late_share": 0.1, "g2_prev_count": None},
        model_key="A",
    )
    ScoreRequest.model_validate(payload)


def test_fastapi_response_is_accepted_by_django():
    response = ScoreResponse(
        probability=0.2,
        raw_probability=0.19,
        model_version="g1g2-catboost-test",
        data_scope="real",
        features={"AMT_CREDIT": 450000.0},
        reasons=[
            ReasonOut(feature="AMT_CREDIT", text="x", contribution=0.1, increases_risk=True)
        ],
    )
    assert parse_score_response(response.model_dump())["model_version"] == "g1g2-catboost-test"


def test_django_form_fields_match_fastapi():
    form = cast(ApplicationFormSerializer, ApplicationFormSerializer())
    assert set(form.fields) == set(ApplicationIn.model_fields)