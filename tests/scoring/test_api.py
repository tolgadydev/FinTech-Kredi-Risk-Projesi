import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from credit_features.application import build_g1
from fastapi.testclient import TestClient

from app.main import create_app
from ml.train_final import train_bundle

REASON_FILE = Path(__file__).parents[2] / "config" / "reason_codes.yaml"
VALID = {
    "application": {"AMT_INCOME_TOTAL": 150000, "AMT_CREDIT": 450000, "NAME_INCOME_TYPE": "Working"},
    "behaviour": {"g2_inst_late_share": 0.3},
}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    rng = np.random.default_rng(0)
    n = 400
    raw = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": rng.uniform(5e4, 3e5, n),
            "AMT_CREDIT": rng.uniform(1e5, 1e6, n),
            "AMT_ANNUITY": rng.uniform(5e3, 5e4, n),
            "AMT_GOODS_PRICE": rng.uniform(1e5, 1e6, n),
            "DAYS_EMPLOYED": rng.integers(-10000, -100, n).astype(float),
            "REGION_RATING_CLIENT": rng.integers(1, 4, n).astype(float),
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
            "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner"], n),
            "OCCUPATION_TYPE": rng.choice(np.array(["Laborers", "Drivers", None], dtype=object), n),
            "ORGANIZATION_TYPE": rng.choice(["School", "XNA"], n),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], n),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n),
        }
    )
    late = rng.uniform(0, 1, n)
    X = build_g1(raw).assign(g2_inst_late_share=late)
    y = pd.Series((late + rng.normal(0, 0.3, n) > 0.8).astype(int))
    out = tmp_path_factory.mktemp("models") / "model_a"
    return train_bundle(X, y, out, group="G1G2", params={"iterations": 30})


@pytest.fixture(scope="module")
def client(model_dir):
    with TestClient(create_app({"A": model_dir}, REASON_FILE)) as test_client:
        yield test_client


def test_health_reports_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "A" in response.json()["models"]


def test_score_returns_probability_and_three_reasons(client):
    response = client.post("/score", json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["probability"] <= 1
    assert len(body["reasons"]) == 3
    assert body["data_scope"] == "real"


def test_zero_income_still_scores(client):
    payload = {**VALID, "application": {**VALID["application"], "AMT_INCOME_TOTAL": 0}}
    assert client.post("/score", json=payload).status_code == 200


def test_empty_request_is_scored(client):
    assert client.post("/score", json={}).status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"application": {"AMT_CREDIT": "abc"}},
        {"application": {"CODE_GENDER": "F"}},
        {"behaviour": {"TARGET": 1.0}},
    ],
)
def test_invalid_requests_get_422(client, payload):
    assert client.post("/score", json=payload).status_code == 422


def test_model_that_is_not_loaded_gets_503(client):
    assert client.post("/score", json={**VALID, "model": "B"}).status_code == 503


def test_mismatched_bundle_is_not_served(model_dir, tmp_path):
    broken = tmp_path / "broken"
    shutil.copytree(model_dir, broken)
    spec = json.loads((broken / "features.json").read_text(encoding="utf-8"))
    spec["numeric"].remove("g2_inst_late_share")
    (broken / "features.json").write_text(json.dumps(spec), encoding="utf-8")

    with TestClient(create_app({"A": broken}, REASON_FILE)) as broken_client:
        assert broken_client.get("/health").status_code == 503
        assert broken_client.post("/score", json=VALID).status_code == 503


def test_missing_bundle_is_not_served(tmp_path):
    with TestClient(create_app({"A": tmp_path / "yok"}, REASON_FILE)) as missing_client:
        assert missing_client.get("/health").status_code == 503