import pytest
from django.contrib.auth.models import User

from credit.models import Application
from rest_framework.test import APIClient

from credit import scoring_client
from credit.scoring_client import ScoringUnavailable



@pytest.fixture
def user(db):
    return User.objects.create_user("kurum", password="test-sifre-123")


@pytest.fixture
def application(user):
    return Application.objects.create(
        sk_id_curr=100001, form={"AMT_CREDIT": 450000}, created_by=user
    )

FAKE_RESPONSE = {
    "probability": 0.20,
    "raw_probability": 0.19,
    "model_version": "g1g2-catboost-test",
    "data_scope": "real",
    "features": {"AMT_CREDIT": 450000.0},
    "reasons": [
        {"feature": "AMT_CREDIT", "text": "x", "contribution": 0.1, "increases_risk": True}
    ],
}


@pytest.fixture
def auth(user):
    client = APIClient()
    client.force_authenticate(user)
    return client


@pytest.fixture
def fake_scoring(monkeypatch):
    monkeypatch.setattr(scoring_client, "score", lambda payload, **kw: dict(FAKE_RESPONSE))


@pytest.fixture
def scoring_offline(monkeypatch):
    def fake_score(payload, **kwargs):
        raise ScoringUnavailable("servis kapalı")

    monkeypatch.setattr(scoring_client, "score", fake_score)