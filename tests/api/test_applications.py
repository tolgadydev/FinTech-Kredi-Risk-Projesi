# pyright: reportAttributeAccessIssue=false
from django.contrib.auth.models import User
from rest_framework.test import APIClient

from credit.models import Application, Decision

VALID = {"sk_id_curr": 100001, "application": {"AMT_CREDIT": 450000, "NAME_INCOME_TYPE": "Working"}}


def test_api_requires_login(db):
    assert APIClient().get("/api/v1/applications/1").status_code == 401


def test_create_scores_and_returns_decision(auth, fake_scoring):
    response = auth.post("/api/v1/applications", VALID, format="json")
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "scored"
    assert body["decision"]["band"] == "manual_review"
    assert Decision.objects.count() == 1


def test_unknown_person_is_flagged(auth, fake_scoring):
    response = auth.post("/api/v1/applications", VALID, format="json")
    assert response.json()["decision"]["behaviour_missing"] is True


def test_forbidden_field_gets_400(auth, fake_scoring):
    payload = {**VALID, "application": {"CODE_GENDER": "F"}}
    assert auth.post("/api/v1/applications", payload, format="json").status_code == 400
    assert Application.objects.count() == 0


def test_wrong_type_gets_400(auth, fake_scoring):
    payload = {**VALID, "application": {"AMT_CREDIT": "abc"}}
    assert auth.post("/api/v1/applications", payload, format="json").status_code == 400


def test_scoring_offline_gives_503_and_keeps_application_pending(auth, scoring_offline):
    response = auth.post("/api/v1/applications", VALID, format="json")
    assert response.status_code == 503
    assert Application.objects.get().status == "pending"
    assert Decision.objects.count() == 0


def test_pending_application_can_be_retried(auth, application, fake_scoring):
    response = auth.post(f"/api/v1/applications/{application.pk}/score")
    assert response.status_code == 200
    assert response.json()["decision"]["band"] == "manual_review"


def test_rescoring_decided_application_gets_409(auth, application, fake_scoring):
    auth.post(f"/api/v1/applications/{application.pk}/score")
    response = auth.post(f"/api/v1/applications/{application.pk}/score")
    assert response.status_code == 409
    assert Decision.objects.count() == 1


def test_get_returns_application_with_decision(auth, fake_scoring):
    created = auth.post("/api/v1/applications", VALID, format="json").json()
    response = auth.get(f"/api/v1/applications/{created['id']}")
    assert response.status_code == 200
    assert response.json()["decision"]["band"] == "manual_review"


def test_other_institution_cannot_see_application(application):
    other = APIClient()
    other.force_authenticate(User.objects.create_user("baska-kurum", password="x-123456789"))
    assert other.get(f"/api/v1/applications/{application.pk}").status_code == 404