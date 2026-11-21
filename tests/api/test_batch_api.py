# pyright: reportAttributeAccessIssue=false
from types import SimpleNamespace

import pytest
from django.contrib.auth.models import User
from rest_framework.test import APIClient

from credit import tasks
from credit.models import Application, Batch
from credit.services import score_batch

PAYLOAD = {
    "model": "A",
    "applications": [
        {"sk_id_curr": 100001, "application": {"AMT_CREDIT": 450000}},
        {"sk_id_curr": 100002},
    ],
}


@pytest.fixture
def queued(monkeypatch):
    sent = []
    monkeypatch.setattr(
        tasks, "score_batch_task", SimpleNamespace(delay=lambda batch_id: sent.append(batch_id))
    )
    return sent


def test_create_batch_returns_202_and_queues_job(auth, queued, django_capture_on_commit_callbacks):
    with django_capture_on_commit_callbacks(execute=True):
        response = auth.post("/api/v1/batches", PAYLOAD, format="json")
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["total"] == 2
    assert queued == [body["id"]]


def test_job_is_queued_only_after_commit(auth, queued):
    auth.post("/api/v1/batches", PAYLOAD, format="json")
    assert queued == []


def test_batch_progress_is_visible(auth, queued, fake_scoring, django_capture_on_commit_callbacks):
    with django_capture_on_commit_callbacks(execute=True):
        batch_id = auth.post("/api/v1/batches", PAYLOAD, format="json").json()["id"]
    score_batch(batch_id)
    body = auth.get(f"/api/v1/batches/{batch_id}").json()
    assert body["status"] == "done"
    assert body["scored"] == 2
    assert all(item["decision"]["band"] == "manual_review" for item in body["items"])


def test_empty_batch_is_rejected(auth, queued):
    response = auth.post("/api/v1/batches", {"applications": []}, format="json")
    assert response.status_code == 400
    assert Batch.objects.count() == 0


def test_one_invalid_application_rejects_whole_batch(auth, queued):
    payload = {"applications": [{"sk_id_curr": 100001}, {"sk_id_curr": 100002, "application": {"CODE_GENDER": "F"}}]}
    assert auth.post("/api/v1/batches", payload, format="json").status_code == 400
    assert Batch.objects.count() == 0
    assert Application.objects.count() == 0


def test_broker_down_still_accepts_batch(auth, monkeypatch, django_capture_on_commit_callbacks):
    def broken_delay(batch_id):
        raise ConnectionError("Redis çalışmıyor")

    monkeypatch.setattr(tasks, "score_batch_task", SimpleNamespace(delay=broken_delay))
    with django_capture_on_commit_callbacks(execute=True):
        response = auth.post("/api/v1/batches", PAYLOAD, format="json")
    assert response.status_code == 202
    assert Batch.objects.get().status == "queued"


def test_other_institution_cannot_see_batch(auth, queued):
    batch_id = auth.post("/api/v1/batches", PAYLOAD, format="json").json()["id"]
    other = APIClient()
    other.force_authenticate(User.objects.create_user("baska-kurum", password="x-123456789"))
    assert other.get(f"/api/v1/batches/{batch_id}").status_code == 404