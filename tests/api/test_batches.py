import pytest

from credit import scoring_client
from credit.models import Application, Batch, BatchItem, Decision
from credit.scoring_client import ScoringUnavailable
from credit.services import score_application, score_batch
from tests.api.conftest import FAKE_RESPONSE


@pytest.fixture
def batch(user):
    batch = Batch.objects.create(created_by=user)
    for sk_id_curr in (100001, 100002, 100003):
        application = Application.objects.create(
            sk_id_curr=sk_id_curr, form={}, created_by=user
        )
        BatchItem.objects.create(batch=batch, application=application)
    return batch


@pytest.fixture
def fails_after_first(monkeypatch):
    calls = []

    def fake_score(payload, **kwargs):
        calls.append(payload)
        if len(calls) > 1:
            raise ScoringUnavailable("servis kapandı")
        return dict(FAKE_RESPONSE)

    monkeypatch.setattr(scoring_client, "score", fake_score)
    return calls


def _statuses(batch):
    return [item.application.status for item in batch.items.order_by("pk")]


def test_fully_scored_batch_is_done(batch, fake_scoring):
    assert score_batch(batch.pk) == "done"
    batch.refresh_from_db()
    assert batch.status == "done"
    assert _statuses(batch) == ["scored", "scored", "scored"]


def test_service_dying_midway_gives_partial(batch, fails_after_first):
    assert score_batch(batch.pk) == "partial"
    assert _statuses(batch) == ["scored", "pending", "pending"]
    assert [bool(item.error) for item in batch.items.order_by("pk")] == [False, True, True]


def test_stops_calling_service_once_it_is_down(batch, fails_after_first):
    score_batch(batch.pk)
    assert len(fails_after_first) == 2


def test_already_decided_items_are_skipped(batch, fake_scoring):
    score_application(batch.items.order_by("pk").first().application)
    assert score_batch(batch.pk) == "done"
    assert Decision.objects.count() == 3


def test_rerun_after_recovery_completes_batch(batch, fails_after_first, monkeypatch):
    score_batch(batch.pk)
    monkeypatch.setattr(scoring_client, "score", lambda payload, **kw: dict(FAKE_RESPONSE))
    assert score_batch(batch.pk) == "done"
    assert all(item.error == "" for item in batch.items.all())