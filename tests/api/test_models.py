import pytest
from django.contrib.auth.models import User
from django.db import IntegrityError
from django.db.models import ProtectedError

from credit.models import Application, BehaviourFeatures, Decision, ImmutableRecordError


@pytest.fixture
def user(db):
    return User.objects.create_user("kurum", password="test-sifre-123")


@pytest.fixture
def application(user):
    return Application.objects.create(
        sk_id_curr=100001, form={"AMT_CREDIT": 450000}, created_by=user
    )


@pytest.fixture
def decision(application):
    return Decision.objects.create(
        application=application,
        model_version="g1g2-catboost-test",
        data_scope="real",
        features={"AMT_CREDIT": 450000},
        probability=0.20,
        band="manual_review",
        reasons=[],
    )


def test_new_application_is_pending(application):
    assert application.status == "pending"
    assert application.model_key == "A"


def test_decision_cannot_be_updated(decision):
    decision.band = "approve"
    with pytest.raises(ImmutableRecordError):
        decision.save()


def test_decision_cannot_be_deleted(decision):
    with pytest.raises(ImmutableRecordError):
        decision.delete()


def test_queryset_update_is_blocked(decision):
    with pytest.raises(ImmutableRecordError):
        Decision.objects.filter(pk=decision.pk).update(band="approve")


def test_queryset_delete_is_blocked(decision):
    with pytest.raises(ImmutableRecordError):
        Decision.objects.filter(pk=decision.pk).delete()


def test_application_with_decision_cannot_be_deleted(application, decision):
    with pytest.raises(ProtectedError):
        application.delete()


def test_one_behaviour_summary_per_person_and_scope(db):
    BehaviourFeatures.objects.create(sk_id_curr=100001, scope="real", features={})
    with pytest.raises(IntegrityError):
        BehaviourFeatures.objects.create(sk_id_curr=100001, scope="real", features={})