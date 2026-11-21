import pytest

from credit import scoring_client
from credit.models import Application, BehaviourFeatures, Decision
from credit.scoring_client import ScoringUnavailable
from credit.services import AlreadyDecided, score_application


def _response(probability: float = 0.20) -> dict:
    return {
        "probability": probability,
        "raw_probability": probability,
        "model_version": "g1g2-catboost-test",
        "data_scope": "real",
        "features": {"AMT_CREDIT": 450000.0},
        "reasons": [
            {"feature": "AMT_CREDIT", "text": "x", "contribution": 0.1, "increases_risk": True}
        ],
    }


@pytest.fixture
def sent(monkeypatch):
    calls = []

    def fake_score(payload, **kwargs):
        calls.append(payload)
        return _response()

    monkeypatch.setattr(scoring_client, "score", fake_score)
    return calls


@pytest.fixture
def scoring_down(monkeypatch):
    def fake_score(payload, **kwargs):
        raise ScoringUnavailable("servis kapalı")

    monkeypatch.setattr(scoring_client, "score", fake_score)


def test_scoring_writes_decision_and_marks_application(application, sent):
    decision = score_application(application)
    application.refresh_from_db()
    assert application.status == "scored"
    assert decision.band == "manual_review"
    assert decision.model_version == "g1g2-catboost-test"
    assert Decision.objects.count() == 1


@pytest.mark.parametrize(
    "probability, band", [(0.05, "approve"), (0.20, "manual_review"), (0.30, "reject")]
)
def test_band_follows_decision_rule(application, monkeypatch, probability, band):
    monkeypatch.setattr(scoring_client, "score", lambda payload, **kw: _response(probability))
    assert score_application(application).band == band


def test_behaviour_is_sent_when_available(application, sent):
    BehaviourFeatures.objects.create(sk_id_curr=100001, scope="real", features={"g2_prev_count": 3.0})
    decision = score_application(application)
    assert sent[0]["behaviour"] == {"g2_prev_count": 3.0}
    assert decision.behaviour_missing is False


def test_unknown_person_is_scored_and_flagged(application, sent):
    decision = score_application(application)
    assert sent[0]["behaviour"] == {}
    assert decision.behaviour_missing is True


def test_model_a_never_receives_synthetic_data(application, sent):
    BehaviourFeatures.objects.create(
        sk_id_curr=100001, scope="synthetic", features={"syn_order_count": 5.0}
    )
    score_application(application)
    assert sent[0]["behaviour"] == {}


def test_model_b_receives_real_and_synthetic_data(user, sent):
    application = Application.objects.create(
        sk_id_curr=100001, form={}, model_key="B", created_by=user
    )
    BehaviourFeatures.objects.create(sk_id_curr=100001, scope="real", features={"g2_prev_count": 3.0})
    BehaviourFeatures.objects.create(
        sk_id_curr=100001, scope="synthetic", features={"syn_order_count": 5.0}
    )
    score_application(application)
    assert sent[0]["behaviour"] == {"g2_prev_count": 3.0, "syn_order_count": 5.0}


def test_scoring_down_leaves_application_pending(application, scoring_down):
    with pytest.raises(ScoringUnavailable):
        score_application(application)
    application.refresh_from_db()
    assert application.status == "pending"
    assert Decision.objects.count() == 0


def test_already_decided_application_is_not_scored_again(application, sent):
    score_application(application)
    with pytest.raises(AlreadyDecided):
        score_application(application)
    assert Decision.objects.count() == 1
    assert len(sent) == 1