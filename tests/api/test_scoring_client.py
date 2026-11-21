import httpx
import pytest

from credit.scoring_client import ScoringUnavailable, build_score_payload, score

RESPONSE = {
    "probability": 0.2,
    "raw_probability": 0.19,
    "model_version": "g1g2-catboost-test",
    "data_scope": "real",
    "features": {},
    "reasons": [],
}


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://scoring")


def test_payload_shape():
    payload = build_score_payload(
        form={"AMT_CREDIT": 450000.0}, behaviour={"g2_prev_count": 2.0}, model_key="A"
    )
    assert payload == {
        "model": "A",
        "application": {"AMT_CREDIT": 450000.0},
        "behaviour": {"g2_prev_count": 2.0},
    }


def test_score_sends_to_score_endpoint_and_returns_response():
    def handler(request):
        assert request.url.path == "/score"
        return httpx.Response(200, json=RESPONSE)

    assert score({"model": "A"}, client=_client(handler))["probability"] == 0.2


def test_connection_error_means_unavailable():
    def handler(request):
        raise httpx.ConnectError("servis kapalı")

    with pytest.raises(ScoringUnavailable):
        score({"model": "A"}, client=_client(handler))


def test_error_status_means_unavailable():
    with pytest.raises(ScoringUnavailable):
        score({"model": "A"}, client=_client(lambda request: httpx.Response(503)))


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="bu JSON değil"),
        httpx.Response(200, json={"beklenmeyen": 1}),
        httpx.Response(200, json={**RESPONSE, "probability": 1.5}),
    ],
)
def test_malformed_response_means_unavailable(response):
    with pytest.raises(ScoringUnavailable):
        score({"model": "A"}, client=_client(lambda request: response))