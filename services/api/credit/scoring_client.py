import httpx
from django.conf import settings


class ScoringUnavailable(Exception):
    """Skorlama servisine ulaşılamadı veya güvenilir bir cevap alınamadı."""


def build_score_payload(form: dict, behaviour: dict | None, model_key: str) -> dict:
    return {"model": model_key, "application": dict(form), "behaviour": dict(behaviour or {})}


def parse_score_response(data) -> dict:
    if not isinstance(data, dict):
        raise ScoringUnavailable("Cevap bir JSON nesnesi değil")
    try:
        probability = float(data["probability"])
        types_ok = (
            isinstance(data["model_version"], str)
            and isinstance(data["data_scope"], str)
            and isinstance(data["features"], dict)
            and isinstance(data["reasons"], list)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ScoringUnavailable(f"Cevapta eksik veya hatalı alan: {error}") from error
    if not types_ok:
        raise ScoringUnavailable("Cevaptaki alan tipleri hatalı")
    if not 0 <= probability <= 1:
        raise ScoringUnavailable(f"Olasılık 0-1 arasında değil: {probability}")
    return data


def score(payload: dict, client: httpx.Client | None = None, timeout: float = 5.0) -> dict:
    owns_client = client is None
    if client is None:
        client = httpx.Client(base_url=settings.SCORING_URL, timeout=timeout)
    try:
        response = client.post("/score", json=payload)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise ScoringUnavailable(str(error)) from error
    finally:
        if owns_client:
            client.close()
    return parse_score_response(data)