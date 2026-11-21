from django.conf import settings
from django.db import transaction
from credit_decision import assign_band, load_thresholds

from credit import scoring_client
from credit.models import Application, BehaviourFeatures, Decision

SCOPES_BY_MODEL = {"A": ["real"], "B": ["real", "synthetic"]}


class AlreadyDecided(Exception):
    """Bu başvuru için zaten bir karar verilmiş."""


def behaviour_for(sk_id_curr: int, model_key: str) -> dict | None:
    rows = BehaviourFeatures.objects.filter(
        sk_id_curr=sk_id_curr, scope__in=SCOPES_BY_MODEL[model_key]
    ).order_by("scope")
    if not rows.exists():
        return None
    merged = {}
    for row in rows:
        merged.update(row.features)
    return merged


def score_application(application: Application) -> Decision:
    if Decision.objects.filter(application=application).exists():
        raise AlreadyDecided(f"Başvuru {application.pk} için zaten karar verilmiş")

    behaviour = behaviour_for(application.sk_id_curr, application.model_key)
    payload = scoring_client.build_score_payload(
        application.form, behaviour, application.model_key
    )
    result = scoring_client.score(payload)
    thresholds = load_thresholds(settings.DECISION_CONFIG)

    with transaction.atomic():
        locked = Application.objects.select_for_update().get(pk=application.pk)
        if Decision.objects.filter(application=locked).exists():
            raise AlreadyDecided(f"Başvuru {application.pk} için zaten karar verilmiş")
        decision = Decision.objects.create(
            application=locked,
            model_version=result["model_version"],
            data_scope=result["data_scope"],
            features=result["features"],
            probability=result["probability"],
            band=assign_band(result["probability"], thresholds),
            reasons=result["reasons"],
            behaviour_missing=behaviour is None,
        )
        locked.status = Application.Status.SCORED
        locked.save(update_fields=["status"])
    return decision