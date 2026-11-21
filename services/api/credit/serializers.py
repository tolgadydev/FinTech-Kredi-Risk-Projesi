from rest_framework import serializers

from credit.models import Application, Batch, BatchItem

NUMBER = {"required": False, "allow_null": True}
TEXT = {"required": False, "allow_null": True, "max_length": 100}
MAX_BATCH_SIZE = 500


class ApplicationFormSerializer(serializers.Serializer):
    AMT_INCOME_TOTAL = serializers.FloatField(**NUMBER)
    AMT_CREDIT = serializers.FloatField(**NUMBER)
    AMT_ANNUITY = serializers.FloatField(**NUMBER)
    AMT_GOODS_PRICE = serializers.FloatField(**NUMBER)
    DAYS_EMPLOYED = serializers.FloatField(**NUMBER)
    REGION_RATING_CLIENT = serializers.FloatField(**NUMBER)
    NAME_CONTRACT_TYPE = serializers.CharField(**TEXT)
    NAME_INCOME_TYPE = serializers.CharField(**TEXT)
    OCCUPATION_TYPE = serializers.CharField(**TEXT)
    ORGANIZATION_TYPE = serializers.CharField(**TEXT)
    FLAG_OWN_CAR = serializers.CharField(**TEXT)
    FLAG_OWN_REALTY = serializers.CharField(**TEXT)

    def to_internal_value(self, data):
        if isinstance(data, dict):
            unknown = sorted(set(data) - set(self.fields))
            if unknown:
                raise serializers.ValidationError({name: "Bu alan kabul edilmiyor." for name in unknown})
        return super().to_internal_value(data)


class ApplicationCreateSerializer(serializers.Serializer):
    sk_id_curr = serializers.IntegerField(min_value=1)
    model = serializers.ChoiceField(choices=["A", "B"], default="A")
    application = ApplicationFormSerializer(required=False)


class BatchApplicationSerializer(serializers.Serializer):
    sk_id_curr = serializers.IntegerField(min_value=1)
    application = ApplicationFormSerializer(required=False)


class BatchCreateSerializer(serializers.Serializer):
    model = serializers.ChoiceField(choices=["A", "B"], default="A")
    applications = BatchApplicationSerializer(
        many=True, allow_empty=False, max_length=MAX_BATCH_SIZE
    )


def application_data(application: Application) -> dict:
    decision = getattr(application, "decision", None)
    return {
        "id": application.pk,
        "sk_id_curr": application.sk_id_curr,
        "model": application.model_key,
        "status": application.status,
        "created_at": application.created_at.isoformat(),
        "decision": None
        if decision is None
        else {
            "probability": decision.probability,
            "band": decision.band,
            "reasons": decision.reasons,
            "model_version": decision.model_version,
            "data_scope": decision.data_scope,
            "behaviour_missing": decision.behaviour_missing,
            "created_at": decision.created_at.isoformat(),
        },
    }


def batch_data(batch: Batch) -> dict:
    items = list(
        BatchItem.objects.filter(batch=batch)
        .select_related("application__decision")
        .order_by("pk")
    )
    statuses = [item.application.status for item in items]
    return {
        "id": batch.pk,
        "model": batch.model_key,
        "status": batch.status,
        "created_at": batch.created_at.isoformat(),
        "total": len(items),
        "scored": statuses.count(Application.Status.SCORED),
        "pending": statuses.count(Application.Status.PENDING),
        "items": [{**application_data(item.application), "error": item.error} for item in items],
    }