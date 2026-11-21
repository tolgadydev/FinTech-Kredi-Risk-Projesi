from django.conf import settings
from django.db import models

MODEL_KEYS = [("A", "Model A"), ("B", "Model B")]


class ImmutableRecordError(Exception):
    """Karar kayıtları oluşturulduktan sonra değiştirilemez veya silinemez."""


class Application(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Bekliyor"
        SCORED = "scored", "Skorlandı"

    sk_id_curr = models.BigIntegerField(db_index=True)
    form = models.JSONField(default=dict)
    model_key = models.CharField(max_length=1, choices=MODEL_KEYS, default="A")
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.PENDING)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="applications"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"Başvuru {self.pk} (kişi {self.sk_id_curr})"


class BehaviourFeatures(models.Model):
    class Scope(models.TextChoices):
        REAL = "real", "Gerçek"
        SYNTHETIC = "synthetic", "Sentetik"

    sk_id_curr = models.BigIntegerField()
    scope = models.CharField(max_length=10, choices=Scope.choices)
    features = models.JSONField(default=dict)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["sk_id_curr", "scope"], name="one_behaviour_per_scope")
        ]


class DecisionQuerySet(models.QuerySet):
    def update(self, **kwargs):
        raise ImmutableRecordError("Kararlar toplu olarak güncellenemez")

    def delete(self):
        raise ImmutableRecordError("Kararlar toplu olarak silinemez")


class Decision(models.Model):
    class Band(models.TextChoices):
        APPROVE = "approve", "Onay"
        MANUAL_REVIEW = "manual_review", "Manuel inceleme"
        REJECT = "reject", "Ret"

    application = models.OneToOneField(
        Application, on_delete=models.PROTECT, related_name="decision"
    )
    model_version = models.CharField(max_length=100)
    data_scope = models.CharField(max_length=20)
    features = models.JSONField()
    probability = models.FloatField()
    band = models.CharField(max_length=20, choices=Band.choices)
    reasons = models.JSONField(default=list)
    behaviour_missing = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    objects = DecisionQuerySet.as_manager()

    def save(self, *args, **kwargs):
        if not self._state.adding:
            raise ImmutableRecordError("Kaydedilmiş bir karar değiştirilemez")
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        raise ImmutableRecordError("Kararlar silinemez")


class Batch(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued", "Sırada"
        RUNNING = "running", "Çalışıyor"
        DONE = "done", "Tamamlandı"
        PARTIAL = "partial", "Kısmen tamamlandı"

    model_key = models.CharField(max_length=1, choices=MODEL_KEYS, default="A")
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.QUEUED)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="batches"
    )
    created_at = models.DateTimeField(auto_now_add=True)


class BatchItem(models.Model):
    batch = models.ForeignKey(Batch, on_delete=models.CASCADE, related_name="items")
    application = models.ForeignKey(
        Application, on_delete=models.PROTECT, related_name="batch_items"
    )
    error = models.TextField(blank=True, default="")