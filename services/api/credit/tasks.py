from celery import shared_task

from credit.loading import load_behaviour
from credit.services import score_batch


@shared_task
def score_batch_task(batch_id: int) -> str:
    return score_batch(batch_id)


@shared_task
def load_behaviour_task(path: str, scope: str) -> int:
    return load_behaviour(path, scope)