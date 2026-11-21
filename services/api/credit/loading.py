import json
from itertools import islice
from pathlib import Path

from django.db import transaction

from credit.models import BehaviourFeatures

CHUNK_SIZE = 5000
PREFIXES = {"real": "g2_", "synthetic": "syn_"}


def _rows(lines: list[str], scope: str):
    prefix = PREFIXES[scope]
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        wrong = [name for name in record["features"] if not name.startswith(prefix)]
        if wrong:
            raise ValueError(
                f"{record['sk_id_curr']} numaralı kişide '{scope}' türüne uymayan alanlar: {wrong}"
            )
        yield BehaviourFeatures(
            sk_id_curr=int(record["sk_id_curr"]), scope=scope, features=record["features"]
        )


def load_behaviour(path: Path, scope: str) -> int:
    if scope not in PREFIXES:
        raise ValueError(f"Bilinmeyen tür: {scope}. Seçenekler: {list(PREFIXES)}")

    total = 0
    with Path(path).open(encoding="utf-8") as file, transaction.atomic():
        while lines := list(islice(file, CHUNK_SIZE)):
            rows = list(_rows(lines, scope))
            BehaviourFeatures.objects.bulk_create(
                rows,
                update_conflicts=True,
                unique_fields=["sk_id_curr", "scope"],
                update_fields=["features"],
            )
            total += len(rows)
    return total