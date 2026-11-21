import json

import pytest
from django.core.management import call_command

from credit.loading import load_behaviour
from credit.models import BehaviourFeatures


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def real_file(tmp_path):
    return _write(
        tmp_path / "real.jsonl",
        [
            {"sk_id_curr": 100001, "features": {"g2_prev_count": 3.0, "g2_pos_dpd_max": None}},
            {"sk_id_curr": 100002, "features": {"g2_prev_count": 1.0, "g2_pos_dpd_max": 5.0}},
        ],
    )


def test_loads_one_row_per_person(db, real_file):
    assert load_behaviour(real_file, "real") == 2
    row = BehaviourFeatures.objects.get(sk_id_curr=100001, scope="real")
    assert row.features == {"g2_prev_count": 3.0, "g2_pos_dpd_max": None}


def test_loading_again_updates_instead_of_duplicating(db, real_file, tmp_path):
    load_behaviour(real_file, "real")
    updated = _write(
        tmp_path / "updated.jsonl",
        [{"sk_id_curr": 100001, "features": {"g2_prev_count": 9.0}}],
    )
    load_behaviour(updated, "real")
    assert BehaviourFeatures.objects.count() == 2
    assert BehaviourFeatures.objects.get(sk_id_curr=100001).features == {"g2_prev_count": 9.0}


def test_real_and_synthetic_are_kept_apart(db, real_file, tmp_path):
    synthetic = _write(
        tmp_path / "syn.jsonl", [{"sk_id_curr": 100001, "features": {"syn_order_count": 5.0}}]
    )
    load_behaviour(real_file, "real")
    load_behaviour(synthetic, "synthetic")
    assert BehaviourFeatures.objects.filter(sk_id_curr=100001).count() == 2


def test_synthetic_feature_cannot_be_loaded_as_real(db, tmp_path):
    mixed = _write(
        tmp_path / "mixed.jsonl",
        [
            {"sk_id_curr": 100001, "features": {"g2_prev_count": 3.0}},
            {"sk_id_curr": 100002, "features": {"syn_order_count": 5.0}},
        ],
    )
    with pytest.raises(ValueError, match="syn_order_count"):
        load_behaviour(mixed, "real")
    assert BehaviourFeatures.objects.count() == 0


def test_unknown_scope_is_rejected(db, real_file):
    with pytest.raises(ValueError):
        load_behaviour(real_file, "test")


def test_management_command_loads_file(db, real_file):
    call_command("load_features", str(real_file), "--scope", "real")
    assert BehaviourFeatures.objects.count() == 2