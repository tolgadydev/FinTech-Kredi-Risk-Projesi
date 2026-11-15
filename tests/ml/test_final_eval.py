import numpy as np
import pytest

from ml.final_eval import final_evaluate
from ml.train_final import train_bundle

FAST = {"iterations": 50}


@pytest.fixture
def setup(data, tmp_path):
    X, y, _ = data
    train, test = X.iloc[:400], X.iloc[400:]
    main = train_bundle(train, y.iloc[:400], tmp_path / "main", group="G1G2", params=FAST)
    compare = train_bundle(
        train[["a", "job"]], y.iloc[:400], tmp_path / "g1", group="G1", params=FAST
    )
    thin = np.arange(len(test)) % 2 == 0
    return main, compare, test, y.iloc[400:], thin


def test_report_contains_criteria(setup, tmp_path):
    report = final_evaluate(*setup, marker=tmp_path / "used.json", n_boot=100)
    assert {"all", "thin_file", "criterion_2", "criterion_3"} <= set(report)
    assert isinstance(report["criterion_2"]["passed"], bool)


def test_holdout_is_single_use(setup, tmp_path):
    marker = tmp_path / "used.json"
    final_evaluate(*setup, marker=marker, n_boot=100)
    assert marker.exists()
    with pytest.raises(RuntimeError, match="holdout"):
        final_evaluate(*setup, marker=marker, n_boot=100)


def test_synthetic_bundle_cannot_take_the_exam(data, setup, tmp_path):
    X, y, _ = data
    synthetic = train_bundle(
        X.iloc[:400], y.iloc[:400], tmp_path / "syn",
        group="G1G2", params=FAST, data_scope="synthetic_demo",
    )
    _, compare, test, y_test, thin = setup
    marker = tmp_path / "used.json"
    with pytest.raises(ValueError, match="synthetic_demo"):
        final_evaluate(synthetic, compare, test, y_test, thin, marker=marker, n_boot=100)
    assert not marker.exists()