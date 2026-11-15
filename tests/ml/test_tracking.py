import pytest
from mlflow.tracking import MlflowClient

from ml.tracking import log_bundle, tracking_uri
from ml.train_final import train_bundle

FAST = {"iterations": 50}
REPORT = {"all": {"roc_auc": 0.70}, "thin_file": {"roc_auc": 0.72}}


@pytest.fixture
def bundle(data, tmp_path):
    X, y, _ = data
    return train_bundle(X, y, tmp_path / "model", group="G1G2", params=FAST)


@pytest.fixture
def synthetic_bundle(data, tmp_path):
    X, y, _ = data
    return train_bundle(
        X, y, tmp_path / "syn", group="G1G2", params=FAST, data_scope="synthetic_demo"
    )


def test_logs_tags_and_oof_metrics(bundle, tmp_path):
    store = tmp_path / "mlruns"
    run = MlflowClient(tracking_uri(store)).get_run(log_bundle(bundle, store))
    assert run.data.tags["data_scope"] == "real"
    assert run.data.tags["group"] == "G1G2"
    assert "oof_roc_auc" in run.data.metrics


def test_logs_test_metrics_when_report_given(bundle, tmp_path):
    store = tmp_path / "mlruns"
    run_id = log_bundle(bundle, store, test_report=REPORT)
    metrics = MlflowClient(tracking_uri(store)).get_run(run_id).data.metrics
    assert metrics["test_roc_auc"] == 0.70
    assert metrics["test_thin_roc_auc"] == 0.72


def test_bundle_files_are_attached(bundle, tmp_path):
    store = tmp_path / "mlruns"
    run_id = log_bundle(bundle, store)
    files = [a.path for a in MlflowClient(tracking_uri(store)).list_artifacts(run_id, "bundle")]
    assert "bundle/metadata.json" in files


def test_synthetic_model_cannot_log_test_metrics(synthetic_bundle, tmp_path):
    store = tmp_path / "mlruns"
    with pytest.raises(ValueError, match="synthetic_demo"):
        log_bundle(synthetic_bundle, store, test_report=REPORT)
    assert not (store / "mlflow.db").exists()