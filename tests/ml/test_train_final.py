import numpy as np
import pytest

from ml.train_final import fit_calibrator, load_bundle, train_bundle

FAST = {"iterations": 50}


@pytest.fixture
def bundle_dir(data, tmp_path):
    X, y, _ = data
    return train_bundle(X, y, tmp_path / "model", group="TEST", params=FAST)


def test_calibrator_outputs_probabilities():
    calibrator = fit_calibrator(np.linspace(0, 1, 100), np.r_[np.zeros(80), np.ones(20)])
    prob = calibrator.predict(np.array([-1.0, 0.5, 2.0]))
    assert ((prob >= 0) & (prob <= 1)).all()


def test_bundle_round_trip(data, bundle_dir):
    X, _, _ = data
    prob = load_bundle(bundle_dir).predict(X)
    assert len(prob) == len(X)
    assert ((prob >= 0) & (prob <= 1)).all()


def test_bundle_metadata(bundle_dir):
    meta = load_bundle(bundle_dir).metadata
    assert meta["data_scope"] == "real"
    assert meta["algorithm"] == "catboost"
    assert meta["group"] == "TEST"
    assert meta["model_version"]


def test_bundle_handles_missing_columns_and_new_categories(data, bundle_dir):
    X, _, _ = data
    new = X.drop(columns="b").astype({"job": object})
    new.loc[0, "job"] = "YENI"
    assert len(load_bundle(bundle_dir).predict(new)) == len(X)


def test_calibration_keeps_ranking(data, bundle_dir):
    X, _, _ = data
    bundle = load_bundle(bundle_dir)
    calibrated = bundle.predict(X)[np.argsort(bundle.predict_raw(X))]
    assert (np.diff(calibrated) >= -1e-12).all()


def test_synthetic_scope_is_recorded(data, tmp_path):
    X, y, _ = data
    path = train_bundle(
        X, y, tmp_path / "b", group="TEST", params=FAST, data_scope="synthetic_demo"
    )
    assert load_bundle(path).metadata["data_scope"] == "synthetic_demo"