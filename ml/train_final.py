import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from credit_features.bundle import Bundle, load_bundle
from credit_features.frame import align_frame, fit_spec
from sklearn.isotonic import IsotonicRegression

from ml.cv import cross_val_oof
from ml.metrics import evaluate
from ml.models import make_model

__all__ = ["Bundle", "fit_calibrator", "load_bundle", "train_bundle"]


def fit_calibrator(oof, y) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    return calibrator.fit(np.asarray(oof), np.asarray(y))


def train_bundle(
    X_raw: pd.DataFrame,
    y: pd.Series,
    out_dir: Path,
    group: str,
    params: dict | None = None,
    data_scope: str = "real",
    model_name: str = "catboost",
    arm: str = "none",
) -> Path:
    spec = fit_spec(X_raw)
    X = align_frame(X_raw, spec)

    cv = cross_val_oof(lambda: make_model(model_name, arm, spec, params), X, y)
    calibrator = fit_calibrator(cv.oof, y)
    model = make_model(model_name, arm, spec, params).fit(X, y)

    metadata = {
        "model_version": f"{group.lower()}-{model_name}-{datetime.now(UTC):%Y%m%d%H%M%S}",
        "data_scope": data_scope,
        "algorithm": model_name,
        "arm": arm,
        "group": group,
        "params": params or {},
        "n_train": int(len(y)),
        "oof_metrics": evaluate(np.asarray(y), cv.oof),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(calibrator, out_dir / "calibrator.joblib")
    (out_dir / "features.json").write_text(json.dumps(asdict(spec), indent=2), encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir