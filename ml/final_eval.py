import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from ml.metrics import evaluate
from ml.stats import bootstrap_auc_diff
from ml.train_final import load_bundle


def final_evaluate(
    main_dir: Path,
    compare_dir: Path,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thin: np.ndarray,
    marker: Path,
    n_boot: int = 2000,
) -> dict:
    marker = Path(marker)
    if marker.exists():
        raise RuntimeError(f"Test seti (holdout) zaten kullanıldı: {marker}")

    main = load_bundle(main_dir)
    compare = load_bundle(compare_dir)
    for bundle in (main, compare):
        scope = bundle.metadata["data_scope"]
        if scope != "real":
            raise ValueError(f"data_scope={scope} olan model sınava giremez, yalnızca 'real'")

    y = np.asarray(y_test)
    thin = np.asarray(thin, dtype=bool)
    raw = main.predict_raw(X_test)
    calibrated = main.predict(X_test)
    g1_raw = compare.predict_raw(X_test)

    low, high = bootstrap_auc_diff(y[thin], raw[thin], g1_raw[thin], n=n_boot)
    brier_raw = float(brier_score_loss(y, raw))
    brier_calibrated = float(brier_score_loss(y, calibrated))

    report = {
        "model_version": main.metadata["model_version"],
        "compare_version": compare.metadata["model_version"],
        "n_test": int(len(y)),
        "n_thin_file": int(thin.sum()),
        "all": evaluate(y, raw),
        "thin_file": evaluate(y[thin], raw[thin]),
        "g1_all": evaluate(y, g1_raw),
        "g1_thin_file": evaluate(y[thin], g1_raw[thin]),
        "criterion_2": {"auc_gain_ci95": [low, high], "passed": bool(low > 0)},
        "criterion_3": {
            "brier_raw": brier_raw,
            "brier_calibrated": brier_calibrated,
            "passed": bool(brier_calibrated <= brier_raw),
        },
    }

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {"used_at": datetime.now(UTC).isoformat(), "model_version": report["model_version"]}
        ),
        encoding="utf-8",
    )
    return report