import json
from pathlib import Path

import numpy as np
import pandas as pd
from credit_decision import load_thresholds, thresholds_from
from credit_features.frame import align_frame

from ml.cv import cross_val_oof
from ml.dataset import build_dataset, build_features
from ml.drift import psi, psi_table
from ml.experiments.feature_comparison import DATA_DIR, load_dev_ids
from ml.fairness import age_groups, assign_bands, band_shares, fairness_report
from ml.models import make_model
from ml.train_final import load_bundle

BUNDLE = Path("models/model_a")
CONFIG = Path("config/decision.yaml")
OUTPUT = Path("results/decision_analysis.json")
MARGINS = (0.10, 0.15, 0.20)
LGDS = (0.35, 0.45, 0.60)


def _records(frame: pd.DataFrame) -> list[dict]:
    table = frame.reset_index().astype(object)
    return table.where(table.notna(), None).to_dict(orient="records")


def approved_default_rate(y: np.ndarray, bands: np.ndarray) -> float | None:
    approved = bands == "approve"
    return float(y[approved].mean()) if approved.any() else None


def sensitivity(probs: np.ndarray, y: np.ndarray) -> list[dict]:
    rows = []
    for margin in MARGINS:
        for lgd in LGDS:
            t = thresholds_from(margin, lgd)
            rows.append(
                {
                    "margin": margin,
                    "lgd": lgd,
                    "approve_below": t.approve,
                    "reject_from": t.reject,
                    **band_shares(probs, t),
                    "approved_default_rate": approved_default_rate(y, assign_bands(probs, t)),
                }
            )
    return rows


def main() -> None:
    bundle = load_bundle(BUNDLE)
    meta = bundle.metadata
    dev_ids = load_dev_ids()

    X_all, y_all = build_dataset("G1G2", DATA_DIR)
    X_raw = X_all.loc[dev_ids]
    y = y_all.loc[dev_ids]
    X = align_frame(X_raw, bundle.spec)

    print("OOF tahminleri yeniden üretiliyor...", flush=True)
    cv = cross_val_oof(
        lambda: make_model(meta["algorithm"], meta["arm"], bundle.spec, meta["params"] or None),
        X,
        y,
    )
    y = y.to_numpy()
    probs = bundle.calibrator.predict(cv.oof)
    thresholds = load_thresholds(CONFIG)
    bands = assign_bands(probs, thresholds)

    people = (
        pd.read_parquet(
            DATA_DIR / "application_train.parquet",
            columns=["SK_ID_CURR", "CODE_GENDER", "DAYS_BIRTH"],
        )
        .set_index("SK_ID_CURR")
        .loc[dev_ids]
    )
    gender = fairness_report(y, bands, people["CODE_GENDER"])
    age = fairness_report(y, bands, age_groups(people["DAYS_BIRTH"]))

    print("Kayma hesaplanıyor...", flush=True)
    current_raw = build_features("G1G2", DATA_DIR, "application_test")
    drift = psi_table(X_raw, current_raw)
    score_psi = psi(pd.Series(cv.oof), pd.Series(bundle.predict_raw(current_raw)))

    output = {
        "model_version": meta["model_version"],
        "n_dev": int(len(y)),
        "note": "OOF tahminleri kullanıldı. Kalibrasyon aynı tahminlerle öğrenildiği için oranlar hafif iyimser olabilir.",
        "thresholds": {"approve_below": thresholds.approve, "reject_from": thresholds.reject},
        "band_shares": band_shares(probs, thresholds),
        "approved_default_rate": approved_default_rate(y, bands),
        "sensitivity": sensitivity(probs, y),
        "fairness": {
            "gender": _records(gender),
            "age": _records(age),
        },
        "drift": {
            "score_psi": score_psi,
            "features": _records(drift.rename_axis("feature")),
        },
    }
    OUTPUT.write_text(
        json.dumps(output, indent=2, default=float, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nKarar dağılımı: {output['band_shares']}")
    print(f"Onaylananların ödememe oranı: {output['approved_default_rate']:.4f}")
    print("\nCinsiyete göre:\n", gender.round(3))
    print("\nYaşa göre:\n", age.round(3))
    print(f"\nSkor PSI: {score_psi:.4f}")
    print("\nEn çok kayan 5 değişken:\n", drift.head(5).round(4))
    print(f"\nSonuçlar yazıldı: {OUTPUT}")


if __name__ == "__main__":
    main()