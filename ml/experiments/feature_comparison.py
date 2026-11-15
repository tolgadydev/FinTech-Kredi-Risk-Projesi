import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from credit_features.frame import align_frame, fit_spec

from ml.cv import cross_val_oof
from ml.dataset import GROUPS, build_dataset, thin_file_mask
from ml.metrics import evaluate
from ml.models import make_baseline, make_model
from ml.split import make_holdout

DATA_DIR = Path("data/parquet")
SPLIT_FILE = Path("data/splits/holdout.parquet")
RESULTS_DIR = Path("results")
MODEL_NAMES = ("logreg_woe", "lightgbm", "catboost")


def load_dev_ids() -> np.ndarray:
    if not SPLIT_FILE.exists():
        app = pd.read_parquet(
            DATA_DIR / "application_train.parquet", columns=["SK_ID_CURR", "TARGET"]
        )
        dev_ids, test_ids = make_holdout(app)
        SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(
            [
                pd.DataFrame({"SK_ID_CURR": dev_ids, "split": "dev"}),
                pd.DataFrame({"SK_ID_CURR": test_ids, "split": "test"}),
            ]
        ).to_parquet(SPLIT_FILE, index=False)
    split = pd.read_parquet(SPLIT_FILE)
    return split.loc[split["split"] == "dev", "SK_ID_CURR"].to_numpy()


def summarize(y: np.ndarray, oof: np.ndarray, thin: np.ndarray, folds: list[dict]) -> dict:
    return {
        "cv_mean": {key: float(np.mean([f[key] for f in folds])) for key in folds[0]},
        "cv_std_roc_auc": float(np.std([f["roc_auc"] for f in folds])),
        "oof_all": evaluate(y, oof),
        "oof_thin_file": evaluate(y[thin], oof[thin]),
        "thin_file_count": int(thin.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-frac", type=float, default=1.0)
    args = parser.parse_args()

    dev_ids = load_dev_ids()
    if args.sample_frac < 1:
        dev_ids = pd.Series(dev_ids).sample(frac=args.sample_frac, random_state=42).to_numpy()
    bureau = pd.read_parquet(DATA_DIR / "bureau.parquet", columns=["SK_ID_CURR"])

    results = {}
    for group in GROUPS:
        X_all, y_all = build_dataset(group, DATA_DIR)
        X_raw, y = X_all.loc[dev_ids], y_all.loc[dev_ids]
        spec = fit_spec(X_raw)
        X = align_frame(X_raw, spec)
        thin = thin_file_mask(X.index, bureau).to_numpy()

        factories = {
            "logreg_woe": lambda: make_baseline(spec),
            "lightgbm": lambda: make_model("lightgbm", "none", spec),
            "catboost": lambda: make_model("catboost", "none", spec),
        }
        for name in MODEL_NAMES:
            print(f"{group} / {name} çalışıyor...", flush=True)
            cv = cross_val_oof(factories[name], X, y)
            results[f"{group}/{name}"] = summarize(y.to_numpy(), cv.oof, thin, cv.fold_metrics)
            print(f"  roc_auc: {results[f'{group}/{name}']['cv_mean']['roc_auc']:.4f}", flush=True)

    best = max(results[f"G1G2/{m}"]["cv_mean"]["roc_auc"] for m in ("lightgbm", "catboost"))
    gain = best - results["G1G2/logreg_woe"]["cv_mean"]["roc_auc"]
    output = {
        "sample_frac": args.sample_frac,
        "n_dev": int(len(dev_ids)),
        "criterion_1_gain": float(gain),
        "criterion_1_passed": bool(gain >= 0.01),
        "results": results,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    name = "feature_comparison.json" if args.sample_frac == 1 else "feature_comparison_sample.json"
    (RESULTS_DIR / name).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Sonuçlar yazıldı: {RESULTS_DIR / name}")


if __name__ == "__main__":
    main()