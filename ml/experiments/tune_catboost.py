import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from credit_features.frame import align_frame, fit_spec

from ml.cv import cross_val_oof
from ml.dataset import build_dataset
from ml.experiments.feature_comparison import DATA_DIR, load_dev_ids
from ml.metrics import evaluate
from ml.models import make_model
from ml.selection import calibrated_brier
from ml.tune import tune

GROUP = "G1G2"
MODEL = "catboost"
ARM = "none"
MIN_GAIN = 0.002
RESULTS_DIR = Path("results")


def _sample(ids: np.ndarray, frac: float) -> np.ndarray:
    if frac >= 1:
        return ids
    return pd.Series(ids).sample(frac=frac, random_state=42).to_numpy()


def _validate(X: pd.DataFrame, y: pd.Series, spec, params: dict | None) -> dict:
    cv = cross_val_oof(lambda: make_model(MODEL, ARM, spec, params), X, y)
    return {
        **evaluate(y.to_numpy(), cv.oof),
        "brier_calibrated": calibrated_brier(y.to_numpy(), cv.oof),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-frac", type=float, default=0.2)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--validate-frac", type=float, default=1.0)
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.INFO)

    dev_ids = load_dev_ids()
    X_all, y_all = build_dataset(GROUP, DATA_DIR)
    spec = fit_spec(X_all.loc[dev_ids])
    X_dev = align_frame(X_all.loc[dev_ids], spec)
    y_dev = y_all.loc[dev_ids]

    search_ids = _sample(dev_ids, args.search_frac)
    print(f"Arama: {len(search_ids)} kişi, {args.n_trials} deneme", flush=True)
    start = time.time()
    params, search_score = tune(
        MODEL, ARM, X_dev.loc[search_ids], y_dev.loc[search_ids], spec,
        n_trials=args.n_trials, n_splits=3,
    )
    print(f"En iyi ayarlar: {params} (arama PR-AUC {search_score:.4f})", flush=True)
    print(f"Arama süresi: {(time.time() - start) / 60:.1f} dakika", flush=True)

    val_ids = _sample(dev_ids, args.validate_frac)
    X_val, y_val = X_dev.loc[val_ids], y_dev.loc[val_ids]
    print("Varsayılan ayarlar doğrulanıyor...", flush=True)
    default = _validate(X_val, y_val, spec, None)
    print("Bulunan ayarlar doğrulanıyor...", flush=True)
    tuned = _validate(X_val, y_val, spec, params)

    gain = tuned["pr_auc"] - default["pr_auc"]
    chosen = "tuned" if gain > MIN_GAIN else "default"
    print(f"Varsayılan PR-AUC {default['pr_auc']:.4f}, bulunan {tuned['pr_auc']:.4f}, fark {gain:+.4f}")
    print(f"Seçilen: {chosen}")

    output = {
        "model": MODEL,
        "arm": ARM,
        "group": GROUP,
        "search": {
            "sample_frac": args.search_frac,
            "n_trials": args.n_trials,
            "pr_auc": search_score,
            "params": params,
        },
        "validation": {"sample_frac": args.validate_frac, "default": default, "tuned": tuned},
        "pr_auc_gain": gain,
        "chosen": chosen,
        "params": params if chosen == "tuned" else {},
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    name = "best_params.json" if args.validate_frac == 1 else "best_params_sample.json"
    (RESULTS_DIR / name).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Sonuçlar yazıldı: {RESULTS_DIR / name}")


if __name__ == "__main__":
    main()