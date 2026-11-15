import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from credit_features.frame import align_frame, fit_spec

from ml.cv import cross_val_oof
from ml.dataset import build_dataset
from ml.experiments.feature_comparison import DATA_DIR, load_dev_ids
from ml.metrics import evaluate
from ml.models import ARMS, MODEL_NAMES, make_model
from ml.selection import calibrated_brier, select_best

GROUP = "G1G2"
RESULTS_DIR = Path("results")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-frac", type=float, default=1.0)
    args = parser.parse_args()

    dev_ids = load_dev_ids()
    if args.sample_frac < 1:
        dev_ids = pd.Series(dev_ids).sample(frac=args.sample_frac, random_state=42).to_numpy()

    X_all, y_all = build_dataset(GROUP, DATA_DIR)
    X_raw, y = X_all.loc[dev_ids], y_all.loc[dev_ids]
    spec = fit_spec(X_raw)
    X = align_frame(X_raw, spec)

    rows = []
    for name in MODEL_NAMES:
        for arm in ARMS:
            print(f"{name} / {arm} çalışıyor...", flush=True)
            cv = cross_val_oof(lambda: make_model(name, arm, spec), X, y)
            row = {
                "id": f"{name}/{arm}",
                "model": name,
                "arm": arm,
                **evaluate(y.to_numpy(), cv.oof),
                "brier_calibrated": calibrated_brier(y.to_numpy(), cv.oof),
                "cv_std_roc_auc": float(np.std([f["roc_auc"] for f in cv.fold_metrics])),
                "mean_prob": float(cv.oof.mean()),
            }
            rows.append(row)
            print(
                f"  pr_auc {row['pr_auc']:.4f}  roc_auc {row['roc_auc']:.4f}  "
                f"brier_cal {row['brier_calibrated']:.4f}  ort.risk {row['mean_prob']:.3f}",
                flush=True,
            )

    winner = select_best(rows)
    output = {
        "sample_frac": args.sample_frac,
        "n_dev": int(len(dev_ids)),
        "group": GROUP,
        "actual_rate": float(y.mean()),
        "winner": winner["id"],
        "results": rows,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    name = "imbalance_comparison.json" if args.sample_frac == 1 else "imbalance_comparison_sample.json"
    (RESULTS_DIR / name).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Kazanan: {winner['id']}")
    print(f"Sonuçlar yazıldı: {RESULTS_DIR / name}")


if __name__ == "__main__":
    main()