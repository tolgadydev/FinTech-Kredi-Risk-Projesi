from pathlib import Path

import pandas as pd
from credit_features.frame import align_frame, fit_spec

from ml.dataset import build_dataset, build_features
from ml.experiments.feature_comparison import DATA_DIR, load_dev_ids
from ml.models import make_model

GROUP = "G1G2G3"
MODEL = "catboost"
OUTPUT = Path("results/kaggle_submission.csv")


def main() -> None:
    dev_ids = load_dev_ids()
    X_all, y_all = build_dataset(GROUP, DATA_DIR)
    X_dev = X_all.loc[dev_ids]
    spec = fit_spec(X_dev)

    print(f"{GROUP} / {MODEL} {len(dev_ids)} kişiyle eğitiliyor...", flush=True)
    model = make_model(MODEL, "none", spec)
    model.fit(align_frame(X_dev, spec), y_all.loc[dev_ids])

    X_test = build_features(GROUP, DATA_DIR, "application_test")
    prob = model.predict_proba(align_frame(X_test, spec))[:, 1]

    OUTPUT.parent.mkdir(exist_ok=True)
    pd.DataFrame({"SK_ID_CURR": X_test.index, "TARGET": prob}).to_csv(OUTPUT, index=False)
    print(f"{len(prob)} tahmin yazıldı: {OUTPUT} (ortalama risk {prob.mean():.3f})")


if __name__ == "__main__":
    main()