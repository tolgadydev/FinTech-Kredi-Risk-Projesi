import json
from pathlib import Path

from ml.dataset import build_dataset
from ml.experiments.feature_comparison import DATA_DIR, load_dev_ids
from ml.train_final import train_bundle

MODELS_DIR = Path("models")
BEST_PARAMS = Path("results/best_params.json")
TARGETS = [("G1G2", "model_a"), ("G1", "compare_g1")]


def main() -> None:
    best = json.loads(BEST_PARAMS.read_text(encoding="utf-8"))
    dev_ids = load_dev_ids()

    for group, folder in TARGETS:
        print(f"{group} modeli eğitiliyor ({len(dev_ids)} kişi)...", flush=True)
        X_all, y_all = build_dataset(group, DATA_DIR)
        path = train_bundle(
            X_all.loc[dev_ids],
            y_all.loc[dev_ids],
            MODELS_DIR / folder,
            group=group,
            params=best["params"] or None,
            model_name=best["model"],
            arm=best["arm"],
        )
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        print(
            f"  {meta['model_version']} kaydedildi, OOF roc_auc {meta['oof_metrics']['roc_auc']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()