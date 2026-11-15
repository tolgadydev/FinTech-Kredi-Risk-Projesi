import json
from pathlib import Path

import pandas as pd

from ml.dataset import build_dataset, thin_file_mask
from ml.experiments.feature_comparison import DATA_DIR, SPLIT_FILE
from ml.final_eval import final_evaluate

MARKER = Path("results/holdout_used.json")
OUTPUT = Path("results/final_evaluation.json")


def main() -> None:
    split = pd.read_parquet(SPLIT_FILE)
    test_ids = split.loc[split["split"] == "test", "SK_ID_CURR"].to_numpy()

    X_all, y_all = build_dataset("G1G2", DATA_DIR)
    X_test, y_test = X_all.loc[test_ids], y_all.loc[test_ids]
    bureau = pd.read_parquet(DATA_DIR / "bureau.parquet", columns=["SK_ID_CURR"])
    thin = thin_file_mask(X_test.index, bureau).to_numpy()

    print(f"Sınav: {len(test_ids)} kişi, banka kaydı olmayan {int(thin.sum())}", flush=True)
    report = final_evaluate(
        Path("models/model_a"), Path("models/compare_g1"),
        X_test, y_test, thin, marker=MARKER,
    )
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    c2, c3 = report["criterion_2"], report["criterion_3"]
    print(f"Herkes        G1G2 roc_auc {report['all']['roc_auc']:.4f}  G1 {report['g1_all']['roc_auc']:.4f}")
    print(
        f"Kaydı olmayan G1G2 roc_auc {report['thin_file']['roc_auc']:.4f}  "
        f"G1 {report['g1_thin_file']['roc_auc']:.4f}"
    )
    print(f"2. ölçüt: fark %95 aralığı {c2['auc_gain_ci95']}, geçti: {c2['passed']}")
    print(
        f"3. ölçüt: Brier ham {c3['brier_raw']:.4f}, kalibrasyonlu {c3['brier_calibrated']:.4f}, "
        f"geçti: {c3['passed']}"
    )
    print(f"Sonuçlar yazıldı: {OUTPUT}")


if __name__ == "__main__":
    main()