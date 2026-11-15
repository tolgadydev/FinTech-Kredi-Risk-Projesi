import json
from pathlib import Path

from ml.tracking import log_bundle

STORE = Path("mlruns")
REPORT = Path("results/final_evaluation.json")


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    targets = [
        (Path("models/model_a"), {"all": report["all"], "thin_file": report["thin_file"]}),
        (
            Path("models/compare_g1"),
            {"all": report["g1_all"], "thin_file": report["g1_thin_file"]},
        ),
    ]
    for bundle, test_report in targets:
        run_id = log_bundle(bundle, STORE, test_report=test_report)
        print(f"{bundle} kaydedildi, run {run_id}")


if __name__ == "__main__":
    main()