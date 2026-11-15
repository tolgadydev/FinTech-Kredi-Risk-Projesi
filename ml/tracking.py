import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT = "kredi-risk"


def tracking_uri(store: Path) -> str:
    return f"sqlite:///{(Path(store).resolve() / 'mlflow.db').as_posix()}"


def _experiment_id(client: MlflowClient, store: Path, name: str) -> str:
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    artifacts = (Path(store).resolve() / "artifacts").as_uri()
    return client.create_experiment(name, artifact_location=artifacts)


def log_bundle(
    bundle_dir: Path,
    store: Path,
    test_report: dict | None = None,
    experiment: str = EXPERIMENT,
) -> str:
    bundle_dir = Path(bundle_dir)
    meta = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))
    if test_report is not None and meta["data_scope"] != "real":
        raise ValueError(f"data_scope={meta['data_scope']} olan model için sınav puanı kaydedilemez")

    Path(store).mkdir(parents=True, exist_ok=True)
    uri = tracking_uri(store)
    client = MlflowClient(uri)
    experiment_id = _experiment_id(client, store, experiment)
    mlflow.set_tracking_uri(uri)

    with mlflow.start_run(experiment_id=experiment_id, run_name=meta["model_version"]) as run:
        mlflow.set_tags(
            {
                "data_scope": meta["data_scope"],
                "group": meta["group"],
                "model_version": meta["model_version"],
            }
        )
        mlflow.log_params(
            {
                "algorithm": meta["algorithm"],
                "arm": meta["arm"],
                "n_train": meta["n_train"],
                **{f"param_{key}": value for key, value in meta["params"].items()},
            }
        )
        mlflow.log_metrics({f"oof_{k}": v for k, v in meta["oof_metrics"].items()})
        if test_report is not None:
            mlflow.log_metrics({f"test_{k}": v for k, v in test_report["all"].items()})
            mlflow.log_metrics({f"test_thin_{k}": v for k, v in test_report["thin_file"].items()})
        mlflow.log_artifacts(str(bundle_dir), artifact_path="bundle")
        return run.info.run_id