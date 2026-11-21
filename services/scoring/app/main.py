import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from credit_features.bundle import Bundle, load_bundle
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.features import request_to_frame
from app.reasons import load_reason_texts, shap_contributions, top_reasons
from app.schemas import ReasonOut, ScoreRequest, ScoreResponse

ROOT = Path(__file__).resolve().parents[3]


def _check(bundle: Bundle) -> None:
    trained_on = list(getattr(bundle.model, "feature_names_", bundle.spec.columns))
    if trained_on != bundle.spec.columns:
        raise ValueError("features.json, modelin eğitildiği kolonlarla uyuşmuyor")


def _feature_value(value):
    if isinstance(value, str):
        return value
    return None if pd.isna(value) else float(value)


def create_app(model_dirs: dict[str, Path], reason_file: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundles = {}
        app.state.errors = {}
        for name, path in model_dirs.items():
            try:
                bundle = load_bundle(path)
                _check(bundle)
                app.state.bundles[name] = bundle
            except Exception as error:  # noqa: BLE001
                app.state.errors[name] = f"{type(error).__name__}: {error}"
        app.state.texts = load_reason_texts(reason_file)
        yield

    app = FastAPI(title="Kredi Risk Skorlama", lifespan=lifespan)

    @app.get("/health")
    def health() -> JSONResponse:
        bundles = app.state.bundles
        body = {
            "models": {name: b.metadata["model_version"] for name, b in bundles.items()},
            "errors": app.state.errors,
        }
        return JSONResponse(body, status_code=200 if "A" in bundles else 503)

    @app.post("/score", response_model=ScoreResponse)
    def score(request: ScoreRequest) -> ScoreResponse:
        bundle = app.state.bundles.get(request.model)
        if bundle is None:
            detail = app.state.errors.get(request.model, f"Model {request.model} yüklü değil")
            raise HTTPException(status_code=503, detail=detail)

        X = request_to_frame(request, bundle.spec)
        raw = float(bundle.model.predict_proba(X)[0, 1])
        probability = float(bundle.calibrator.predict([raw])[0])
        contributions = shap_contributions(bundle.model, X, list(bundle.spec.categorical))[0]
        reasons = top_reasons(contributions, app.state.texts)

        return ScoreResponse(
            probability=probability,
            raw_probability=raw,
            model_version=bundle.metadata["model_version"],
            data_scope=bundle.metadata["data_scope"],
            features={str(column): _feature_value(v) for column, v in X.iloc[0].items()},
            reasons=[ReasonOut(**asdict(reason)) for reason in reasons],
        )

    return app


def _default_model_dirs() -> dict[str, Path]:
    return {
        "A": Path(os.environ.get("MODEL_A_DIR", ROOT / "models" / "model_a")),
        "B": Path(os.environ.get("MODEL_B_DIR", ROOT / "models" / "model_b")),
    }


app = create_app(
    _default_model_dirs(),
    Path(os.environ.get("REASON_CODES", ROOT / "config" / "reason_codes.yaml")),
)