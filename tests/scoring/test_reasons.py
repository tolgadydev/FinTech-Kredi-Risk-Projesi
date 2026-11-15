from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from credit_features.application import build_g1
from credit_features.behaviour import G2_COLUMNS
from credit_features.frame import align_frame, fit_spec

from app.reasons import load_reason_texts, shap_contributions, top_reasons

REASON_FILE = Path(__file__).parents[2] / "config" / "reason_codes.yaml"


@pytest.fixture
def fitted():
    rng = np.random.default_rng(0)
    raw = pd.DataFrame(
        {
            "a": rng.normal(size=300),
            "b": np.where(rng.random(300) < 0.3, np.nan, rng.normal(size=300)),
            "job": rng.choice(["x", "y", None], size=300),
        }
    )
    y = ((raw["a"] + (raw["job"] == "x")) > 1).astype(int)
    spec = fit_spec(raw)
    X = align_frame(raw, spec)
    model = CatBoostClassifier(
        iterations=30, random_seed=42, verbose=0, allow_writing_files=False,
        cat_features=list(spec.categorical),
    ).fit(X, y)
    return model, X, spec


def test_top_reasons_sorted_by_risk_impact():
    out = top_reasons({"a": 0.5, "b": -0.2, "c": 1.0, "d": 0.1}, {"c": "Gecikme oranı"})
    assert [r.feature for r in out] == ["c", "a", "d"]
    assert out[0].text == "Gecikme oranı riski artırdı"
    assert out[0].increases_risk


def test_always_three_reasons_even_for_low_risk():
    out = top_reasons({"a": -3.0, "b": -2.0, "c": -1.0, "d": -4.0}, {})
    assert [r.feature for r in out] == ["c", "b", "a"]
    assert not any(r.increases_risk for r in out)
    assert out[0].text == "c riski azalttı"


def test_contributions_fully_explain_predictions(fitted):
    model, X, spec = fitted
    rows = X.iloc[:2]
    contributions = shap_contributions(model, rows, list(spec.categorical))
    assert set(contributions[0]) == set(X.columns)
    prob = model.predict_proba(rows)[:, 1]
    logits = np.log(prob / (1 - prob))
    offsets = [logit - sum(c.values()) for logit, c in zip(logits, contributions)]
    assert offsets[0] == pytest.approx(offsets[1])


def test_every_model_feature_has_a_turkish_label(app_df):
    texts = load_reason_texts(REASON_FILE)
    features = list(build_g1(app_df).columns) + G2_COLUMNS
    assert not [f for f in features if f not in texts]