from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from catboost import Pool


@dataclass(frozen=True)
class Reason:
    feature: str
    text: str
    contribution: float
    increases_risk: bool


def load_reason_texts(path: Path) -> dict[str, str]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def shap_contributions(model, X: pd.DataFrame, cat_features: list[str]) -> list[dict[str, float]]:
    values = model.get_feature_importance(Pool(X, cat_features=cat_features), type="ShapValues")
    return [{column: float(v) for column, v in zip(X.columns, row[:-1])} for row in values]


def _direction(value: float) -> str:
    if value > 0:
        return "artırdı"
    if value < 0:
        return "azalttı"
    return "etkilemedi"


def top_reasons(contributions: dict[str, float], texts: dict[str, str], k: int = 3) -> list[Reason]:
    ordered = sorted(contributions.items(), key=lambda item: item[1], reverse=True)[:k]
    return [
        Reason(
            feature=feature,
            text=f"{texts.get(feature, feature)} riski {_direction(value)}",
            contribution=float(value),
            increases_risk=value > 0,
        )
        for feature, value in ordered
    ]