from dataclasses import dataclass

import pandas as pd

MISSING = "MISSING"


@dataclass(frozen=True)
class FeatureSpec:
    numeric: list[str]
    categorical: dict[str, list[str]]

    @property
    def columns(self) -> list[str]:
        return self.numeric + list(self.categorical)


def fit_spec(df: pd.DataFrame) -> FeatureSpec:
    numeric = df.select_dtypes("number").columns.tolist()
    categorical = {}
    for column in df.columns.difference(numeric, sort=False):
        levels = {str(value) for value in df[column].dropna().unique()}
        categorical[column] = sorted(levels | {MISSING})
    return FeatureSpec(numeric=numeric, categorical=categorical)


def align_frame(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    out = df.reindex(columns=spec.columns).copy()
    for column in spec.numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    for column, levels in spec.categorical.items():
        values = out[column].astype(object).where(out[column].isin(levels), MISSING)
        out[column] = pd.Categorical(values, categories=levels)
    return out