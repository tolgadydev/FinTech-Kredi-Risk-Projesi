import numpy as np
import pandas as pd

EPSILON = 1e-4
LEVELS = [(0.10, "stabil"), (0.25, "izlenmeli"), (np.inf, "önemli kayma")]


def _numeric_shares(expected: pd.Series, actual: pd.Series, bins: int):
    clean = expected.dropna().to_numpy(dtype=float)
    if len(clean):
        edges = np.unique(np.quantile(clean, np.linspace(0, 1, bins + 1)))
        inner = edges[1:-1]
    else:
        inner = np.array([])

    def shares(values: pd.Series) -> np.ndarray:
        values = values.to_numpy(dtype=float)
        missing = np.isnan(values)
        idx = np.searchsorted(inner, values[~missing], side="right")
        counts = np.bincount(idx, minlength=len(inner) + 1).astype(float)
        return np.append(counts, missing.sum()) / len(values)

    return shares(expected), shares(actual)


def _category_shares(expected: pd.Series, actual: pd.Series):
    def labelled(values: pd.Series) -> pd.Series:
        values = values.astype(object)
        return values.where(values.notna(), "__eksik__")

    expected, actual = labelled(expected), labelled(actual)
    categories = sorted(set(expected) | set(actual), key=str)

    def shares(values: pd.Series) -> np.ndarray:
        counts = values.value_counts(normalize=True)
        return counts.reindex(categories, fill_value=0).to_numpy(dtype=float)

    return shares(expected), shares(actual)


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    if pd.api.types.is_numeric_dtype(expected):
        before, after = _numeric_shares(expected, actual, bins)
    else:
        before, after = _category_shares(expected, actual)
    before = np.clip(before, EPSILON, None)
    after = np.clip(after, EPSILON, None)
    return float(np.sum((after - before) * np.log(after / before)))


def _level(value: float) -> str:
    return next(label for limit, label in LEVELS if value < limit)


def psi_table(reference: pd.DataFrame, current: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    values = {column: psi(reference[column], current[column], bins) for column in reference.columns}
    table = pd.Series(values, name="psi").sort_values(ascending=False).to_frame()
    table["level"] = table["psi"].map(_level)
    return table