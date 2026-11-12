from catboost import CatBoostClassifier
from credit_features.frame import FeatureSpec
from lightgbm import LGBMClassifier
from optbinning import BinningProcess
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

SEED = 42
MODEL_NAMES = ("lightgbm", "catboost")
DEFAULT_PARAMS = {
    "lightgbm": {"n_estimators": 500, "learning_rate": 0.05},
    "catboost": {"iterations": 500, "learning_rate": 0.05},
}


def make_baseline(spec: FeatureSpec) -> Pipeline:
    return Pipeline(
        [
            (
                "woe",
                BinningProcess(
                    variable_names=spec.columns,
                    categorical_variables=list(spec.categorical),
                ),
            ),
            ("logreg", LogisticRegression(max_iter=1000)),
        ]
    )


def make_model(name: str, arm: str, spec: FeatureSpec, params: dict | None = None):
    if name not in MODEL_NAMES:
        raise ValueError(f"Bilinmeyen model: {name}. Seçenekler: {MODEL_NAMES}")
    if arm != "none":
        raise NotImplementedError(f"'{arm}' yöntemi Aşama 5'te eklenecek")

    settings = {**DEFAULT_PARAMS[name], **(params or {})}
    if name == "lightgbm":
        return LGBMClassifier(random_state=SEED, verbose=-1, **settings)
    return CatBoostClassifier(
        random_seed=SEED,
        verbose=0,
        allow_writing_files=False,
        cat_features=list(spec.categorical),
        **settings,
    )