from catboost import CatBoostClassifier
from credit_features.frame import FeatureSpec
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from optbinning import BinningProcess
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

SEED = 42
MODEL_NAMES = ("lightgbm", "catboost")
ARMS = ("none", "class_weight", "smotenc")
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


def _booster(name: str, arm: str, spec: FeatureSpec, settings: dict, with_categories: bool):
    if name == "lightgbm":
        weight = {"class_weight": "balanced"} if arm == "class_weight" else {}
        return LGBMClassifier(random_state=SEED, verbose=-1, **weight, **settings)

    weight = {"auto_class_weights": "Balanced"} if arm == "class_weight" else {}
    return CatBoostClassifier(
        random_seed=SEED,
        verbose=0,
        allow_writing_files=False,
        cat_features=list(spec.categorical) if with_categories else None,
        **weight,
        **settings,
    )


def _smote_pipeline(model, spec: FeatureSpec) -> ImbPipeline:
    prep = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), spec.numeric),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                list(spec.categorical),
            ),
        ]
    )
    categorical_positions = list(range(len(spec.numeric), len(spec.columns)))
    return ImbPipeline(
        [
            ("prep", prep),
            ("smote", SMOTENC(categorical_features=categorical_positions, random_state=SEED)),
            ("model", model),
        ]
    )


def make_model(name: str, arm: str, spec: FeatureSpec, params: dict | None = None):
    if name not in MODEL_NAMES:
        raise ValueError(f"Bilinmeyen model: {name}. Seçenekler: {MODEL_NAMES}")
    if arm not in ARMS:
        raise ValueError(f"Bilinmeyen yöntem: {arm}. Seçenekler: {ARMS}")

    settings = {**DEFAULT_PARAMS[name], **(params or {})}
    if arm == "smotenc":
        return _smote_pipeline(_booster(name, arm, spec, settings, with_categories=False), spec)
    return _booster(name, arm, spec, settings, with_categories=True)