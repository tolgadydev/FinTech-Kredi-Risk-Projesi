import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from credit_features.frame import FeatureSpec, align_frame


@dataclass
class Bundle:
    model: object
    calibrator: object
    spec: FeatureSpec
    metadata: dict

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(align_frame(X, self.spec))[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict(self.predict_raw(X))


def load_bundle(path: Path) -> Bundle:
    path = Path(path)
    spec = FeatureSpec(**json.loads((path / "features.json").read_text(encoding="utf-8")))
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return Bundle(
        model=joblib.load(path / "model.joblib"),
        calibrator=joblib.load(path / "calibrator.joblib"),
        spec=spec,
        metadata=metadata,
    )