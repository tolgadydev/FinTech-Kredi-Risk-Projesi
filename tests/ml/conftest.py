import numpy as np
import pandas as pd
import pytest
from credit_features.frame import align_frame, fit_spec


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 600
    raw = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": np.where(rng.random(n) < 0.3, np.nan, rng.normal(size=n)),
            "job": rng.choice(["x", "y", "z", None], size=n),
        }
    )
    signal = raw["a"] + (raw["job"] == "x") + rng.normal(scale=0.5, size=n)
    y = pd.Series((signal > 1).astype(int))
    spec = fit_spec(raw)
    return align_frame(raw, spec), y, spec