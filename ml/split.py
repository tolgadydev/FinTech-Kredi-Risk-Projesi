import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
SEED = 42


def make_holdout(df: pd.DataFrame, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    dev_ids, test_ids = train_test_split(
        df["SK_ID_CURR"].to_numpy(),
        test_size=TEST_SIZE,
        stratify=df["TARGET"],
        random_state=seed,
    )
    return dev_ids, test_ids