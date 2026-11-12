from pathlib import Path

import pandas as pd
from credit_features.application import build_g1
from credit_features.behaviour import build_g2
from credit_features.bureau import build_g3

GROUPS = ("G1", "G1G2", "G1G2G3")


def _read(data_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / f"{name}.parquet")


def build_dataset(group: str, data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    if group not in GROUPS:
        raise ValueError(f"Bilinmeyen grup: {group}. Seçenekler: {GROUPS}")

    app = _read(data_dir, "application_train").set_index("SK_ID_CURR")
    X = build_g1(app)

    if "G2" in group:
        X = X.join(
            build_g2(
                _read(data_dir, "installments_payments"),
                _read(data_dir, "POS_CASH_balance"),
                _read(data_dir, "credit_card_balance"),
                _read(data_dir, "previous_application"),
            )
        )
    if "G3" in group:
        X = X.join(
            build_g3(
                app.reset_index(),
                _read(data_dir, "bureau"),
                _read(data_dir, "bureau_balance"),
            )
        )
    return X, app["TARGET"]


def thin_file_mask(index: pd.Index, bureau: pd.DataFrame) -> pd.Series:
    return pd.Series(~index.isin(bureau["SK_ID_CURR"]), index=index, name="thin_file")