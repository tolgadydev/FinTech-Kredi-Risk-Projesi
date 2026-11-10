import pandas as pd

EXT_SOURCE_COLUMNS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
G3_COLUMNS = [
    "g3_bureau_count",
    "g3_bureau_active_share",
    "g3_bureau_overdue_max",
    "g3_bb_dpd_share",
    *EXT_SOURCE_COLUMNS,
]
DPD_STATUSES = ["1", "2", "3", "4", "5"]


def _bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    ids = bureau["SK_ID_CURR"]
    active = bureau["CREDIT_ACTIVE"].eq("Active").astype(float)
    return pd.DataFrame(
        {
            "g3_bureau_count": active.groupby(ids).size().astype(float),
            "g3_bureau_active_share": active.groupby(ids).mean(),
            "g3_bureau_overdue_max": bureau["AMT_CREDIT_MAX_OVERDUE"].groupby(ids).max(),
        }
    )


def _balance_features(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    owners = bureau[["SK_ID_BUREAU", "SK_ID_CURR"]]
    rows = bureau_balance.merge(owners, on="SK_ID_BUREAU", how="inner")
    late = rows["STATUS"].astype(str).isin(DPD_STATUSES).astype(float)
    return late.groupby(rows["SK_ID_CURR"]).mean().to_frame("g3_bb_dpd_share")


def build_g3(
    application: pd.DataFrame,
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame,
) -> pd.DataFrame:
    out = application.set_index("SK_ID_CURR")[EXT_SOURCE_COLUMNS].copy()
    parts = [_bureau_features(bureau), _balance_features(bureau, bureau_balance)]
    return out.join(parts).reindex(columns=G3_COLUMNS)