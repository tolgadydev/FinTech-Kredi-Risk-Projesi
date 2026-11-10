import numpy as np
import pandas as pd

LAST_YEAR_DAYS = -365

G2_COLUMNS = [
    "g2_inst_late_days_mean",
    "g2_inst_late_share",
    "g2_inst_late_share_last12",
    "g2_inst_underpay_ratio_mean",
    "g2_pos_dpd_max",
    "g2_cc_utilization_mean",
    "g2_prev_count",
    "g2_prev_refused_share",
]


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator > 0, np.nan)


def _installment_features(installments: pd.DataFrame) -> pd.DataFrame:
    late_days = (installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]).clip(lower=0)
    shortfall = (installments["AMT_INSTALMENT"] - installments["AMT_PAYMENT"]).clip(lower=0)
    rows = pd.DataFrame(
        {
            "SK_ID_CURR": installments["SK_ID_CURR"],
            "late_days": late_days,
            "is_late": late_days.gt(0).astype(float).where(late_days.notna()),
            "underpay": _ratio(shortfall, installments["AMT_INSTALMENT"]),
            "recent": installments["DAYS_INSTALMENT"] >= LAST_YEAR_DAYS,
        }
    )
    grouped = rows.groupby("SK_ID_CURR")
    out = pd.DataFrame(
        {
            "g2_inst_late_days_mean": grouped["late_days"].mean(),
            "g2_inst_late_share": grouped["is_late"].mean(),
            "g2_inst_underpay_ratio_mean": grouped["underpay"].mean(),
        }
    )
    out["g2_inst_late_share_last12"] = rows[rows["recent"]].groupby("SK_ID_CURR")["is_late"].mean()
    return out


def _pos_features(pos: pd.DataFrame) -> pd.DataFrame:
    return pos.groupby("SK_ID_CURR")["SK_DPD"].max().to_frame("g2_pos_dpd_max")


def _credit_card_features(credit_card: pd.DataFrame) -> pd.DataFrame:
    utilization = _ratio(credit_card["AMT_BALANCE"], credit_card["AMT_CREDIT_LIMIT_ACTUAL"])
    return utilization.groupby(credit_card["SK_ID_CURR"]).mean().to_frame("g2_cc_utilization_mean")


def _previous_features(previous: pd.DataFrame) -> pd.DataFrame:
    refused = previous["NAME_CONTRACT_STATUS"].eq("Refused").astype(float)
    grouped = refused.groupby(previous["SK_ID_CURR"])
    return pd.DataFrame(
        {"g2_prev_count": grouped.size().astype(float), "g2_prev_refused_share": grouped.mean()}
    )


def build_g2(
    installments: pd.DataFrame,
    pos: pd.DataFrame,
    credit_card: pd.DataFrame,
    previous: pd.DataFrame,
) -> pd.DataFrame:
    parts = [
        _installment_features(installments),
        _pos_features(pos),
        _credit_card_features(credit_card),
        _previous_features(previous),
    ]
    return pd.concat(parts, axis=1).reindex(columns=G2_COLUMNS).rename_axis("SK_ID_CURR")