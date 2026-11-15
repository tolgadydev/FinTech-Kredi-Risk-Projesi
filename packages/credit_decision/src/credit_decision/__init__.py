from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

Band = Literal["approve", "manual_review", "reject"]


@dataclass(frozen=True)
class Thresholds:
    approve: float
    reject: float


def thresholds_from(margin: float, lgd: float) -> Thresholds:
    if margin <= 0 or lgd <= 0:
        raise ValueError(f"Marj ve kayıp pozitif olmalı: margin={margin}, lgd={lgd}")
    reject = margin / (margin + lgd)
    return Thresholds(approve=reject / 2, reject=reject)


def load_thresholds(path: Path) -> Thresholds:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return thresholds_from(config["margin"], config["lgd"])


def assign_band(pd_value: float, thresholds: Thresholds) -> Band:
    if not 0 <= pd_value <= 1:
        raise ValueError(f"Olasılık 0 ile 1 arasında olmalı: {pd_value}")
    if pd_value < thresholds.approve:
        return "approve"
    if pd_value < thresholds.reject:
        return "manual_review"
    return "reject"