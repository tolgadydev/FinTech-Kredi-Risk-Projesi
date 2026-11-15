from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

BEHAVIOUR_PREFIXES = ("g2_", "syn_")


class ApplicationIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    AMT_INCOME_TOTAL: float | None = None
    AMT_CREDIT: float | None = None
    AMT_ANNUITY: float | None = None
    AMT_GOODS_PRICE: float | None = None
    DAYS_EMPLOYED: float | None = None
    REGION_RATING_CLIENT: float | None = None
    NAME_CONTRACT_TYPE: str | None = None
    NAME_INCOME_TYPE: str | None = None
    OCCUPATION_TYPE: str | None = None
    ORGANIZATION_TYPE: str | None = None
    FLAG_OWN_CAR: str | None = None
    FLAG_OWN_REALTY: str | None = None


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["A", "B"] = "A"
    application: ApplicationIn = Field(default_factory=ApplicationIn)
    behaviour: dict[str, float | None] = Field(default_factory=dict)

    @field_validator("behaviour")
    @classmethod
    def only_behaviour_features(cls, value: dict) -> dict:
        unknown = [name for name in value if not name.startswith(BEHAVIOUR_PREFIXES)]
        if unknown:
            raise ValueError(f"İzin verilmeyen davranış alanları: {unknown}")
        return value


class ReasonOut(BaseModel):
    feature: str
    text: str
    contribution: float
    increases_risk: bool


class ScoreResponse(BaseModel):
    probability: float
    raw_probability: float
    model_version: str
    data_scope: str
    features: dict[str, float | str | None]
    reasons: list[ReasonOut]