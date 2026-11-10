import pandera.pandas as pa


def _numeric(*names: str) -> dict[str, pa.Column]:
    return {name: pa.Column(float, nullable=True, coerce=True) for name in names}


def _present(*names: str) -> dict[str, pa.Column]:
    return {name: pa.Column(nullable=True) for name in names}


def _key(name: str, unique: bool = False) -> dict[str, pa.Column]:
    return {name: pa.Column(int, nullable=False, unique=unique, coerce=True)}


_APPLICATION = {
    **_key("SK_ID_CURR", unique=True),
    **_numeric(
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_EMPLOYED",
        "REGION_RATING_CLIENT",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ),
    **_present(
        "NAME_CONTRACT_TYPE",
        "NAME_INCOME_TYPE",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "CODE_GENDER",
        "DAYS_BIRTH",
    ),
}

SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "application_train": pa.DataFrameSchema(
        {**_APPLICATION, "TARGET": pa.Column(int, pa.Check.isin([0, 1]), coerce=True)}
    ),
    "application_test": pa.DataFrameSchema(_APPLICATION),
    "bureau": pa.DataFrameSchema(
        {
            **_key("SK_ID_CURR"),
            **_key("SK_ID_BUREAU"),
            **_numeric("AMT_CREDIT_MAX_OVERDUE"),
            **_present("CREDIT_ACTIVE"),
        }
    ),
    "bureau_balance": pa.DataFrameSchema(
        {**_key("SK_ID_BUREAU"), **_numeric("MONTHS_BALANCE"), **_present("STATUS")}
    ),
    "previous_application": pa.DataFrameSchema(
        {**_key("SK_ID_CURR"), **_present("NAME_CONTRACT_STATUS")}
    ),
    "POS_CASH_balance": pa.DataFrameSchema({**_key("SK_ID_CURR"), **_numeric("SK_DPD")}),
    "installments_payments": pa.DataFrameSchema(
        {
            **_key("SK_ID_CURR"),
            **_numeric("DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"),
        }
    ),
    "credit_card_balance": pa.DataFrameSchema(
        {**_key("SK_ID_CURR"), **_numeric("AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL")}
    ),
}