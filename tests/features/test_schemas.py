import pandas as pd
import pandera
import pytest

from credit_features.schemas import SCHEMAS
from ml.ingest import convert_raw


def test_application_rejects_duplicate_ids(app_df):
    bad = pd.concat([app_df, app_df.head(1)])
    with pytest.raises(pandera.errors.SchemaError):
        SCHEMAS["application_train"].validate(bad)


def test_convert_raw_writes_parquet(tmp_path, raw_csv_dir):
    paths = convert_raw(raw_csv_dir, tmp_path)
    assert all(p.suffix == ".parquet" for p in paths) and paths