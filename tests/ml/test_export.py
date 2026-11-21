import json

import numpy as np
import pandas as pd

from ml.export_features import write_jsonl


def _summaries() -> pd.DataFrame:
    return pd.DataFrame(
        {"g2_inst_late_share": [0.1, np.nan], "g2_prev_count": [3.0, 1.0]},
        index=pd.Index([100001, 100002], name="SK_ID_CURR"),
    )


def test_one_line_per_person(tmp_path):
    out = tmp_path / "behaviour.jsonl"
    assert write_jsonl(_summaries(), out) == 2
    lines = out.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    assert first == {"sk_id_curr": 100001, "features": {"g2_inst_late_share": 0.1, "g2_prev_count": 3.0}}


def test_missing_values_become_null(tmp_path):
    out = tmp_path / "behaviour.jsonl"
    write_jsonl(_summaries(), out)
    text = out.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert json.loads(text.splitlines()[1])["features"]["g2_inst_late_share"] is None


def test_output_folder_is_created(tmp_path):
    out = tmp_path / "yeni" / "klasor" / "behaviour.jsonl"
    write_jsonl(_summaries(), out)
    assert out.exists()