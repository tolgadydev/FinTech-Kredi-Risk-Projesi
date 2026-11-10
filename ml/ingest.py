from pathlib import Path

import pandas as pd
from credit_features.schemas import SCHEMAS


def convert_raw(raw_dir: Path, out_dir: Path) -> list[Path]:
    sources = [(name, raw_dir / f"{name}.csv") for name in SCHEMAS]
    sources = [(name, path) for name, path in sources if path.exists()]
    if not sources:
        raise FileNotFoundError(f"{raw_dir} içinde Home Credit CSV dosyası bulunamadı")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, path in sources:
        target = out_dir / f"{name}.parquet"
        SCHEMAS[name].validate(pd.read_csv(path)).to_parquet(target, index=False)
        written.append(target)
    return written