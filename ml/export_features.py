import json
from pathlib import Path

import pandas as pd
from credit_features.behaviour import build_g2

DATA_DIR = Path("data/parquet")
OUTPUT = Path("data/features/behaviour_real.jsonl")


def write_jsonl(frame: pd.DataFrame, out: Path) -> int:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as file:
        for sk_id_curr, row in frame.iterrows():
            features = {
                str(column): None if pd.isna(value) else float(value)
                for column, value in row.items()
            }
            file.write(json.dumps({"sk_id_curr": int(sk_id_curr), "features": features}) + "\n")
    return len(frame)


def main() -> None:
    def read(name: str) -> pd.DataFrame:
        return pd.read_parquet(DATA_DIR / f"{name}.parquet")

    summaries = build_g2(
        read("installments_payments"),
        read("POS_CASH_balance"),
        read("credit_card_balance"),
        read("previous_application"),
    )
    count = write_jsonl(summaries, OUTPUT)
    print(f"{count} kişinin ödeme özeti yazıldı: {OUTPUT}")


if __name__ == "__main__":
    main()