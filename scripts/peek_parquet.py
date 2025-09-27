# scripts/peek_parquet.py
import pandas as pd
from pathlib import Path

DF_PATH = Path("data/interim/hourly_uniform.parquet")

def main():
    if not DF_PATH.exists():
        raise SystemExit(f"File not found: {DF_PATH}. Run `python main.py --step clean` first.")

    df = pd.read_parquet(DF_PATH)
    print("rows, cols:", df.shape)
    print("date range (local):", df["timestamp_local"].min(), "→", df["timestamp_local"].max())
    print("sites:", df["site_id"].nunique(), "pollutants:", df["pollutant"].nunique())
    print("\nUnique pollutant×unit combos (top 15):")
    print(df[["pollutant","unit"]].value_counts().head(15))

    print("\nSample (10 rows):")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()