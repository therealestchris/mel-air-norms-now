import argparse
from pathlib import Path

import pandas as pd

# Resolve project root as "one level up from this file"
ROOT = Path(__file__).resolve().parents[1]
DF_PATH = ROOT / "data" / "interim" / "hourly_uniform.parquet"


def main():
    parser = argparse.ArgumentParser(
        description="Quick peek into hourly_uniform.parquet with optional filters."
    )
    parser.add_argument(
        "-n", "--nrows",
        type=int,
        default=10,
        help="Number of rows to display (default: 10)",
    )
    parser.add_argument(
        "--pollutant",
        type=str,
        help="Filter by pollutant (case-insensitive, e.g. PM2.5, PM25, NO2)",
    )
    parser.add_argument(
        "--site-id",
        type=str,
        help="Filter by site_id",
    )
    parser.add_argument(
        "--site-name",
        type=str,
        help="Filter by site_name (if available in the parquet)",
    )

    args = parser.parse_args()

    if not DF_PATH.exists():
        raise SystemExit(
            f"File not found: {DF_PATH}. "
            "Run `python main.py --step clean` first."
        )

    df = pd.read_parquet(DF_PATH)

    # ---- Basic info ----
    print("rows, cols:", df.shape)
    if "timestamp_local" in df.columns:
        print(
            "date range (local):",
            df["timestamp_local"].min(),
            "→",
            df["timestamp_local"].max(),
        )
    if "site_id" in df.columns:
        print("sites:", df["site_id"].nunique(), end="")
        if "pollutant" in df.columns:
            print(", pollutants:", df["pollutant"].nunique())
        else:
            print()
    print()

    # ---- Apply filters if provided ----
    df_filtered = df.copy()

    if args.pollutant and "pollutant" in df_filtered.columns:
        target = args.pollutant.upper().replace(" ", "").replace(".", "")
        df_filtered["pollutant_norm"] = (
            df_filtered["pollutant"]
            .astype(str)
            .str.upper()
            .str.replace(" ", "")
            .str.replace(".", "")
        )
        df_filtered = df_filtered[df_filtered["pollutant_norm"] == target]
        df_filtered = df_filtered.drop(columns=["pollutant_norm"])
        print(f"Filtered by pollutant = {args.pollutant!r} → {len(df_filtered)} rows")

    if args.site_id and "site_id" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["site_id"] == args.site_id]
        print(f"Filtered by site_id = {args.site_id!r} → {len(df_filtered)} rows")

    if args.site_name and "site_name" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["site_name"] == args.site_name]
        print(f"Filtered by site_name = {args.site_name!r} → {len(df_filtered)} rows")

    if df_filtered.empty:
        print("\nNo rows left after filtering.")
        return

    # ---- Show top pollutant×unit combos (for context) ----
    if "pollutant" in df_filtered.columns and "unit" in df_filtered.columns:
        print("\nUnique pollutant×unit combos (top 15 after filters):")
        print(df_filtered[["pollutant", "unit"]].value_counts().head(15))
        print()

    # ---- Sample rows ----
    n = args.nrows
    print(f"\nSample ({min(n, len(df_filtered))} rows):")
    print(df_filtered.head(n).to_string(index=False))


if __name__ == "__main__":
    main()
