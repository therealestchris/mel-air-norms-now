from __future__ import annotations
from pathlib import Path

import pandas as pd
import numpy as np

# '.' means "from this directory, also import"
from . import config, utils

PROCESSED = config.PROCESSED

# Baseline windows to compare
BASELINE_PRE_START = "2019-01-01"
BASELINE_PRE_END = "2023-12-31"

BASELINE_LONG_START = "2019-01-01"
BASELINE_LONG_END = "2025-01-01"

# Function that raises helpful error if expected file is missing
def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Ensure to run 'python main.py --step features'")
    
# Function that loads baseline climatology and checks if it has
# the columns needed
def _load_climatology(start: str, end: str) -> pd.DataFrame:
    path = PROCESSED / f"climatology_{start}_{end}.parquet"
    _require(path)

    df = pd.read_parquet(path)

    # question: are these the names of the required features in the daily.parquet file, after preprocessing?
    # question: is "norm_median" calculated using mean imputation?
    needed = {"site_id", "site_name", "pollutant", "unit", "mmdd", "norm_median"} # norm_median is the actual normal we are comparing to
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns needed for comparison: {missing}")
    
    # question: why do we need df.copy again?
    df = df.copy()
    df["baseline_start"] = start
    df["baseline_end"] = end
    return df

# Function that compares the two climatology baselines - pre-now window, and long window including 
# recent years (specified above). The function returns 'detailed_comparison', one row per (site, pollutant, mmdd)
# and a 'summary': one row per (site, pollutant) aggregating over the year
# question: why do we decide this split of pre-now and long window? what is the logic, is there a 
# data science concept behind it?
# question: please explain more about what detailed_comparison and summary is.
def _compare_baselines() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Load both baseline tables
    base_pre = _load_climatology(BASELINE_PRE_START, BASELINE_PRE_END)
    base_long = _load_climatology(BASELINE_LONG_START, BASELINE_LONG_END)

    # question: why is norm_median not in here?
    key_cols = ["site_id", "site_name", "pollutant", "unit", "mmdd"]

    # Inner join: keep only days (MM-DD) that exists in BOTH baselines. merge() is a DataFrame method.
    # question: why this process? 
    merged = base_pre.merge(
        base_long,
        on=key_cols,
        how="inner",
        suffixes=("_pre", "_long"),
        validate="1:1",
    )

    # Finding out the changes from the "normal" daily median
    # question: where does norm_median_long come from?
    merged["delta_norm"] = merged["norm_median_long"] - merged["norm_median_pre"]

    # Percentage change; avoid division by zero
    merged["delta_norm_pct"] = np.where(
        merged["norm_median_pre"] != 0,
        100.0 * merged["delta_norm"] / merged["norm_median_pre"],
        np.nan,
    )

    # For easier high-level view, we summarize across the year
    summary = (
        merged
        .groupby(["site_id", "site_name", "pollutant", "unit"], as_index=False)
        .agg(
            # question: can you explain more clearly what does these calculate?
            median_delta_norm=("delta_norm", "median"),
            mean_delta_norm=("delta_norm", "mean"),
            median_delta_norm_pct=("delta_norm_pct", "median"),
            mean_delta_norm_pct=("delta_norm_pct", "mean"),
        )
    )

    return merged, summary




