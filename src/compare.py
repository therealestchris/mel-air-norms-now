from __future__ import annotations
from pathlib import Path

import pandas as pd
import numpy as np

# '.' means "from this directory, also import"
from . import config, utils

PROCESSED = config.PROCESSED

# Baseline windows to compare
# "What were the typical conditions before the very recent period we care about?"
BASELINE_PRE_START = "2019-01-01"
BASELINE_PRE_END = "2023-12-31"

# "If I treat a longer period including recent years as normal, how does that change
# my notion of 'normal'?"
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

    # 'needed' are the columns we expect in each climatology file, created in 
    # features._build_climatology() and saved inside features._build_all():
    # 1. climatology_2019-01-01_2023-12-31.parquet
    # 2. climatology_2019-01-01_2025-01-01.parquet
    # norm_median is the median of daily_median values across the baseline years for 
    # that calendar day; not median-imputed values.
    needed = {"site_id", "site_name", "pollutant", "unit", "mmdd", "norm_median"} # norm_median is the actual normal we are comparing to
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns needed for comparison: {missing}")
    
    # pd.read_parquet usually gives a fresh DataFrame, but in more complex code
    # may just give a 'view'. df.copy() ensures we actually get a dataframe to work
    # with, not just a view.
    df = df.copy()
    df["baseline_start"] = start
    df["baseline_end"] = end
    return df

# Function that compares the two climatology baselines - pre-now window, and long window including 
# recent years (specified above). The function returns 'detailed_comparison', one row per (site, pollutant, mmdd)
# and a 'summary': one row per (site, pollutant) aggregating over the year
# 'detailed_comparison' helps answers questions like: “For PM2.5 at Alphington on 01-15, 
# how did the normal change when I include 2024–2025?”
def _compare_baselines() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Load both baseline tables
    base_pre = _load_climatology(BASELINE_PRE_START, BASELINE_PRE_END)
    base_long = _load_climatology(BASELINE_LONG_START, BASELINE_LONG_END)

    # These are the columns that define the identity of a climatology row, regardless of baseline.
    # hence why we do not include 'norm_median' unlike above
    key_cols = ["site_id", "site_name", "pollutant", "unit", "mmdd"]

    # Inner join: keep only days (MM-DD) that exists in BOTH baselines. merge() is a DataFrame method.
    merged = base_pre.merge(
        base_long,
        on=key_cols,
        how="inner",
        suffixes=("_pre", "_long"),
        validate="1:1",
    )

    # Finding out the changes from the "normal" daily median
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
            # 'delta' tells about typical change in 'normal' over the whole year
            median_delta_norm=("delta_norm", "median"),
            mean_delta_norm=("delta_norm", "mean"),
            median_delta_norm_pct=("delta_norm_pct", "median"),
            mean_delta_norm_pct=("delta_norm_pct", "mean"),
        )
    )

    return merged, summary

# Function to tidy daily table for modelling. It returns one row per (site_id, 
# pollutant, date) with 1. daily median and rolling medians, 2. norm_median
# and deltas vs norm, 3. z-score and WHO exceedance flags
def _build_ml_daily_features() -> pd.DataFrame:
    daily_path = PROCESSED / "daily_with_norms.parquet"
    _require(daily_path)

    df = pd.read_parquet(daily_path).copy()

    # We ensure date is datetime, as sometimes parquet stores it as a plain object
    df["date"] = pd.to_datetime(df["date"])

    # We can tweak this part depending on what period we want to model
    # e.g. keep 2020+ for modelling
    df["year"] = df["date"].dt.year
    ml = df[df["year"] >= 2020].copy()

    # We keep a focused set of columns for ML/correlation notebooks
    keep_cols = [
        "site_id",
        "site_name",
        "pollutant",
        "unit",
        "date",
        "year",

        "daily_median",
        "daily_mean",
        "roll7",
        "roll30",

        "norm_median",
        "delta_abs",
        "delta_pct",
        "z",

        "exceed_pm25_who",
        "exceed_pm10_who",
    ]

    # Only keep columns that exist (in case some aren't in your file)
    keep_cols = [c for c in keep_cols if c in ml.columns]
    ml = ml[keep_cols].sort_values(["site_id", "pollutant", "date"])

    return ml

# Function that runs all the functionalities of compare.py
def run() -> None:
    # question: can you explain how utils.log works?
    utils.log("[compare] Comparing baselines...")
    baseline_detail, baseline_summary = _compare_baselines()

    detail_path = PROCESSED / "baseline_norms_comparison.parquet"
    baseline_detail.to_parquet(detail_path, index=False)    
    utils.log(f"[compare] Wrote detailed baseline comparison -> {detail_path} ({len(baseline_detail):,} rows)")

    summary_path = PROCESSED / "baseline_norms_summary.parquet"
    baseline_summary.to_parquet(summary_path, index=False)
    utils.log(f"[compare] Wrote baseline summary -> {summary_path} ({len(baseline_summary):,} rows)")

    utils.log("[compare] Building ML-ready daily feature table...")
    ml = _build_ml_daily_features()
    ml_path = PROCESSED / "ml_daily_features.parquet"
    # question: does this part of the function create the parquet file? (to_parquet())
    # and why do we set index=False?
    ml.to_parquet(ml_path, index=False)
    utils.log(f"[compare] Wrote ML daily features -> {ml_path} ({len(ml):,} rows)")

    utils.log("[compare] Done.")

""" Note to self:
1. Yet to ask questions to ChatGPT
2. Progress up until _compare_baselines function, still need to complete
steps 5-6.
Chat Link: https://chatgpt.com/g/g-p-68d5f36ca1e481919c19a357c89d3bb0/c/68d5fa1d-d164-8324-8c3c-3e90e0dbad92 """



