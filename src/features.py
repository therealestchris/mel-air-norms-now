from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from . import config, utils

INTERIM = config.INTERIM
PROCESSED = config.PROCESSED

# accept a daily value only if we have at least this many valid hours that day
MIN_VALID_HOURS = 18

# Baseline windows for "norms" (override via CLI later if desired)
DEFAULT_BASELINE_START = getattr(config, "BASELINE_START", "2019-01-01")
DEFAULT_BASELINE_END = getattr(config, "BASELINE_END", "2023-12-31")

# WHO daily guideline thresholds (ug/m3)
WHO_PM25_DAILY = 15.0
WHO_PM10_DAILY = 45.0

def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Did you run `python main.py --step clean`?")
    
def _auto_select_baseline_window(daily: pd.DataFrame,
                                 req_start: str,
                                 req_end: str) -> tuple[str, str]:
    """
    If the requested baseline window has 0 rows, or if it overlaps only the
    most recent year, (same_year problem), choose a sensible multi-year past
    window from whatever years are available before the latest year
    """

    # ensure 'date' is datetime (robust if caller forgot)  #
    if not pd.api.types.is_datetime64_any_dtype(daily["date"]):         
        daily = daily.copy()                                            
        daily["date"] = pd.to_datetime(daily["date"])                   

    ds = pd.to_datetime(daily["date"].min())
    de = pd.to_datetime(daily["date"].max())
    req_s, req_e = pd.to_datetime(req_start), pd.to_datetime(req_end)

    # requested baseline rows
    base = daily[(daily["date"] >= req_s) & (daily["date"] <= req_e)]
    if len(base) > 0:
        # if the requested window equals the most recent year only, warn
        years = base["date"].dt.year.unique()
        if len(years) == 1 and years[0] == de.year:
            utils.log(f"[features] Baseline only covers current year {de.year}; "
                      f"auto-selecting older years.")
        else:
            return req_start, req_end # keep user window
        
    # build an automatic window: use up to 4 years before most recent year
    avail_years = sorted(daily["date"].dt.year.unique())
    latest = de.year
    older = [y for y in avail_years if y < latest]
    if not older:
        utils.log("[features] No older years available for baseline; norms will be NaN.")
        return req_start, req_end
    
    # take the last up to 4 older years
    chosen = older[-4:]
    auto_start = pd.Timestamp(f"{min(chosen)}-01-01").strftime("%Y-%m-%d")
    auto_end = pd.Timestamp(f"{max(chosen)}-12-31").strftime("%Y-%m-%d")
    utils.log(f"[features] Auto baseline: {auto_start}..{auto_end} from available years {chosen}")
    return auto_start, auto_end
    
def _load_hourly() -> pd.DataFrame:
    """Load the tidy hourly table produced by clean.py"""
    path = INTERIM / "hourly_uniform.parquet"
    _require(path)
    df = pd.read_parquet(path)
    # sanity columns (check that the columns exist)
    needed = {"timestamp_local","site_id","site_name","pollutant","unit","value"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in hourly table: {missing}")
    return df

# function that assists in creating a cleaned daily DataFrame, sorted by site,
# pollutant, data
def _to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly -> daily per (site, pollutant).
    - Use LOCAL date (timestamp_local) to define the day (respects DST)
    - Keep robust 'daily_median' and also 'daily_mean'
    - Require >= MIN_VALID_HOURS to accept a day
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp_local"]).dt.date
    grp = df.groupby(["site_id","site_name","pollutant","unit","date"], as_index=False)
    daily = grp.agg(
        # median and average hourly concentrations for the day
        daily_median = ("value", "median"),
        daily_mean = ("value", "mean"),

        # how many valid hourly readings the day had
        valid_hours = ("value", "count"),
    )

    before = len(daily)
    daily = daily[daily["valid_hours"] >= MIN_VALID_HOURS]
    utils.log(f"[features] Daily acceptance: kept {len(daily)}/{before} days (>= {MIN_VALID_HOURS} hrs)")
    # back to timestamp index for time ops
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values(["site_id", "pollutant", "date"])

# function that creates rolling features, for time-series analysis (analysis where future/current data
# is dependent on past data.) a rolling feature (also called a moving window statistic) 
# is when you calculate a summary (like mean, median, sum, std) over a sliding window of time.
def _add_rollings(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Add 7-day and 30-day rolling medians of the daily_median.
    Uses groupby-apply for clarity; efficient enough for typical sizes.
    """

    out = daily.sort_values(["site_id", "pollutant", "date"]).copy()

    # rolling by group on the daily_median series
    grp = out.groupby(["site_id", "pollutant"])["daily_median"]

    out["roll7"] = grp.transform(lambda s: s.rolling(7, min_periods=1).median())
    out["roll30"] = grp.transform(lambda s: s.rolling(30, min_periods=7).median())

    return out

# returns another series where each element is a string key like "01-15" for Jan 15.
def _mmdd_key(dts: pd.Series) -> pd.Series:
    """
    Calendar-day key 'MM-DD' for climatology. We map Feb-29 → Feb-28
    so leap days borrow the nearby baseline. This is simple & defensible.
    """

    # dts stands for date-times; pd.to_datetime converts whatever is in dts to 
    # proper pandas datetimes. anything that is un-parseable becomes NaT
    d = pd.to_datetime(dts)

    # d.dt.month extracts the month number; astype turns it into a string
    # zfill: left-pads with zeros to length 2. e.g. "1" → "01"
    mm = d.dt.month.astype(str).str.zfill(2)
    dd = d.dt.day.astype(str).str.zfill(2)
    mmdd = (mm + "-" + dd).where(~((mm == "02") & (dd == "29")), "02-28")
    return mmdd

def _build_climatology(daily: pd.DataFrame,
                       base_start: str,
                       base_end: str) -> pd.DataFrame:
    """
    Build per-station, per-pollutant baseline norms by calendar day (MM-DD):
      - norm_median, norm_q25, norm_q75
    """
    base = daily[(daily["date"] >= base_start) & (daily["date"] <= base_end)].copy()
    if base.empty:
        utils.log(f"[features] WARNING: baseline window {base_start}..{base_end} has 0 rows.")
    base["mmdd"] = _mmdd_key(base["date"])

    # Aggregate baseline daily_median by calendar key
    def p25(x):
        return np.percentile(x, 25)
    
    def p75(x):
        return np.percentile(x, 75)
    
    clim = (base
            .groupby(["site_id","site_name","pollutant","unit","mmdd"], as_index = False)
            .agg(norm_median=("daily_median", "median"),
                 norm_q25=("daily_median", p25),
                 norm_q75=("daily_median", p75)))
    return clim

# function that clarifies where day-to-day readings get compared to seasonal “norms” 
# so we can quantify how unusual a day was. 'daily' is the day-to-day metrics, e.g.
# daily_median for each site x pollutant. 'clim' is the climatology table (the seasonable
# "normal" for each calendar day), with columns like 'norm_median', 'norm_q25', 'norm_q75'
# this function attaches the norm values to each daily row -> computes differences, a robust
# z-score, and WHO exceedance flags. (z-score is the amount of standard deviations a data point 
# is from the mean of its distribution)
def _join_norms_and_compute(daily: pd.DataFrame,
                            clim: pd.DataFrame) -> pd.DataFrame:
    """
    Join norms onto each daily row via 'mmdd', then compute:
      - delta_abs, delta_pct
      - z (IQR-scaled; sigma ≈ IQR/1.349)
      - PM exceedance flags vs WHO guidelines
    """
    # daily.copy() works on a copy so we don't mutate the caller's DataFrame
    out = daily.copy()

    # creates a calendar-day key like "03-21", ignoring the year
    out["mmdd"] = _mmdd_key(out["date"])

    # absolute difference for the norm; out.merge() does a relational join,
    # similar to the concept in SQL. it finds matching rows between out (daily)
    # and clim (norms) where all join keys match.
    out = out.merge(
        clim, 
        on = ["site_id", "site_name", "pollutant", "unit", "mmdd"],
        # how = "left" acts like a left join, keep all daily rows, even if a 
        # matching norm is missing.
        how = "left",

        # enforces the expected join cardinality. "many-to-one": many daily 
        # rows can map to a single climatology row (for that site x pollutant
        # x unit x mmdd)
        validate="m:1"
    )

    # Deltas (absolute differences)
    out["delta_abs"] = out["daily_median"] - out["norm_median"]

    # percentage difference from the norm
    out["delta_pct"] = np.where(
        (out["norm_median"].notna()) & (out["norm_median"] != 0),
        100.0 * out["delta_abs"] / out["norm_median"],
        np.nan
    )

    # z-score using IQR as robust scale
    iqr = (out["norm_q75"] - out["norm_q25"]).replace(0, np.nan)
    out["z"] = (out["daily_median"] - out["norm_median"]) / (iqr / 1.349) # IQR is 1.349 * stdev.

    # Exceedance flags (did the day exceed WHO daily guideline thresholds for PM2.5/PM10?)
    pol = out["pollutant"].str.upper()
    out["exceed_pm25_who"] = np.where((pol == "PM2.5") & (out["daily_median"] > WHO_PM25_DAILY), 1, 0)
    out["exceed_pm10_who"] = np.where((pol == "PM10") & (out["daily_median"] > WHO_PM10_DAILY), 1, 0)

    # Clean up, using the 3 natural indexing keys for final daily time series.
    return out.sort_values(["site_id","pollutant","date"])

# after this part, we create the public entry point; which is the place in the program where the
# execution begins.
def build_all(base_start: str = DEFAULT_BASELINE_START,
              base_end:   str = DEFAULT_BASELINE_END) -> None:
    """
    Orchestrate feature building:
    1) hourly -> daily
    2) rolling windows -> 
    3) baseline climatology
    4) join + deltas / z / exceedance
    5) write outputs
    """
    # from config.py; ensure all directories exist, and create folders if missing
    config.ensure_dirs()
    utils.log("[features] Loading hourly tidy data...")
    hourly = _load_hourly()
    # {len(hourly):,} means: "format the integer with thousands seperators."
    # e.g., len(hourly) == 1234567 → prints 1,234,567.
    utils.log(f"[features] Read hourly rows: {len(hourly):,}")

    # converts hourly rows into daily metrics per (site, pollutant): median/mean and
    # valid_hours (then filters by MIN_VALID_HOURS)
    utils.log("[features] Aggregating hourly -> daily...")
    daily = _to_daily(hourly)

    utils.log("[features] Adding rolling medians (7d, 30d)...")
    daily = _add_rollings(daily)
    daily_path = PROCESSED / "daily.parquet"
    daily.to_parquet(daily_path, index=False)
    utils.log(f"[features] Wrote daily -> {daily_path} ({len(daily):,} rows)")

    auto_start, auto_end = _auto_select_baseline_window(daily, base_start, base_end)
    utils.log(f"[features] Building baseline climatology {auto_start}..{auto_end}")
    clim = _build_climatology(daily, auto_start, auto_end)
    clim_path = PROCESSED / f"climatology_{auto_start}_{auto_end}.parquet"
    clim.to_parquet(clim_path, index=False)
    utils.log(f"[features] Wrote climatology -> {clim_path} ({len(clim):,} rows)")

    utils.log("[features] Joining norms and computing deltas/z/exceedance...")
    joined = _join_norms_and_compute(daily, clim)
    joined_path = PROCESSED / "daily_with_norms.parquet"
    joined.to_parquet(joined_path, index = False)
    utils.log(f"[features] Wrote joined daily -> {joined_path} ({len(joined):,} rows)")

    # Quick hints if norms are missing
    # joined["norm_median"].isna() returns a Boolean series: True when the norm is missing.
    # .mean() on booleans treats True=1, False=0, so you get the fraction of missing norms.
    missing_norms = joined["norm_median"].isna().mean()
    if missing_norms > 0:
        utils.log(f"[features] NOTE: {missing_norms:.1%} of days lacked norms "
                  f"(outside baseline window or sparse baseline). Consider adjusting baseline. ")