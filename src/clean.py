from __future__ import annotations
from pathlib import Path
import pandas as pd
from zoneinfo import ZoneInfo
from . import config, utils

RAW = config.RAW
INTERIM = config.INTERIM
LOCAL_TZ = ZoneInfo("Australia/Melbourne")

# Exact mapping based on your file's headers (AllData sheet)
COLMAP = {
    "datetime_local":  "timestamp_local",
    "datetime_AEST":   "timestamp_aest",
    "location_id":     "site_id",
    "location_name":   "site_name",
    "parameter_name":  "pollutant",
    "unit_of_measure": "unit",
    "value":           "value",
    "validation_flag": "validation_flag",
}

# the final, canonical set of columns for the cleaned dataset
TARGET_COLS = [
    "timestamp_local",
    "timestamp_utc",
    "site_id",
    "site_name",
    "pollutant",
    "unit",
    "value",
    "validation_flag",
]

# restricts read_excel(..., usecols=...) to just the columns you need.
USECOLS_READ = [
    "datetime_local", "datetime_AEST",
    "location_id", "location_name",
    "parameter_name", "unit_of_measure",
    "value", "validation_flag",
]

# how many rows you’ll show in debug logs (e.g., df.head(DEBUG_SAMPLE_MAX)).
DEBUG_SAMPLE_MAX = 5  

# the "_" in the function name just signifies that it is an internal function.
# Tries to read a file at path into a pandas DataFrame.
# If it’s an Excel file (.xlsx/.xls), it first tries to read the 
# "AllData" sheet with only the columns you care about (USECOLS_READ);
# if that fails (sheet missing, different name, etc.), it auto-detects the best sheet by looking 
# at a tiny sample of each sheet’s header and picking the one with the biggest column overlap.
# if it’s not Excel, it falls back to CSV (pd.read_csv).
def _read_any(path: Path) -> pd.DataFrame:
    """Read from the correct sheet; fall back to auto-detection."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        try:
            # loads data from an excel file into a pandas DataFrame.
            # openpyxl is a python package that can read/write .xlsx files.
            df = pd.read_excel(path, sheet_name="AllData", usecols=USECOLS_READ, engine="openpyxl")

            # utils.log is your project’s logging helper (defined in your utils module).
            # it likely prints a timestamped / prefixed message to stdout or to a log file.
            utils.log(f"[clean] Loaded sheet=AllData rows={len(df):,} cols={list(df.columns)}")  
            return df
        
        except Exception as e:
            utils.log(f"[clean] Could not read 'AllData' directly ({e}); scanning sheets…")  
            sheets = pd.read_excel(path, sheet_name=None, nrows=2, engine="openpyxl")
            wanted = set(USECOLS_READ)
            best_name, best_overlap = None, -1

            for name, head in sheets.items():
                # map(str, head.columns) applies str(...) to each column name, ensuring everything is a plain string
                cols = set(map(str, head.columns))
                overlap = len(wanted & cols)
                if overlap > best_overlap:
                    best_name, best_overlap = name, overlap

            if best_name is None or best_overlap < 4:
                raise ValueError("Could not find a sheet with expected columns. Please check the file.")
            
            utils.log(f"[clean] Auto-selected sheet: {best_name} (overlap={best_overlap})")
            df = pd.read_excel(path, sheet_name=best_name, usecols=USECOLS_READ, engine="openpyxl")
            utils.log(f"[clean] Loaded sheet={best_name} rows={len(df):,} cols={list(df.columns)}")  
            return df
        
    # CSV fallback
    df = pd.read_csv(path)
    utils.log(f"[clean] Loaded CSV rows={len(df):,} cols={list(df.columns)}")  
    return df

# function that parses whatever time columns exist, and makes sure they are timezone-aware,
# so pandas knows what clock they belong to. the end result will be that every row has both a 
# timestamp_local (Melbourne time) and a timestamp_aest (australian eastern standard time), and guaranteed
# to be consistent with each other.
def _coerce_time(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure tz-aware local time and derive UTC."""

    # Prefer 'timestamp_local' (renamed from datetime_local)
    if "timestamp_local" in df.columns and df["timestamp_local"].notna().any():
        tl = pd.to_datetime(df["timestamp_local"], errors="coerce")
    elif "timestamp_aest" in df.columns:
        tl = pd.to_datetime(df["timestamp_aest"], errors="coerce")
        df = df.drop(columns=["timestamp_aest"])
    else:
        tl = pd.Series(pd.NaT, index=df.index)

    # Localize Melbourne if naive; treat DST oddities as NaT
    if getattr(tl.dt, "tz", None) is None:
        tl = tl.dt.tz_localize(LOCAL_TZ, nonexistent="NaT", ambiguous="NaT")

    df["timestamp_local"] = tl
    df["timestamp_utc"] = df["timestamp_local"].dt.tz_convert("UTC")

    before = len(df)
    df = df.dropna(subset=["timestamp_local", "timestamp_utc"])
    dropped = before - len(df)
    if dropped:
        utils.log(f"[clean] Dropped {dropped} rows with missing timestamps")  # UPDATED (more explicit)
    return df

# this function acts as a "sanity gate", that keeps only usable measurements. it drops
# rows where the measurement is missing, forces the measurement to be numeric (where
# bad strings = NaN), remove impossible negatives, and if a "validation_flag" column exists,
# keep rows that look valid (or blank).
def _quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Conservative QC: numeric, non-negative; DO NOT drop on flags yet."""
    before = len(df)
    df = df.dropna(subset=["value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["value"] >= 0]
    utils.log(f"[clean] QC numeric/non-negative kept {len(df)}/{before} rows")

    # Log validation_flag distribution, but don't filter yet
    if "validation_flag" in df.columns:
        v = df["validation_flag"].astype(str).str.strip().str.lower()
        counts = v.value_counts(dropna=False).head(10)
        utils.log(f"[clean] validation_flag sample counts:\n{counts.to_string()}")

    return df

def _normalise_units(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise pollutant labels & units."""
    if "pollutant" in df.columns:
        df["pollutant"] = df["pollutant"].astype(str).str.upper().str.replace(" ", "", regex=False)

    if "unit" in df.columns:
        df["unit"] = (
            df["unit"].astype(str)
            .str.replace("μ", "u", regex=False)
            .str.replace("³", "3", regex=False)
            .str.strip()
        )

    pm_mask = df.get("pollutant", pd.Series(False, index=df.index)).isin(["PM2.5", "PM10"])
    if "unit" in df.columns:
        mg_mask = df["unit"].str.lower().eq("mg/m3")
        df.loc[pm_mask & mg_mask, "value"] = df.loc[pm_mask & mg_mask, "value"] * 1000.0
        df.loc[pm_mask & mg_mask, "unit"] = "ug/m3"
        df.loc[pm_mask & df["unit"].str.contains("ug", case=False, na=False), "unit"] = "ug/m3"

        ppm_mask = df["unit"].str.lower().eq("ppm")
        df.loc[~pm_mask & ppm_mask, "value"] = df.loc[~pm_mask & ppm_mask, "value"] * 1000.0
        df.loc[~pm_mask & ppm_mask, "unit"] = "ppb"
        df.loc[~pm_mask & df["unit"].str.contains("ppb", case=False, na=False), "unit"] = "ppb"
    return df

def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate hourly records for the same (UTC time, site, pollutant)."""
    key_cols = {"timestamp_utc", "site_id", "pollutant"}
    if not key_cols.issubset(df.columns):
        utils.log(f"[clean] Skip dedupe; missing key cols {key_cols - set(df.columns)}")  
        return df
    before = len(df)
    df = (
        df.sort_values(["timestamp_utc"])
          .drop_duplicates(subset=["timestamp_utc", "site_id", "pollutant"], keep="first")
    )
    removed = before - len(df)
    if removed:
        utils.log(f"[clean] Deduped {removed} duplicates")
    return df

# driver function that does the whole cleaning step. return type hint '-> None' says the function
# does not return a value.
def run() -> None:
    """Entry point called by CLI: read raw -> clean -> write interim parquet."""
    config.ensure_dirs()

    paths = sorted(RAW.glob("*.xlsx")) + sorted(RAW.glob("*.csv"))
    if not paths:
        utils.log(f"[clean] No raw files in {RAW}. Put your file(s) there and rerun.")
        return

    frames = []
    for p in paths:
        utils.log(f"[clean] Reading {p.name}")
        df = _read_any(p)
        utils.log(f"[clean] After read: {len(df):,} rows | cols={list(df.columns)}")  

        # Rename vendor columns to schema where present
        df.columns = [c.strip() for c in df.columns]
        rename = {k: v for k, v in COLMAP.items() if k in df.columns}
        df = df.rename(columns=rename)
        utils.log(f"[clean] Renamed cols present: {list(rename.values())}") 

        # Keep a consistent subset if columns exist
        keep_candidates = ["timestamp_local", "timestamp_aest", "site_id", "site_name",
                           "pollutant", "unit", "value", "validation_flag"]
        keep = [c for c in keep_candidates if c in df.columns]
        df = df[keep].copy()
        utils.log(f"[clean] After rename/keep: {len(df):,} rows | keep={keep}")  

        df = _coerce_time(df)
        utils.log(f"[clean] After time coercion: {len(df):,} rows")  

        df = _quality_filters(df)
        utils.log(f"[clean] After QC: {len(df):,} rows")  

        df = _normalise_units(df)
        df = _dedupe(df)
        utils.log(f"[clean] After dedupe: {len(df):,} rows")  

        # Minimal string hygiene
        for col in ("site_id", "site_name"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Print tiny sample to ensure values look sane
        utils.log(f"[clean] Sample rows:\n{df.head(DEBUG_SAMPLE_MAX).to_string(index=False)}")  

        frames.append(df)

    if not frames:
        utils.log("[clean] No frames collected.")
        return

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["site_id", "pollutant", "timestamp_utc"])

    out_path = INTERIM / "hourly_uniform.parquet"
    out.to_parquet(out_path, index=False)
    utils.log(f"[clean] Wrote {len(out):,} rows → {out_path}")

    # if empty, dump a CSV snapshot so we can inspect structure quickly
    if len(out) == 0:
        snap = INTERIM / "hourly_uniform_DEBUG.csv"
        out.to_csv(snap, index=False)
        utils.log(f"[clean] DEBUG: wrote empty snapshot CSV → {snap}")
