from __future__ import annotations
from pathlib import Path
import pandas as pd
import re
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

def _norm(s: str) -> str:
    """
    normalize a column name for fuzzy matching
    """
    return re.sub(r"[^a-z0-9]", "", s.lower())

# for each canonical field, list acceptable header spellings (normalised)
ALIAS = {
    "datetime_local": {
        "datetimelocal","datetimeaedt","localdatetime","localdate","localtime",
        "datetimelocal(aest)","datetime_local","datetimeaest","date_time_local"
    },
    "datetime_aest": {
        "datetimeaest","datetimelocal(aest)","datetime_aest","aesttime"
    },
    "location_id": {
        "locationid","siteid","stationid","locationcode","site_id"
    },
    "location_name": {
        "locationname","sitename","stationname","site_name","station"
    },
    "parameter_name": {
        "parametername","parameter","pollutant","pollutantname"
    },
    "unit_of_measure": {
        "unitofmeasure","unit","units","uom"
    },
    "value": {
        "value","concentration","reading","measurement"
    },
    "validation_flag": {
        "validationflag","validity","qaflag","flag"
    },
}

def _build_rename_map(df_cols: list[str]) -> dict:
    """
    return rename map from raw -> canonical using aliases
    """
    norm_to_raw = {_norm(c): c for c in df_cols}
    rename = {}
    def pick(key: str):
        for alias in ALIAS[key]:
            if alias in norm_to_raw:
                return norm_to_raw[alias]
        return None
    
    candidates = {
        "datetime_local": pick("datetime_local"),
        "datetime_AEST":  pick("datetime_aest"),
        "location_id":    pick("location_id"),
        "location_name":  pick("location_name"),
        "parameter_name": pick("parameter_name"),
        "unit_of_measure":pick("unit_of_measure"),
        "value":          pick("value"),
        "validation_flag":pick("validation_flag"),
    }
    # Build raw→canonical map (only for found columns)
    colmap = {
        "datetime_local":  "timestamp_local",
        "datetime_AEST":   "timestamp_aest",
        "location_id":     "site_id",
        "location_name":   "site_name",
        "parameter_name":  "pollutant",
        "unit_of_measure": "unit",
        "value":           "value",
        "validation_flag": "validation_flag",
    }
    for raw_key, canon in colmap.items():
        raw_col = candidates.get(raw_key)
        if raw_col:
            rename[raw_col] = canon
    return rename


# the "_" in the function name just signifies that it is an internal function.
# Tries to read a file at path into a pandas DataFrame.
# If it’s an Excel file (.xlsx/.xls), it first tries to read the 
# "AllData" sheet with only the columns you care about (USECOLS_READ);
# if that fails (sheet missing, different name, etc.), it auto-detects the best sheet by looking 
# at a tiny sample of each sheet’s header and picking the one with the biggest column overlap.
# if it’s not Excel, it falls back to CSV (pd.read_csv).
def _read_any(path: Path) -> pd.DataFrame:
    """Robust Excel reader: scan sheets & header rows, fuzzy-match columns, then read once."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        try:
            # loads data from an excel file into a pandas DataFrame.
            # openpyxl is a python package that can read/write .xlsx files.
            df = pd.read_excel(path, sheet_name="AllData", engine="openpyxl")

            # utils.log is your project’s logging helper (defined in your utils module).
            # it likely prints a timestamped / prefixed message to stdout or to a log file.
            utils.log(f"[clean] Loaded sheet=AllData rows={len(df):,} cols={list(map(str, df.columns))}")
            return df
        except Exception as e:
            utils.log(f"[clean] No 'AllData' sheet ({e}); scanning all sheets…") 

        # scan all sheets & try multiple header rows for best match
        xl = pd.ExcelFile(path, engine="openpyxl")  
        best = None                                  
        best_score = -1                              
        best_info = None                             

        # Union of all alias tokens (normalized)
        alias_union = set().union(*ALIAS.values())   

        for sheet in xl.sheet_names:                 
            # try first 5 rows as header
            for header_row in range(0, 5):       
                try:
                    head = xl.parse(sheet, nrows=1, header=header_row)
                except Exception:
                    continue
                # map(str, head.columns) applies str(...) to each column name, ensuring everything is a plain string
                raw_cols = list(map(str, head.columns))
                norm_cols = {_norm(c) for c in raw_cols}
                # score = how many of our alias tokens appear in this header
                score = len(norm_cols & alias_union)
                if score > best_score:
                    best_score = score
                    best = (sheet, header_row, raw_cols)
        # require at least 4 useful cols
        if best is None or best_score < 4:         
            raise ValueError("Could not find a sheet+header with expected columns. Please check the file.")

        sheet, header_row, raw_cols = best
        utils.log(f"[clean] Auto-selected sheet='{sheet}' header_row={header_row} (score={best_score})")  

        # read full sheet once (no usecols), then we’ll rename+select downstream
        df = xl.parse(sheet, header=header_row)      
        utils.log(f"[clean] Loaded sheet={sheet} rows={len(df):,} cols={list(map(str, df.columns))}")  
        return df

    # CSV fallback
    df = pd.read_csv(path)
    utils.log(f"[clean] Loaded CSV rows={len(df):,} cols={list(map(str, df.columns))}")
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
        rename = _build_rename_map(list(df.columns))
        df = df.rename(columns=rename)
        utils.log(f"[clean] Renamed → {rename}")  # helpful debug

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
