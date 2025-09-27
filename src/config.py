from pathlib import Path
DATA_DIR = Path("data")
RAW = DATA_DIR / "raw"
INTERIM = DATA_DIR / "interim"
PROCESSED = DATA_DIR / "processed"

REPORTS_DIR = Path("reports")
REPORT_FIGS = REPORTS_DIR / "figures"
REPORT_TABLES = REPORTS_DIR / "tables"

def ensure_dirs() -> None:
    """Create all project directories if missing."""
    for p in (RAW, INTERIM, PROCESSED, REPORT_FIGS, REPORT_TABLES):
        p.mkdir(parents=True, exist_ok=True)