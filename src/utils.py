from datetime import datetime

def log(msg: str) -> None:
    """Lightweight timestamped logger used across the pipeline."""
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")