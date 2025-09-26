from src import config
config.REPORT_FIGS.mkdir(parents=True, exist_ok=True)
config.REPORT_TABLES.mkdir(parents=True, exist_ok=True)

def export_all():
    # TODO: matplotlib charts
    print("[viz] (stub) export figures to:", config.REPORT_FIGS)
