from src import config
config.RAW.mkdir(parents=True, exist_ok=True)

def fetch_all():
    # TODO: download/load actual datasets
    print("[fetch] (stub) put raw files into:", config.RAW)
