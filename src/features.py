from src import config
config.PROCESSED.mkdir(parents=True, exist_ok=True)

def build_all():
    # TODO: build daily aggregates & climatology
    print("[features] (stub) write features to:", config.PROCESSED)
