from src import config
config.INTERIM.mkdir(parents=True, exist_ok=True)

def run():
    # TODO: parse/clean to interim files
    print("[clean] (stub) write cleaned data to:", config.INTERIM)
