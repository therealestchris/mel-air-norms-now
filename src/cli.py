# cli.py serves as a single command line interface that lets you run the entire processing pipeline

import argparse
from . import io_fetch, clean, features, compare, viz

def main():
    ap = argparse.ArgumentParser("Melbourne AQ: Norms vs Now")
    ap.add_argument("--step", choices=["fetch","clean","features","compare","viz","all"], default="all")
    args = ap.parse_args()

    if args.step in ("fetch","all"):    io_fetch.fetch_all()
    if args.step in ("clean","all"):    clean.run()
    if args.step in ("features","all"): features.build_all()
    if args.step in ("compare","all"):  compare.run()
    if args.step in ("viz","all"):      viz.export_all()
    print("Done.")
