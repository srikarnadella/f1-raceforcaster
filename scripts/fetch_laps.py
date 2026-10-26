"""Backfill race lap data for completed races, respecting Jolpica's 500 req/hour
limit. Safe to interrupt and rerun: races already in data/laps/ are skipped.

Usage: python scripts/fetch_laps.py --from 2019 [--order asc|desc] [--max-minutes 45]
Prints BACKFILL_COMPLETE (and exits 0) when every race since --from is stored.
"""
import argparse
import os
import time

import requests

from f1 import ingest, jolpica, laps
from f1.config import LAPS_DIR


def missing(start: int) -> list[tuple[int, int]]:
    r = ingest.load_results()
    keys = r[r.season >= start][["season", "round"]].drop_duplicates()
    return [(int(s), int(n)) for s, n in keys.itertuples(index=False)
            if not (LAPS_DIR / f"{int(s)}_{int(n):02d}.csv").exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=2019)
    ap.add_argument("--order", choices=["asc", "desc"], default="desc")
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args()
    jolpica._MIN_INTERVAL = 7.5  # ~480 requests/hour
    t0 = time.time()
    todo = sorted(missing(a.start), reverse=a.order == "desc")
    print(f"{len(todo)} races missing", flush=True)
    for season, rnd in todo:
        if a.max_minutes and time.time() - t0 > a.max_minutes * 60:
            print("time budget reached", flush=True)
            break
        for attempt in range(6):
            try:
                l = laps.fetch_race_laps(season, rnd)
                print(season, rnd, len(l), flush=True)
                break
            except jolpica.DataUnavailable as e:
                print(season, rnd, "no lap data:", e, flush=True)
                break
            except (RuntimeError, requests.RequestException) as e:
                print(season, rnd, "retry", e, flush=True)
                time.sleep(600 if not a.max_minutes else 120)
    left = missing(a.start)
    print(f"{len(left)} races still missing", flush=True)
    if not left:
        print("BACKFILL_COMPLETE", flush=True)
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write("complete=true\n")


if __name__ == "__main__":
    main()
