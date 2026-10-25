"""Backfill race lap data for completed races (most recent first), respecting
Jolpica's 500 req/hour limit. Safe to interrupt and rerun.

Usage: python scripts/fetch_laps.py --from 2022
"""
import argparse
import time

import _path  # noqa: F401
import requests

from f1 import ingest, jolpica, laps

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="start", type=int, default=2022)
a = ap.parse_args()
jolpica._MIN_INTERVAL = 7.5  # ~480 requests/hour
r = ingest.load_results()
keys = r[r.season >= a.start][["season", "round"]].drop_duplicates().sort_values(["season", "round"], ascending=False)
for season, rnd in keys.itertuples(index=False):
    for attempt in range(6):
        try:
            l = laps.fetch_race_laps(int(season), int(rnd))
            print(season, rnd, len(l), flush=True)
            break
        except (RuntimeError, requests.RequestException) as e:
            print(season, rnd, "retry", e, flush=True)
            time.sleep(600)
