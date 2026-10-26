"""Download OpenF1 weekends (practice, quali, sprint, race laps with compounds,
weather, race control, quali telemetry) into data/openf1/. Resumable.

Usage: python scripts/fetch_openf1.py [--from 2023] [--max-minutes 45] [--season 2026 --round 19]
A weekend is complete once its race is stored and the race is > 2 days old;
newer weekends are refreshed on every run (data lands in stages).
Prints OPENF1_COMPLETE when nothing is missing.
"""
import argparse
import os
import time

import pandas as pd

from f1 import ingest, openf1
from f1.jolpica import DataUnavailable


def todo(start: int, now: pd.Timestamp) -> list[tuple[int, int, bool]]:
    cal = ingest.load_calendar()
    cal = cal[(cal["season"] >= start) & (cal["fp1_utc"].fillna(cal["race_utc"]) < now)]
    out = []
    for s, r, ru in zip(cal["season"], cal["round"], cal["race_utc"]):
        fresh = now - ru < pd.Timedelta(days=2)
        if fresh or "R" not in openf1.stored(int(s), int(r)):
            out.append((int(s), int(r), fresh))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=openf1.FIRST_SEASON)
    ap.add_argument("--season", type=int)
    ap.add_argument("--round", type=int)
    ap.add_argument("--max-minutes", type=float)
    a = ap.parse_args()
    now = pd.Timestamp.now(tz="UTC")
    t0 = time.time()
    jobs = ([(a.season, a.round, True)] if a.season else todo(a.start, now))
    print(f"{len(jobs)} weekends to fetch", flush=True)
    cal = ingest.load_calendar()
    keys, fetched = {}, 0
    for season, rnd, fresh in jobs:
        if a.max_minutes and time.time() - t0 > a.max_minutes * 60:
            print("time budget reached", flush=True)
            break
        try:
            if season not in keys:
                keys[season] = openf1.meetings(season, cal)
            if rnd not in keys[season]:
                print(season, rnd, "no OpenF1 meeting", flush=True)
                continue
            info = openf1.fetch_weekend(season, rnd, keys[season][rnd], ttl_hours=0.5 if fresh else None)
            print(season, rnd, info, flush=True)
            fetched += 1
        except openf1.LiveSession:
            print("live session in progress; stopping until the next run", flush=True)
            break
        except DataUnavailable as e:
            print(season, rnd, "unavailable:", e, flush=True)
        except Exception as e:  # one malformed weekend must not stop the backfill
            print(season, rnd, "failed:", repr(e)[:200], flush=True)
    if fetched and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write("updated=true\n")
    left = [j for j in todo(a.start, now) if not j[2]]
    print(f"{len(left)} weekends still missing", flush=True)
    if not left:
        print("OPENF1_COMPLETE", flush=True)
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write("complete=true\n")


if __name__ == "__main__":
    main()
