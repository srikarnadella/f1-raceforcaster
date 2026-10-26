"""One definition of an on-track pass, shared by every lap source.

A pass = two drivers swap order between consecutive laps while *both* are on
green laps (not lap 1, not a pit in/out lap, not a safety-car/VSC lap). Pit
cycles and neutralised reshuffles are therefore never counted. Used by
weekend.overtakes (OpenF1 laps, 2023+) and eras.count_overtakes (Jolpica laps,
2019+), so every era and source is counted the same way.
"""
from __future__ import annotations

import pandas as pd


def count_passes(positions: pd.DataFrame, green: set[tuple[str, int]]) -> tuple[int, dict, dict]:
    """positions: lap x driver table of positions at the end of each lap.
    green: (driver, lap) pairs that were green laps.
    Returns (passes, passes made per driver, passes conceded per driver)."""
    n = 0
    made: dict[str, int] = {}
    lost: dict[str, int] = {}
    for lap in positions.index[1:]:
        if lap - 1 not in positions.index:
            continue
        a, b = positions.loc[lap - 1], positions.loc[lap]
        ds = [d for d in positions.columns
              if (d, lap) in green and (d, lap - 1) in green and pd.notna(a[d]) and pd.notna(b[d])]
        for i, x in enumerate(ds):
            for y in ds[i + 1:]:
                if (a[x] - a[y]) * (b[x] - b[y]) < 0:
                    n += 1
                    w, lo = (x, y) if b[x] < b[y] else (y, x)
                    made[w] = made.get(w, 0) + 1
                    lost[lo] = lost.get(lo, 0) + 1
    return n, made, lost


def green_set(laps: pd.DataFrame, categories) -> set[tuple[str, int]]:
    g = laps[laps["category"].isin(categories)]
    return set(zip(g["driver_id"], g["lap"]))
