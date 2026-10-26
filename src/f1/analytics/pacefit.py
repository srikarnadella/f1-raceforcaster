"""The one lap-time model shared by every pace measure (OpenF1 practice long runs and
race pace in analytics/weekend.py, Jolpica race pace in laps.py).

Penalised fixed-effects least squares on green laps:
    lap = driver + compound offset + deg[compound] * tyre_age (+ fuel * lap)
Compound offsets and slopes are shrunk toward COMPOUND_PRIOR / DEG_PRIOR with the
weight of a few laps, so they are data-driven whenever drivers overlap on
compounds and fall back to the priors when they cannot be separated (e.g. every
driver ran one compound). Sources without tyre data pass compound "UNKNOWN".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"]
COMPOUND_PRIOR = {"SOFT": 0.0, "MEDIUM": 0.35, "HARD": 0.7, "INTERMEDIATE": 4.0, "WET": 8.0, "UNKNOWN": 0.0}
DEG_PRIOR = 0.05
PRIOR_WEIGHT = 3.0


def fit(l: pd.DataFrame, fuel: bool, coefficients: bool = False):
    """l: laps with driver_id, lap_s, compound, tyre_age (and lap when fuel=True).
    Returns (driver effect on the first compound at 5-lap-old tyres, attrs), or with
    coefficients=True also the raw fuel and per-compound slopes for correcting laps."""
    l = l[l["lap_s"] < l["lap_s"].median() * 1.08]
    drivers = sorted(l["driver_id"].unique())
    comps = [c for c in COMPOUND_ORDER if c in set(l["compound"])]
    if not comps:
        return None
    D = pd.get_dummies(l["driver_id"]).reindex(columns=drivers, fill_value=0).to_numpy(float)
    C = pd.get_dummies(l["compound"]).reindex(columns=comps, fill_value=0).to_numpy(float)
    age = l["tyre_age"].fillna(0).to_numpy(float)
    A = C * age[:, None]
    cols = [D, C, A] + ([l["lap"].to_numpy(float)[:, None]] if fuel else [])
    X = np.column_stack(cols)
    y = l["lap_s"].to_numpy(float)
    nd, nc = len(drivers), len(comps)
    # prior rows: compound offsets (pin the level) and slopes toward DEG_PRIOR
    P, t = [], []
    for i, c in enumerate(comps):
        r = np.zeros(X.shape[1])
        r[nd + i] = PRIOR_WEIGHT
        P.append(r)
        t.append(PRIOR_WEIGHT * COMPOUND_PRIOR[c])
        r = np.zeros(X.shape[1])
        r[nd + nc + i] = PRIOR_WEIGHT
        P.append(r)
        t.append(PRIOR_WEIGHT * DEG_PRIOR)
    coef, *_ = np.linalg.lstsq(np.vstack([X, P]), np.concatenate([y, t]), rcond=None)
    offs = dict(zip(comps, np.round(coef[nd: nd + nc], 3)))
    deg = dict(zip(comps, np.round(coef[nd + nc: nd + 2 * nc], 4)))
    # reference conditions: first compound, 5-lap-old tyres, mid-distance fuel
    ref = coef[:nd] + coef[nd] - COMPOUND_PRIOR[comps[0]] + 5 * coef[nd + nc]
    if fuel:
        ref = ref + coef[-1] * l["lap"].median()
    extra: dict = {"deg_s_per_lap": {k: float(v) for k, v in deg.items()},
             "compound_offset_s": {k: float(v - offs[comps[0]] + COMPOUND_PRIOR[comps[0]]) for k, v in offs.items()}}
    if fuel:
        extra["fuel_s_per_lap"] = round(float(coef[-1]), 4)
    eff = pd.Series(ref, index=drivers)
    if coefficients:
        return eff, extra, {"fuel": float(coef[-1]) if fuel else 0.0,
                            "deg": dict(zip(comps, coef[nd + nc: nd + 2 * nc].astype(float))),
                            "offset": dict(zip(comps, coef[nd: nd + nc].astype(float)))}
    return eff, extra
