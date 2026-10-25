"""OPTIONAL practice-session features via FastF1.

Status: NOT verified on real data. Since FastF1 3.9, lap timing from
livetiming.formula1.com requires an F1 account with an active F1TV
subscription (env F1_USERNAME / F1_PASSWORD). Without it, `available()` is
False and the pipeline runs on qualifying + race data only.

When available, practice laps are grouped into runs (laps between pit exits)
and runs are classified, unsupervised, into quali-sims / long-runs / other by
clustering on (run length, pace offset vs driver best, tyre life at start).
Compound is tracked from FastF1's `Compound` column. Long-run pace is
fuel-corrected (~0.06 s/lap) and compared within compound where possible.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .config import CACHE, DATA

PRACTICE_DIR = DATA / "practice"
FUEL_S_PER_LAP = 0.06


def available() -> bool:
    return bool(os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD"))


def _load_laps(season: int, rnd: int, session: str) -> pd.DataFrame | None:
    import fastf1
    fastf1.Cache.enable_cache(str(CACHE / "fastf1"))
    try:
        s = fastf1.get_session(season, rnd, session)
        s.load(laps=True, telemetry=False, weather=False, messages=False)
        return s.laps
    except Exception:
        return None


def classify_runs(laps: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """One row per run with an unsupervised run type."""
    from sklearn.cluster import KMeans
    l = laps.dropna(subset=["LapTime"]).copy()
    l["t"] = l["LapTime"].dt.total_seconds()
    l["run"] = l.groupby("Driver")["PitOutTime"].transform(lambda s: s.notna().cumsum())
    best = l.groupby("Driver")["t"].transform("min")
    l["offset_pct"] = (l["t"] - best) / best * 100
    runs = l[l["PitInTime"].isna() & l["PitOutTime"].isna()].groupby(["Driver", "Team", "run"]).agg(
        n=("t", "size"), offset=("offset_pct", "median"), tyre_life=("TyreLife", "min"),
        compound=("Compound", "first"), median_t=("t", "median"), first_lap=("LapNumber", "min"))
    runs = runs[runs["n"] >= 1].reset_index()
    if len(runs) < 6:
        runs["run_type"] = "other"
        return runs
    Z = runs[["n", "offset", "tyre_life"]].fillna(0).to_numpy(float)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    lab = KMeans(3, random_state=seed, n_init=10).fit_predict(Z)
    prof = runs.assign(lab=lab).groupby("lab")[["n", "offset"]].mean()
    quali = prof["offset"].idxmin()
    long = prof.drop(index=quali)["n"].idxmax()
    runs["run_type"] = np.select([lab == quali, lab == long], ["quali_sim", "long_run"], "other")
    return runs


def weekend_practice(season: int, rnd: int, sessions=("FP1", "FP2", "FP3")) -> pd.DataFrame | None:
    """Per-driver practice features, or None when no practice data is accessible."""
    if not available():
        return None
    frames = [l for s in sessions if (l := _load_laps(season, rnd, s)) is not None and len(l)]
    if not frames:
        return None
    runs = classify_runs(pd.concat(frames, ignore_index=True))
    q = runs[runs.run_type == "quali_sim"].groupby("Driver")["median_t"].min()
    lr = runs[(runs.run_type == "long_run") & (runs.n >= 5)].copy()
    lr["fuel_corr"] = lr["median_t"] + FUEL_S_PER_LAP * lr["first_lap"]
    # compare long runs within the most common compound to avoid tyre bias
    if len(lr):
        main = lr["compound"].mode().iloc[0]
        lr = lr[lr["compound"] == main]
    lrp = lr.groupby("Driver")["fuel_corr"].median()
    out = pd.DataFrame({"fp_best_gap": (q - q.min()) / q.min() * 100,
                        "fp_longrun_gap": (lrp - lrp.min()) / lrp.min() * 100})
    out.index.name = "code"
    PRACTICE_DIR.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_csv(PRACTICE_DIR / f"{season}_{rnd:02d}.csv", index=False)
    return out.reset_index()
