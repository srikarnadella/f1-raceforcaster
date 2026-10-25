"""Race lap data: download, lap categorisation, fuel/tyre-corrected race pace,
and unsupervised compound inference.

Source: Jolpica lap timings + pit stops (free). Tyre compound is NOT in the
free data (it needs the F1TV-authenticated live-timing feed), so compound-like
classes are inferred by clustering stints on pace offset and degradation slope.
Those classes are labelled as inferred, never as fact.

Lap categories:
  lap1      first lap (start chaos)
  in        lap ending in a pit stop
  out       lap after a pit stop
  neutral   field-wide slow lap (likely SC / VSC / red flag): field median > 115% of race median
  slow      driver's lap > 107% of their own green median (incident, damage, traffic fight)
  traffic   started the lap < 1.5 s behind the car ahead
  clean     everything else
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import jolpica
from .config import LAPS_DIR
from .ingest import lap_seconds

TRAFFIC_GAP = 1.5
NEUTRAL_FACTOR = 1.15
SLOW_FACTOR = 1.07


def fetch_race_laps(season: int, rnd: int, ttl_hours: float | None = None) -> pd.DataFrame:
    """Lap rows (driver_id, lap, time_s, position, stint, tyre_age, pit_this_lap)."""
    path = LAPS_DIR / f"{season}_{rnd:02d}.csv"
    if path.exists():
        return pd.read_csv(path)
    races = jolpica.fetch_races(f"{season}/{rnd}/laps", "Laps", ttl_hours)
    if not races:
        raise jolpica.DataUnavailable(f"no lap data for {season} R{rnd}")
    rows = [{"driver_id": t["driverId"], "lap": int(l["number"]), "time_s": lap_seconds(t["time"]),
             "position": int(t["position"])}
            for l in races[0]["Laps"] for t in l["Timings"]]
    laps = pd.DataFrame(rows).drop_duplicates(["driver_id", "lap"]).sort_values(["driver_id", "lap"])
    stops = jolpica.fetch_races(f"{season}/{rnd}/pitstops", "PitStops", ttl_hours)
    pit = {(p["driverId"], int(p["lap"])) for p in (stops[0]["PitStops"] if stops else [])}
    laps["pit_this_lap"] = [(d, l) in pit for d, l in zip(laps["driver_id"], laps["lap"])]
    laps["stint"] = laps.groupby("driver_id")["pit_this_lap"].transform(lambda s: s.shift(fill_value=False).cumsum()) + 1
    laps["tyre_age"] = laps.groupby(["driver_id", "stint"]).cumcount() + 1
    laps.insert(0, "round", rnd)
    laps.insert(0, "season", season)
    LAPS_DIR.mkdir(parents=True, exist_ok=True)
    laps.to_csv(path, index=False)
    return laps


def categorise(laps: pd.DataFrame) -> pd.DataFrame:
    l = laps.sort_values(["driver_id", "lap"]).copy()
    l["cum"] = l.groupby("driver_id")["time_s"].cumsum()
    # gap to the car ahead at the end of the previous lap (same lap count only)
    prev = l[["driver_id", "lap", "cum", "position"]].copy()
    prev["lap"] += 1
    prev = prev.sort_values(["lap", "position"])
    prev["gap_ahead"] = prev.groupby("lap")["cum"].diff()
    l = l.merge(prev[["driver_id", "lap", "gap_ahead"]], on=["driver_id", "lap"], how="left")
    race_med = l["time_s"].median()
    field_med = l.groupby("lap")["time_s"].transform("median")
    out_lap = l.groupby("driver_id")["pit_this_lap"].shift(fill_value=False)
    cat = np.full(len(l), "clean", dtype=object)
    green_med = l[(l["lap"] > 1) & (field_med < NEUTRAL_FACTOR * race_med)].groupby("driver_id")["time_s"].median()
    slow = l["time_s"] > SLOW_FACTOR * l["driver_id"].map(green_med)
    cat[(l["gap_ahead"] < TRAFFIC_GAP).to_numpy()] = "traffic"
    cat[slow.to_numpy()] = "slow"
    cat[(field_med > NEUTRAL_FACTOR * race_med).to_numpy()] = "neutral"
    cat[out_lap.to_numpy()] = "out"
    cat[l["pit_this_lap"].to_numpy()] = "in"
    cat[(l["lap"] == 1).to_numpy()] = "lap1"
    l["category"] = cat
    return l


def corrected_pace(laps: pd.DataFrame) -> pd.DataFrame:
    """Fit time = driver effect + fuel*lap + deg*tyre_age on green laps (clean+traffic)
    and return corrected lap times: what the lap would be at lap 1 fuel / new tyres.
    Shared fuel and degradation slopes across the field (per race)."""
    l = categorise(laps)
    g = l[l["category"].isin(["clean", "traffic"])].copy()
    if len(g) < 50:
        l["corrected_s"] = np.nan
        return l
    drivers = sorted(g["driver_id"].unique())
    D = pd.get_dummies(g["driver_id"]).reindex(columns=drivers, fill_value=0).to_numpy(float)
    X = np.column_stack([D, g["lap"].to_numpy(float), g["tyre_age"].to_numpy(float)])
    coef, *_ = np.linalg.lstsq(X, g["time_s"].to_numpy(), rcond=None)
    fuel, deg = coef[-2], coef[-1]
    l["corrected_s"] = l["time_s"] - fuel * l["lap"] - deg * l["tyre_age"]
    l.attrs.update(fuel_s_per_lap=float(fuel), deg_s_per_lap=float(deg))
    return l


def driver_race_pace(laps: pd.DataFrame) -> pd.DataFrame:
    """Per-driver race pace summary: clean-air median, corrected median, gap % to best."""
    l = corrected_pace(laps)
    green = l[l["category"].isin(["clean", "traffic"])]
    clean = l[l["category"] == "clean"]
    out = pd.DataFrame({
        "clean_median_s": clean.groupby("driver_id")["time_s"].median(),
        "corrected_median_s": green.groupby("driver_id")["corrected_s"].median(),
        "corrected_clean_s": clean.groupby("driver_id")["corrected_s"].median(),
        "n_clean": clean.groupby("driver_id").size(),
        "n_traffic": l[l["category"] == "traffic"].groupby("driver_id").size(),
        "stops": l.groupby("driver_id")["pit_this_lap"].sum(),
    }).fillna({"n_clean": 0, "n_traffic": 0})
    # need a reasonable sample; otherwise pace is not comparable
    ok = (out["n_clean"] + out["n_traffic"]) >= 10
    # prefer clean-air laps (traffic slows a car regardless of its pace); fall back to all green laps
    out["corrected_median_s"] = np.where(out["n_clean"] >= 10, out["corrected_clean_s"], out["corrected_median_s"])
    best = out.loc[ok, "corrected_median_s"].min()
    out["race_pace_gap"] = np.where(ok, (out["corrected_median_s"] - best) / best * 100, np.nan)
    out.attrs.update(l.attrs, neutral_laps=int((l.groupby("lap")["category"].first() == "neutral").sum()))
    return out.reset_index().rename(columns={"index": "driver_id"})


def infer_compounds(laps: pd.DataFrame, k: int = 3, seed: int = 0) -> pd.DataFrame:
    """Unsupervised: cluster stints by (pace offset vs driver's mean, degradation
    slope, stint length) with a Gaussian mixture. Clusters are sorted by pace
    offset and named 'fast/medium/durable-like'. INFERRED, not observed."""
    from sklearn.mixture import GaussianMixture
    l = categorise(laps)
    g = l[l["category"].isin(["clean", "traffic"])]
    rows = []
    for (d, s), st in g.groupby(["driver_id", "stint"]):
        if len(st) < 5:
            continue
        slope = np.polyfit(st["tyre_age"], st["time_s"] - 0.06 * st["lap"], 1)[0]  # ~0.06 s/lap fuel burn
        rows.append({"driver_id": d, "stint": s, "n": len(st), "median_s": st["time_s"].median(),
                     "deg_slope": slope, "length": int(l[(l.driver_id == d) & (l.stint == s)].shape[0])})
    st = pd.DataFrame(rows)
    if len(st) < k * 3:
        st["inferred_class"] = "unknown"
        return st
    st["offset"] = st["median_s"] - st.groupby("driver_id")["median_s"].transform("mean")
    Z = st[["offset", "deg_slope", "length"]].to_numpy(float)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    lab = GaussianMixture(k, random_state=seed).fit_predict(Z)
    order = st.assign(lab=lab).groupby("lab")["offset"].mean().sort_values().index
    names = dict(zip(order, ["fast-like", "medium-like", "durable-like"][:k]))
    st["inferred_class"] = [names[x] for x in lab]
    return st


def pace_table(refresh: bool = False) -> pd.DataFrame:
    """Race pace gap per driver for every race with lap data (cached in data/laps/pace_summary.csv)."""
    out_path = LAPS_DIR / "pace_summary.csv"
    files = sorted(p for p in LAPS_DIR.glob("*_*.csv") if p.name != "pace_summary.csv")
    if out_path.exists() and not refresh:
        cached = pd.read_csv(out_path)
        done = set(zip(cached.season, cached["round"]))
        files = [f for f in files if tuple(map(int, f.stem.split("_"))) not in done]
    else:
        cached = None
    rows = []
    for f in files:
        season, rnd = map(int, f.stem.split("_"))
        try:
            rp = driver_race_pace(pd.read_csv(f))
        except Exception:
            continue
        rows.append(rp[["driver_id", "race_pace_gap", "n_clean", "n_traffic", "stops"]].assign(season=season, round=rnd))
    new = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    tab = pd.concat([d for d in (cached, new) if d is not None and len(d)], ignore_index=True) if (rows or cached is not None) else pd.DataFrame(
        columns=["season", "round", "driver_id", "race_pace_gap"])
    if rows and LAPS_DIR.exists():
        tab.to_csv(out_path, index=False)
    return tab
