"""Driver and car profiles: the objects the race simulator races against each other.

Driver traits are teammate-relative (same car), so they describe the driver.
They carry across regulation eras with a long memory. Car traits are team
level and only carry within an era. Every profile is the state *entering* a
race, from earlier races only (exponentially weighted).

Driver (negative = better unless noted):
  quali_skill    % one-lap time vs teammate
  race_skill     % race pace vs teammate (fuel, tyre corrected)
  tyre_skill     s/lap tyre wear vs teammate
  consistency    s, lap-time scatter vs teammate
  start_skill    places gained on lap 1 beyond the teammate (positive = better)
  attack         green-flag passes made per race vs teammate (positive = better)
  defence        green-flag passes conceded per race vs teammate
Car (per team, within era):
  car_race       % race pace gap to the best car
  car_quali      % one-lap gap to the best car
  car_deg        s/lap tyre wear (team mean)
  car_dnf        share of entries not classified
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import era_of

DRIVER_HL, CAR_HL = 10.0, 4.0
DRIVER_TRAITS = ["quali_skill", "race_skill", "tyre_skill", "consistency", "start_skill", "attack", "defence"]
CAR_TRAITS = ["car_race", "car_quali", "car_deg", "car_dnf"]
PRIORS = {"quali_skill": 0.0, "race_skill": 0.0, "tyre_skill": 0.0, "consistency": 0.0, "start_skill": 0.0,
          "attack": 0.0, "defence": 0.0, "car_race": 1.0, "car_quali": 1.0, "car_deg": 0.06, "car_dnf": 0.12}


def _tm_delta(df, value, keys=("season", "round", "constructor_id"), pct=True):
    """Driver value minus teammate value (only when the team has exactly two)."""
    g = df.groupby(list(keys))[value]
    n, tot = g.transform("count"), g.transform("sum")
    other = (tot - df[value]) / (n - 1).where(n > 1)
    d = (df[value] / other - 1) * 100 if pct else df[value] - other
    return d.where(n == 2)


def observations(t: dict, results: pd.DataFrame) -> pd.DataFrame:
    """One row per driver per race: what that race showed (an outcome, used only for later races)."""
    keys = ["season", "round", "driver_id"]
    base = results[results["season"] >= 2023][keys + ["constructor_id", "grid", "classified", "circuit_id"]].copy()
    q = t["one_lap"][t["one_lap"]["session"] == "Q"][keys + ["constructor_id", "best_s"]]
    q = q.assign(quali_skill=_tm_delta(q, "best_s")).drop(columns=["constructor_id"])
    rp = t["race_pace"][t["race_pace"]["session"] == "R"][keys + ["constructor_id", "race_pace_s", "race_pace_gap_pct"]]
    rp = rp.assign(race_skill=_tm_delta(rp, "race_pace_s")).drop(columns=["constructor_id", "race_pace_s"])
    o = base.merge(q, on=keys, how="left").merge(rp, on=keys, how="left")
    tr = t.get("driver_traits", pd.DataFrame())
    if len(tr):
        o = o.merge(tr[keys + ["deg_s_per_lap", "consistency_s"]], on=keys, how="left")
        o["tyre_skill"] = _tm_delta(o.assign(v=o["deg_s_per_lap"]), "v", pct=False)
        o["consistency"] = _tm_delta(o.assign(v=o["consistency_s"]), "v", pct=False)
    st = t.get("race_story", pd.DataFrame())
    if len(st):
        l1 = st[st["lap"] == 1][keys + ["position"]].rename(columns={"position": "pos_lap1"})
        o = o.merge(l1, on=keys, how="left")
        o["gain1"] = (o["grid"].where(o["grid"] > 0) - o["pos_lap1"]).clip(-6, 6)
        o["start_skill"] = _tm_delta(o.assign(v=o["gain1"]), "v", pct=False)
    ov = t.get("overtakes", pd.DataFrame())
    if len(ov):
        o = o.merge(ov[keys + ["overtakes", "conceded"]], on=keys, how="left")
        raced = o["season"].astype(str) + "_" + o["round"].astype(str)
        has = raced.isin(set(ov["season"].astype(str) + "_" + ov["round"].astype(str)))
        # teammate-relative, so a slow car starting at the back does not look like a great overtaker
        made = o["overtakes"].where(has).fillna(0).where(has)
        lost = o["conceded"].where(has).fillna(0).where(has)
        o["attack"] = _tm_delta(o.assign(v=made), "v", pct=False)
        o["defence"] = _tm_delta(o.assign(v=lost), "v", pct=False)
    return o


def build(t: dict, results: pd.DataFrame) -> pd.DataFrame:
    """Profiles entering every 2023+ race, for every entrant."""
    o = observations(t, results).sort_values(["season", "round"])
    da, ca = 1 - 0.5 ** (1 / DRIVER_HL), 1 - 0.5 ** (1 / CAR_HL)
    dstate: dict[str, dict] = {}
    cstate: dict[tuple, dict] = {}
    dn: dict[str, int] = {}
    rows = []
    for (s, r), g in o.groupby(["season", "round"], sort=True):
        era = era_of(int(s))
        for _, x in g.iterrows():
            d = dstate.get(x.driver_id, {})
            c = cstate.get((era, x.constructor_id), {})
            rows.append({"season": s, "round": r, "driver_id": x.driver_id, "constructor_id": x.constructor_id,
                         "races_profiled": dn.get(x.driver_id, 0),
                         **{k: d.get(k, PRIORS[k]) for k in DRIVER_TRAITS},
                         **{k: c.get(k, PRIORS[k]) for k in CAR_TRAITS}})
        # fold this race in (after its rows are written)
        team = g.groupby("constructor_id").agg(car_race=("race_pace_gap_pct", "min"), car_deg=("deg_s_per_lap", "mean"),
                                               car_dnf=("classified", lambda v: 1 - v.astype(bool).mean()))
        qb = g.groupby("constructor_id")["best_s"].min()
        team["car_quali"] = (qb / qb.min() - 1) * 100
        team["car_race"] = team["car_race"] - team["car_race"].min()
        for cid, v in team.iterrows():
            c = cstate.setdefault((era, cid), {})
            for k in CAR_TRAITS:
                if pd.notna(v.get(k)):
                    c[k] = v[k] if k not in c else (1 - ca) * c[k] + ca * v[k]
        for _, x in g.iterrows():
            d = dstate.setdefault(x.driver_id, {})
            for k in DRIVER_TRAITS:
                val = x.get(k)
                if val is not None and pd.notna(val):
                    val = float(np.clip(val, -3, 3)) if k in ("quali_skill", "race_skill") else float(val)
                    d[k] = val if k not in d else (1 - da) * d[k] + da * val
            dn[x.driver_id] = dn.get(x.driver_id, 0) + 1
    P = pd.DataFrame(rows)
    # state after the last race, for the drivers of that race (profile cards, next-race simulation)
    s, r = o[["season", "round"]].drop_duplicates().iloc[-1]
    era = era_of(int(s))
    last = o[(o["season"] == s) & (o["round"] == r)]
    P.attrs["latest"] = pd.DataFrame([{"driver_id": x.driver_id, "constructor_id": x.constructor_id,
                                       "races_profiled": dn.get(x.driver_id, 0),
                                       **{k: dstate.get(x.driver_id, {}).get(k, PRIORS[k]) for k in DRIVER_TRAITS},
                                       **{k: cstate.get((era, x.constructor_id), {}).get(k, PRIORS[k]) for k in CAR_TRAITS}}
                                      for _, x in last.iterrows()])
    return P


def archetypes(prof: pd.DataFrame, k: int = 4, seed: int = 0) -> pd.DataFrame:
    """Unsupervised driver archetypes from the skill vector (k-means on standardised traits)."""
    from sklearn.cluster import KMeans
    d = prof[prof["races_profiled"] >= 8].copy()
    if len(d) < k * 2:
        d["archetype"] = "too few races"
        return d
    X = d[DRIVER_TRAITS].to_numpy(float)
    sign = np.array([-1, -1, -1, -1, 1, 1, -1])  # flip so higher = better for every trait
    Z = (X * sign - (X * sign).mean(0)) / ((X * sign).std(0) + 1e-9)
    km = KMeans(k, n_init=20, random_state=seed).fit(Z)
    names = {}
    label = {"quali_skill": "one-lap specialist", "race_skill": "Sunday racer", "tyre_skill": "tyre whisperer",
             "consistency": "metronome", "start_skill": "fast starter", "attack": "overtaker", "defence": "defender"}
    for c in range(k):
        cen = pd.Series(km.cluster_centers_[c], index=DRIVER_TRAITS).sort_values(ascending=False)
        names[c] = label[cen.index[0]] + (" · " + label[cen.index[1]] if cen.iloc[1] > 0.3 else "")
    d["archetype"] = [names[c] for c in km.labels_]
    Zdf = pd.DataFrame(Z, columns=[f"z_{t}" for t in DRIVER_TRAITS], index=d.index)
    return d.join(Zdf)
