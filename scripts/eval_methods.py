"""Which way of measuring pace is best? Every 2023+ weekend, each method scored on:

  agree  Spearman rank correlation with the race's classified finishing order
         (higher = the measure explains the result better)
  repeat Spearman between a team's value at one race and at the next race of
         the same season (higher = more signal, less noise)

Race pace (from race laps, measured after the fact):
  raw_median       median of every timed lap
  green_median     median of green laps (no pit, SC/VSC or lap 1)
  clean_median     median of clean-air laps only
  corrected        fixed-effects fit: fuel, real compound, tyre age (OpenF1)
Qualifying / pre-race pace (known before the race, so `agree` is predictive):
  quali_position   official qualifying position
  quali_best       best qualifying lap gap
  quali_ideal      sum of best sectors in qualifying
  fp_best          best practice lap
  fp_longrun       practice long-run pace, compound and tyre corrected
  sprint_pace      sprint race pace (sprint weekends only)

Usage: python scripts/eval_methods.py
"""
import json
import warnings

import numpy as np
import pandas as pd

from f1 import ingest
from f1.analytics import build as B
from f1.analytics import weekend as W
from f1.config import REPORTS


def race_measures(season, rnd):
    w = W.load(season, rnd)
    l = w["laps"]
    r = l[l["session"] == "R"]
    if r.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "raw_median": r.groupby("driver_id")["lap_s"].median(),
        "green_median": r[r["category"].isin(["clean", "traffic"])].groupby("driver_id")["lap_s"].median(),
        "clean_median": r[r["category"] == "clean"].groupby("driver_id")["lap_s"].median(),
    })
    n = r[r["category"].isin(["clean", "traffic"])].groupby("driver_id").size()
    out.loc[n.reindex(out.index).fillna(0) < 10] = np.nan
    return out.reset_index()


def main():
    warnings.simplefilter("ignore")
    res = ingest.load_results()
    t = {k: B.read(k) for k in ("one_lap", "long_runs", "race_pace")}
    ol, lr, rp = t["one_lap"], t["long_runs"], t["race_pace"]
    rows = []
    for season, rnd in B.weekends():
        fin = res[(res.season == season) & (res["round"] == rnd)]
        if fin.empty:
            continue
        m = fin[["driver_id", "constructor_id", "position", "classified", "quali_pos"]].rename(
            columns={"quali_pos": "quali_position"})
        try:
            m = m.merge(race_measures(season, rnd), on="driver_id", how="left")
        except Exception:
            continue
        k = (ol.season == season) & (ol["round"] == rnd)
        q = ol[k & (ol.session == "Q")].set_index("driver_id")
        m["quali_best"] = m["driver_id"].map(q["best_s"])
        m["quali_ideal"] = m["driver_id"].map(q["ideal_s"])
        fp = ol[k & ol.session.str.startswith("FP")].groupby("driver_id")["best_s"].min()
        m["fp_best"] = m["driver_id"].map(fp)
        if len(lr):
            x = lr[(lr.season == season) & (lr["round"] == rnd)]
            if len(x):
                main_s = x.groupby("session")["long_laps"].sum().idxmax()
                m["fp_longrun"] = m["driver_id"].map(x[x.session == main_s].set_index("driver_id")["long_run_s"])
        x = rp[(rp.season == season) & (rp["round"] == rnd)]
        m["corrected"] = m["driver_id"].map(x[x.session == "R"].set_index("driver_id")["race_pace_s"])
        if (x.session == "S").any():
            m["sprint_pace"] = m["driver_id"].map(x[x.session == "S"].set_index("driver_id")["race_pace_s"])
        rows.append(m.assign(season=season, round=rnd))
    D = pd.concat(rows, ignore_index=True)
    race_m = ["raw_median", "green_median", "clean_median", "corrected"]
    pre_m = ["quali_position", "quali_best", "quali_ideal", "fp_best", "fp_longrun", "sprint_pace"]
    out = {}
    for name in race_m + pre_m:
        if name not in D:
            continue
        ag, rep = [], []
        for (s, r), g in D.groupby(["season", "round"]):
            c = g[g["classified"].astype(bool)].dropna(subset=[name])
            if len(c) >= 8:
                ag.append(c[name].corr(c["position"], method="spearman"))
        # team-level repeatability: team best value, ranked within race, next race in season
        tv = D.dropna(subset=[name]).groupby(["season", "round", "constructor_id"])[name].min().reset_index()
        tv["rank"] = tv.groupby(["season", "round"])[name].rank()
        keys = tv[["season", "round"]].drop_duplicates().sort_values(["season", "round"]).values.tolist()
        for (s1, r1), (s2, r2) in zip(keys[:-1], keys[1:]):
            if s1 != s2:
                continue
            a = tv[(tv.season == s1) & (tv["round"] == r1)].set_index("constructor_id")["rank"]
            b = tv[(tv.season == s2) & (tv["round"] == r2)].set_index("constructor_id")["rank"]
            j = pd.concat([a, b], axis=1, join="inner").dropna()
            if len(j) >= 6:
                rep.append(j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman"))
        out[name] = {"group": "race pace (after the race)" if name in race_m else "pre-race (predictive)",
                     "agree_spearman": round(float(np.nanmean(ag)), 3) if ag else None,
                     "agree_ci": [round(float(np.nanpercentile(_boot(ag), q)), 3) for q in (2.5, 97.5)] if len(ag) > 5 else None,
                     "repeat_spearman": round(float(np.nanmean(rep)), 3) if rep else None,
                     "races": len(ag)}
    rep = {"note": "agree = Spearman with the classified finishing order, averaged over races (95% CI by bootstrap "
                   "over races); repeat = team rank correlation between consecutive races of a season.",
           "weekends": int(D[["season", "round"]].drop_duplicates().shape[0]), "methods": out}
    (REPORTS / "methods.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def _boot(x, n=2000, seed=0):
    x = np.asarray([v for v in x if pd.notna(v)])
    rng = np.random.default_rng(seed)
    return x[rng.integers(0, len(x), (n, len(x)))].mean(1)


if __name__ == "__main__":
    main()
