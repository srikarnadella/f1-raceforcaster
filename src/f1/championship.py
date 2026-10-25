"""Championship Monte Carlo.

From the standings after `after_round`, simulate every remaining race (and
sprint) with the same Plackett-Luce + DNF simulation used for race forecasts.
Future weekends have no qualifying yet, so strengths come from the form-only
variant of the model. A per-driver strength perturbation, fixed for the rest of
the season within each simulation, represents uncertainty about how the
pecking order will move. Assumes the current line-up runs every remaining race.
Fastest-lap bonus points (2019-2024) are not simulated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RACE_POINTS, SEED, sprint_points
from .features import build_features
from .models import RaceModel, form_only_spec, simulate_orders


def standings(results: pd.DataFrame, season: int, after_round: int) -> tuple[pd.Series, pd.Series]:
    r = results[(results["season"] == season) & (results["round"] <= after_round)]
    pts = r["points"] + r["sprint_points"].fillna(0)
    drv = pts.groupby(r["driver_id"]).sum()
    con = pts.groupby(r["constructor_id"]).sum()
    return drv, con


def _points_table(points: list[int], n: int) -> np.ndarray:
    t = np.zeros(n + 1)
    t[1:len(points) + 1] = points[:n]
    return t


def simulate(results: pd.DataFrame, calendar: pd.DataFrame, season: int, after_round: int, spec: dict,
             n_sims: int = 5000, strength_noise: float = 0.35, seed: int = SEED) -> dict:
    history = results[(results["season"] < season) |
                      ((results["season"] == season) & (results["round"] <= after_round))]
    last = history[(history["season"] == season) & (history["round"] == after_round)]
    lineup = last[["driver_id", "code", "constructor_id"]].reset_index(drop=True)
    n = len(lineup)
    remaining = calendar[(calendar["season"] == season) & (calendar["round"] > after_round)]

    drv_pts, con_pts = standings(results, season, after_round)
    cur = lineup["driver_id"].map(drv_pts).fillna(0).to_numpy()

    totals = np.tile(cur, (n_sims, 1)).astype(float)
    if len(remaining):
        fspec = form_only_spec(spec)
        nxt = int(remaining["round"].iloc[0])
        entry = lineup.assign(season=season, round=nxt, circuit_id="__future__", format="conventional",
                              quali_pos=np.nan, q1=np.nan, q2=np.nan, q3=np.nan, sprint_pos=np.nan,
                              sprint_points=0.0, race_name="future", date="")
        feats = build_features(history, entry, **spec.get("form", {}))
        race = feats[(feats["season"] == season) & (feats["round"] == nxt)]
        race = race.set_index("driver_id").loc[lineup["driver_id"]].reset_index()
        model = RaceModel(fspec).fit(feats[feats["position"].notna()])
        base = model.strength.score(race)
        p_dnf = model.dnf.predict(race)
        rng = np.random.default_rng(seed + season * 100 + after_round)
        drift = rng.normal(0, strength_noise, (n_sims, n))
        rpts = _points_table(RACE_POINTS, n)
        spts = _points_table(sprint_points(season), n)
        for _, ev in remaining.iterrows():
            sessions = [rpts] + ([spts] if ev["format"] == "sprint" else [])
            for table in sessions:
                # one shared draw per sim with the sim's drifted strengths
                key = base[None, :] + drift + rng.gumbel(size=(n_sims, n))
                dnf = rng.random((n_sims, n)) < p_dnf[None, :]
                key = np.where(dnf, -1e6 + rng.random((n_sims, n)), key)
                order = np.argsort(-key, axis=1)
                pos = np.empty_like(order)
                pos[np.arange(n_sims)[:, None], order] = np.arange(1, n + 1)[None, :]
                totals += table[pos]

    # final ranks with random tie-breaks
    tieb = np.random.default_rng(seed).random(totals.shape) * 1e-3
    drank = (-(totals + tieb)).argsort(1).argsort(1) + 1
    cons = lineup["constructor_id"].to_numpy()
    cids = sorted(set(cons) | set(con_pts.index[con_pts.index.isin(cons)]))
    ctot = np.stack([totals[:, cons == c].sum(1) for c in cids], 1)
    # constructor points scored by drivers no longer in the line-up still count
    lineup_con_pts = pd.Series(cur, index=cons).groupby(level=0).sum()
    ctot += np.array([con_pts.get(c, 0) - lineup_con_pts.get(c, 0) for c in cids])[None, :]
    crank = (-(ctot + tieb[:, :len(cids)])).argsort(1).argsort(1) + 1

    drivers = pd.DataFrame({
        "driver_id": lineup["driver_id"], "code": lineup["code"], "constructor_id": lineup["constructor_id"],
        "points": cur, "exp_points": totals.mean(0),
        "p_title": (drank == 1).mean(0),
    })
    drivers["rank_dist"] = [np.bincount(drank[:, i], minlength=n + 1)[1:] / n_sims for i in range(n)]
    constructors = pd.DataFrame({
        "constructor_id": cids, "points": [con_pts.get(c, 0) for c in cids],
        "exp_points": ctot.mean(0), "p_title": (crank == 1).mean(0),
    })
    constructors["rank_dist"] = [np.bincount(crank[:, i], minlength=len(cids) + 1)[1:] / n_sims
                                 for i in range(len(cids))]
    return {"season": season, "after_round": after_round, "remaining": int(len(remaining)),
            "drivers": drivers.sort_values("p_title", ascending=False).reset_index(drop=True),
            "constructors": constructors.sort_values("p_title", ascending=False).reset_index(drop=True)}


def to_json(sim: dict) -> dict:
    def rows(df):
        d = df.copy()
        d["rank_dist"] = d["rank_dist"].map(lambda a: [round(float(x), 4) for x in a])
        return d.round(4).to_dict("records")
    return {k: (rows(v) if isinstance(v, pd.DataFrame) else v) for k, v in sim.items()}
