"""Pre-weekend win probabilities: before any session runs.

Monte Carlo over qualifying, then the champion race model on each simulated grid:
  team one-lap gap  ~ Normal(DNA x circuit projection, sigma_team)
  driver lap        = team lap x (1 + teammate split / 2) + Normal(0, sigma_driver)
sigma_team comes from the projection's own walk-forward error (MAE x 1.25,
the normal-distribution ratio of std to mean absolute error), so the spread
is measured, not guessed. Each simulated grid gives the champion model's
weekend features (start position, quali gap, teammate gap); form features
are the real ones. Win/podium probabilities average over grids.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import ingest
from ..config import SEED
from ..features import build_features, weekend_features
from ..models import RaceModel, simulate_orders

SIGMA_DRIVER_PCT = 0.15
BASE_LAP_S = 80.0


def lineup(results: pd.DataFrame, season: int) -> pd.DataFrame:
    r = results[results["season"] == season]
    r = r[r["round"] == r["round"].max()]
    return r[["driver_id", "code", "constructor_id", "number"]].drop_duplicates("driver_id")


def simulate(season: int, rnd: int, team_gap: pd.DataFrame, tm_split: pd.Series, sigma_team: float,
             spec: dict, n_grids: int = 300, n_races: int = 200, results: pd.DataFrame | None = None) -> pd.DataFrame:
    """team_gap: constructor_id, gap_pct (projected one-lap gap to the best team).
    tm_split: driver_id -> teammate quali delta % (negative = faster)."""
    results = ingest.load_results() if results is None else results
    cal = ingest.load_calendar()
    ev = cal[(cal["season"] == season) & (cal["round"] == rnd)].iloc[0]
    hist = results[(results["season"] < season) | ((results["season"] == season) & (results["round"] < rnd))]
    lu = lineup(hist, season).merge(team_gap[["constructor_id", "gap_pct"]], on="constructor_id", how="left")
    lu["gap_pct"] = lu["gap_pct"].fillna(team_gap["gap_pct"].max() if len(team_gap) else 2.0)
    lu["split"] = lu["driver_id"].map(tm_split).fillna(0.0).clip(-0.6, 0.6)
    entry = lu.assign(season=season, round=rnd, race_name=ev.race_name, date=ev.race_utc.strftime("%Y-%m-%d"),
                      circuit_id=ev.circuit_id, format=ev.format, quali_pos=np.arange(1, len(lu) + 1),
                      q1=BASE_LAP_S, q2=np.nan, q3=np.nan, sprint_pos=np.nan, sprint_classified=np.nan,
                      sprint_points=0.0)
    entry = entry.reindex(columns=ingest.RESULT_COLS)
    feats = build_features(hist, entry, **spec.get("form", {}))
    model = RaceModel(spec).fit(feats[feats["position"].notna()])
    race = feats[(feats["season"] == season) & (feats["round"] == rnd)].reset_index(drop=True)
    rng = np.random.default_rng(SEED + season * 100 + rnd)
    teams = lu["constructor_id"].unique()
    win = np.zeros(len(race))
    top3 = np.zeros(len(race))
    pole = np.zeros(len(race))
    gap = race["driver_id"].map(lu.set_index("driver_id")["gap_pct"]).to_numpy()
    split = race["driver_id"].map(lu.set_index("driver_id")["split"]).to_numpy()
    con = race["constructor_id"].to_numpy()
    for g in range(n_grids):
        tnoise = dict(zip(teams, rng.normal(0, sigma_team, len(teams))))
        pct = gap + np.array([tnoise.get(c, 0.0) for c in con]) + split / 2 + rng.normal(0, SIGMA_DRIVER_PCT, len(race))
        lap = BASE_LAP_S * (1 + pct / 100)
        r = race.copy()
        r["q1"], r["q2"], r["q3"] = lap, np.nan, np.nan
        r["quali_pos"] = lap.argsort().argsort() + 1
        w = weekend_features(r)
        for c in ("quali_gap", "tm_quali_gap", "start_pos", "log_start", "sprint_delta"):
            r[c] = w[c].to_numpy()
        pos = simulate_orders(model.strength.score(r), model.dnf.predict(r), n_races, rng)
        win += (pos == 1).mean(0)
        top3 += (pos <= 3).mean(0)
        pole += (r["quali_pos"].to_numpy() == 1)
    out = race[["driver_id", "code", "constructor_id"]].copy()
    out["p_pole"], out["p_win"], out["p_top3"] = pole / n_grids, win / n_grids, top3 / n_grids
    out.attrs.update(sigma_team=sigma_team, n_grids=n_grids, n_races=n_races)
    return out.sort_values("p_win", ascending=False).reset_index(drop=True)
