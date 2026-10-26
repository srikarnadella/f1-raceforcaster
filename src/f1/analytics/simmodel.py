"""The race simulator as a forecasting model with the same fit/predict interface
as models.RaceModel, so the frozen walk-forward harness can score it.

Every input for a race comes from earlier races only: driver and car profiles
entering the race (profiles.build), circuit behaviour (laps, pit loss, safety
cars, tyre wear, strategy) from earlier visits, passing ease from the
regulation-era estimate on earlier races. The one same-weekend input is
qualifying (the grid and the one-lap gaps), which is known before the start.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SEED, era_of
from ..models import summarize
from . import build as B
from . import circuits as C
from . import eras as E
from . import strategy as ST
from .racesim import Circuit, SimParams, simulate


class SimInputs:
    def __init__(self, tables: dict | None = None, results: pd.DataFrame | None = None):
        from .. import ingest
        from . import profiles
        self.t = tables or {k: B.read(k) for k in ("one_lap", "race_pace", "driver_traits", "race_story", "overtakes",
                                                   "weekend")}
        self.results = ingest.load_results() if results is None else results
        P = profiles.build(self.t, self.results)
        self.latest = P.attrs["latest"].set_index("driver_id")
        self.P = P.set_index(["season", "round", "driver_id"])
        self.ov = B.read("overtakes_by_race")
        self._circ: dict[tuple[int, int], Circuit] = {}

    def circuit(self, season: int, rnd: int, cid: str) -> Circuit:
        key = (season, rnd)
        if key in self._circ:
            return self._circ[key]
        w = self.t["weekend"]
        prior = w[(w["season"] < season) | ((w["season"] == season) & (w["round"] < rnd))]
        c = Circuit()
        if len(prior):
            prof = C.profiles(prior)
            if cid in prof.index:
                p = prof.loc[cid]
                c.laps = int(p["race_laps"]) if pd.notna(p.get("race_laps")) else c.laps
                c.pit_loss = float(p["pit_loss_s"]) if pd.notna(p.get("pit_loss_s")) else c.pit_loss
                c.sc_per_race = float(p["sc_rate"]) if pd.notna(p.get("sc_rate")) else c.sc_per_race
                plans = ST.for_circuit(p, prof)
                if plans is not None and len(plans):
                    best = plans.iloc[0]
                    tot = sum(best["stints"])
                    c.plan = [(comp, L / tot) for comp, L in zip(best["compounds"], best["stints"])]
                    c.deg, c.offset = plans.attrs["deg"], plans.attrs["offset"]
        ov = self.ov
        if len(ov):
            past = ov[(ov["season"] < season) | ((ov["season"] == season) & (ov["round"] < rnd))]
            if len(past):
                est = E.current_estimates(past, "per_lap", era_of(season))
                v = est["estimate"].get(cid, np.nan)
                if pd.isna(v):
                    v = past[past["era"] == past["era"].max()]["per_lap"].mean()
                c.passing_ease = float(min(max(v / 0.8, 0.02), 1.0))
        self._circ[key] = c
        return c

    def drivers(self, race: pd.DataFrame, prm: SimParams) -> pd.DataFrame:
        s, r = int(race["season"].iloc[0]), int(race["round"].iloc[0])
        d = race[["driver_id", "start_pos", "quali_gap"]].copy()
        if (s, r) in self.P.index.droplevel(2):
            prof = self.P.reindex([(s, r, x) for x in d["driver_id"]])
        else:  # upcoming race: profiles after the most recent race are the ones entering it
            prof = self.latest.reindex(d["driver_id"].to_numpy())
        prof.index = d.index
        prof = prof.fillna({"car_race": prof["car_race"].median() if prof["car_race"].notna().any() else 1.0})
        from .profiles import PRIORS
        prof = prof.fillna(PRIORS)
        q = race[["q1", "q2", "q3"]].min(axis=1)
        base = float(q.min()) * 1.06 if q.notna().any() else 90.0
        prior_pct = prof["car_race"] + prof["race_skill"] / 2
        pct = prm.w_quali * d["quali_gap"].clip(upper=3.0) + (1 - prm.w_quali) * prior_pct
        d["pace_s"] = base * pct / 100
        d["base_lap_s"] = base
        d["constructor_id"] = race["constructor_id"].to_numpy()
        d["sigma"] = (prm.lap_noise * (1 + prof["consistency"].clip(-0.3, 0.6))).clip(lower=0.1)
        car_deg = prof["car_deg"] / max(float(prof["car_deg"].median()), 1e-3)
        d["deg_mult"] = (car_deg * (1 + 10 * prof["tyre_skill"])).clip(0.5, 1.6)
        d["start_skill"] = prof["start_skill"].clip(-2, 2)
        d["attack"], d["defence"] = prof["attack"].clip(-3, 3), prof["defence"].clip(-3, 3)
        d["p_dnf"] = prof["car_dnf"].clip(0.02, 0.3)
        return d


class RaceSimModel:
    def __init__(self, inputs: SimInputs, params: SimParams | None = None):
        self.inp, self.prm = inputs, params or SimParams()

    def fit(self, feats: pd.DataFrame):
        return self  # parameters are set by calibration on earlier seasons (scripts/race_sim_lab.py)

    def predict(self, race: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
        race = race.reset_index(drop=True)
        s, r = int(race["season"].iloc[0]), int(race["round"].iloc[0])
        circ = self.inp.circuit(s, r, race["circuit_id"].iloc[0])
        d = self.inp.drivers(race, self.prm)
        pos = simulate(d, circ, self.prm, seed=seed + s * 100 + r)
        return summarize(race, pos)


class Blend:
    """Average of two models' probabilities (and expected positions)."""

    def __init__(self, a, b, w: float = 0.5):
        self.a, self.b, self.w = a, b, w

    def fit(self, feats):
        self.a.fit(feats)
        self.b.fit(feats)
        return self

    def predict(self, race, seed=SEED):
        pa, pb = self.a.predict(race), self.b.predict(race)
        m = pa.merge(pb[["driver_id", "p_win", "p_top3", "p_top10", "exp_pos"]], on="driver_id", suffixes=("", "_b"))
        for c in ("p_win", "p_top3", "p_top10", "exp_pos"):
            m[c] = self.w * m[c] + (1 - self.w) * m[f"{c}_b"]
        m = m.drop(columns=[c for c in m.columns if c.endswith("_b")])
        m = m.sort_values(["exp_pos", "p_win"], ascending=[True, False]).reset_index(drop=True)
        m["pred_rank"] = np.arange(1, len(m) + 1)
        return m
