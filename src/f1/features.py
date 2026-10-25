"""Leak-free features.

`build_features` walks races in chronological order. For each race it first
computes features from (a) that weekend's pre-race sessions (qualifying,
sprint) and (b) running state built from *earlier* races only, and only then
folds the race's result into the state. A race's own result can never reach
its features. tests/test_leakage.py enforces this.

Weekend features are field-relative (gap to pole, teammate delta), so they
survive regulation changes better than raw times.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import era_of

QUALI_GAP_CAP = 5.0       # % gap to pole; drivers without a time get the cap
NO_HISTORY_POS = 15.0     # prior for finishing position when a driver/team has no history
EARLY_ERA_RACES = 6

# Columns that hold the race outcome. They must never be used as features.
OUTCOME_COLS = ["grid", "position", "position_text", "classified", "status", "points", "laps"]

STRENGTH_FEATURES = [
    "log_start", "quali_gap", "tm_quali_gap", "sprint_delta",
    "drv_form_pos", "con_form_pos", "drv_form_quali", "drv_gain",
    "drv_form_pos_early", "con_form_pos_early", "no_history",
]
FORM_ONLY_FEATURES = ["drv_form_pos", "con_form_pos", "drv_form_quali",
                      "drv_form_pos_early", "con_form_pos_early", "no_history"]
DNF_FEATURES = ["con_dnf_rate", "drv_dnf_rate", "circuit_dnf_rate", "log_start", "early_era"]


class _EWM:
    """Exponentially weighted mean per key, with half-life in observations."""

    def __init__(self, half_life: float, prior: float):
        self.a = 1 - 0.5 ** (1 / half_life)
        self.prior = prior
        self.v: dict = {}
        self.n: dict = defaultdict(int)

    def get(self, k):
        return self.v.get(k, self.prior)

    def update(self, k, x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return
        self.v[k] = x if k not in self.v else (1 - self.a) * self.v[k] + self.a * x
        self.n[k] += 1


def weekend_features(race: pd.DataFrame) -> pd.DataFrame:
    """Features computable from one weekend's pre-race sessions."""
    r = race.copy()
    n = len(r)
    qpos = r["quali_pos"].fillna(n).clip(upper=n)
    best = r[["q1", "q2", "q3"]].min(axis=1)
    pole = best.min()
    r["quali_gap"] = ((best - pole) / pole * 100).clip(upper=QUALI_GAP_CAP).fillna(QUALI_GAP_CAP)
    tm_mean = r.groupby("constructor_id")["quali_gap"].transform("sum")
    tm_n = r.groupby("constructor_id")["quali_gap"].transform("count")
    r["tm_quali_gap"] = np.where(tm_n > 1, r["quali_gap"] - (tm_mean - r["quali_gap"]) / (tm_n - 1).clip(lower=1), 0.0)
    has_sprint = r["sprint_pos"].notna()
    # 2021-22: the sprint set the race grid. Since 2023 it is a separate race.
    sprint_sets_grid = int(r["season"].iloc[0]) in (2021, 2022) and has_sprint.any()
    start = r["sprint_pos"].fillna(n) if sprint_sets_grid else qpos
    r["start_pos"] = start.astype(float)
    r["log_start"] = np.log(r["start_pos"])
    r["sprint_delta"] = np.where(has_sprint, np.log(r["sprint_pos"].fillna(n)) - np.log(qpos), 0.0)
    r["is_sprint_weekend"] = float(has_sprint.any())
    return r


def build_features(results: pd.DataFrame, entry: pd.DataFrame | None = None,
                   form_half_life: float = 6.0, dnf_half_life: float = 12.0) -> pd.DataFrame:
    """Feature rows for every race in `results`, plus `entry` (an upcoming race
    with pre-race data only) if given. Rows of each race only see earlier races."""
    races = results if entry is None else pd.concat([results, entry], ignore_index=True)
    races = races.sort_values(["season", "round"], kind="stable")

    drv_pos = _EWM(form_half_life, NO_HISTORY_POS)
    con_pos = _EWM(form_half_life, NO_HISTORY_POS)
    drv_q = _EWM(form_half_life, 2.0)
    drv_gain = _EWM(form_half_life, 0.0)
    con_dnf = _EWM(dnf_half_life, 0.12)
    drv_dnf = _EWM(dnf_half_life, 0.12)
    circ_dnf_sum, circ_dnf_n = defaultdict(float), defaultdict(int)
    global_dnf = [0.0, 0]
    era_count = defaultdict(int)

    out = []
    for (season, rnd), race in races.groupby(["season", "round"], sort=False):
        r = weekend_features(race)
        era = era_of(int(season))
        early = float(era_count[era] < EARLY_ERA_RACES)
        circ = r["circuit_id"].iloc[0]
        g = (global_dnf[0] + 1.2) / (global_dnf[1] + 10)
        r["races_in_era"] = era_count[era]
        r["early_era"] = early
        r["drv_form_pos"] = r["driver_id"].map(drv_pos.get)
        r["con_form_pos"] = r["constructor_id"].map(con_pos.get)
        r["drv_form_quali"] = r["driver_id"].map(drv_q.get)
        r["drv_gain"] = r["driver_id"].map(drv_gain.get)
        r["no_history"] = r["driver_id"].map(lambda d: float(drv_pos.n[d] == 0))
        r["drv_form_pos_early"] = r["drv_form_pos"] * early
        r["con_form_pos_early"] = r["con_form_pos"] * early
        r["con_dnf_rate"] = r["constructor_id"].map(con_dnf.get)
        r["drv_dnf_rate"] = r["driver_id"].map(drv_dnf.get)
        # shrink circuit DNF rate toward the global rate (strength of 40 driver-starts)
        r["circuit_dnf_rate"] = (circ_dnf_sum[circ] + 40 * g) / (circ_dnf_n[circ] + 40)
        out.append(r)

        # ---- fold this race's outcome into the state (after its features are fixed) ----
        if r["position"].isna().all():
            continue  # upcoming race: no result
        era_count[era] += 1
        dnf = (~r["classified"].astype(bool)).astype(float)
        for d, c, pos, cl, st, q, dn in zip(r["driver_id"], r["constructor_id"], r["position"],
                                             r["classified"], r["start_pos"], r["quali_gap"], dnf):
            if cl:
                drv_pos.update(d, float(pos))
                drv_gain.update(d, float(st - pos))
            drv_q.update(d, float(q))
            drv_dnf.update(d, dn)
        cls = r[r["classified"].astype(bool)]
        for c, m in cls.groupby("constructor_id")["position"].mean().items():
            con_pos.update(c, float(m))
        for c, m in pd.Series(dnf.values, index=r["constructor_id"]).groupby(level=0).mean().items():
            con_dnf.update(c, float(m))
        circ_dnf_sum[circ] += dnf.sum()
        circ_dnf_n[circ] += len(r)
        global_dnf[0] += dnf.sum()
        global_dnf[1] += len(r)

    return pd.concat(out, ignore_index=True)


def feature_hash(feats: pd.DataFrame, columns: list[str]) -> str:
    """Stable hash of the feature values a forecast used."""
    cols = ["driver_id"] + sorted(set(columns))
    blob = feats[cols].round(8).to_csv(index=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
