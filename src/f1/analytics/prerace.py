"""Pre-race OpenF1 features per driver per race (2023 onward).

Only sessions that finish before the race are used: FP1-3, sprint
qualifying, the sprint and qualifying. The DNA projection uses earlier
weekends' telemetry and the circuit profile from the circuit's previous visit.
tests/test_leakage.py checks that removing the race session changes nothing.

of1_start_x_hard = log(start position) x passing difficulty, where difficulty
= 1 - (overtakes per lap at this circuit's earlier visits / 0.8), capped to
[0, 1]: grid position should matter more where passing is hard. Circuits
with no earlier visit get the mean difficulty of all earlier races.

Every other feature is centred on the race's field median and missing values are 0,
with `has_of1` = 1 when the weekend has OpenF1 data, so pre-2023 rows and
drivers without a long run are neutral rather than wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import build as B
from . import dna as D
from .telemetry import ZONE_CLASSES

OF1_FEATURES = ["of1_longrun", "of1_fp_best", "of1_sprint_pace", "of1_dna_proj", "of1_ideal_gap",
                "of1_start_x_hard", "has_of1"]
EASY_PASSING = 0.8  # overtakes per lap at the easiest tracks


def _centre(s: pd.Series, cap: float = 5.0) -> pd.Series:
    s = s.clip(upper=cap)
    return s - s.median() if s.notna().any() else s  # e.g. sprint pace on a non-sprint weekend


def table(tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """season, round, driver_id + OF1_FEATURES for every stored weekend."""
    t = tables or {k: B.read(k) for k in ("one_lap", "long_runs", "race_pace", "zones", "weekend")}
    ol = t["one_lap"]
    if ol is None or ol.empty:
        return pd.DataFrame(columns=["season", "round", "driver_id"] + OF1_FEATURES)
    keys = ["season", "round", "driver_id"]
    q = ol[ol["session"].isin(["Q"])][keys + ["constructor_id", "ideal_gap_pct"]]
    fp = ol[ol["session"].isin(["FP1", "FP2", "FP3"])]
    fpb = fp.groupby(keys)["best_s"].min().reset_index()
    fpb["of1_fp_best"] = fpb.groupby(["season", "round"])["best_s"].transform(lambda s: (s / s.min() - 1) * 100)
    base = ol[ol["session"].isin(["FP1", "FP2", "FP3", "SQ", "Q"])][keys + ["constructor_id"]].drop_duplicates(keys)
    out = base.merge(fpb[keys + ["of1_fp_best"]], on=keys, how="left")
    out = out.merge(q[keys + ["ideal_gap_pct"]].rename(columns={"ideal_gap_pct": "of1_ideal_gap"}), on=keys, how="left")
    lr = t["long_runs"]
    if lr is not None and len(lr):
        # driver's long-run gap in the practice session with the most long-run laps
        tot = lr.groupby(["season", "round", "session"])["long_laps"].sum().reset_index().sort_values("long_laps")
        main = tot.groupby(["season", "round"]).tail(1)[["season", "round", "session"]]
        m = lr.merge(main, on=["season", "round", "session"])
        out = out.merge(m[keys + ["long_run_gap_pct"]].rename(columns={"long_run_gap_pct": "of1_longrun"}), on=keys, how="left")
    else:
        out["of1_longrun"] = np.nan
    rp = t["race_pace"]
    if rp is not None and len(rp) and (rp["session"] == "S").any():
        s = rp[rp["session"] == "S"][keys + ["race_pace_gap_pct"]].rename(columns={"race_pace_gap_pct": "of1_sprint_pace"})
        out = out.merge(s, on=keys, how="left")
    else:
        out["of1_sprint_pace"] = np.nan
    out["of1_dna_proj"] = np.nan
    z, w = t.get("zones"), t.get("weekend")
    if z is not None and len(z) and w is not None and len(w):
        tw = D.team_weekend(z, ol, pd.DataFrame(), pd.DataFrame())
        dna = D.rolling_dna(tw)
        prof_cols = [f"share_{k}" for k in ZONE_CLASSES]
        ww = w[["season", "round", "circuit_id"] + prof_cols].sort_values(["season", "round"])
        prev = pd.concat([ww[["season", "round", "circuit_id"]], ww.groupby("circuit_id")[prof_cols].shift(1)], axis=1)
        m = dna.merge(prev, on=["season", "round"]).dropna(subset=prof_cols)
        m = m[m["n_prior"] >= 3]
        if len(m):
            m["of1_dna_proj"] = [D.project(r, r) for _, r in m.iterrows()]
            out = out.drop(columns="of1_dna_proj").merge(
                m[["season", "round", "constructor_id", "of1_dna_proj"]], on=["season", "round", "constructor_id"], how="left")
    out = out.merge(_passing_difficulty(w), on=["season", "round"], how="left") if w is not None and len(w) else \
        out.assign(pass_hard=np.nan)
    feats = ["of1_longrun", "of1_fp_best", "of1_sprint_pace", "of1_dna_proj", "of1_ideal_gap"]
    for f in feats:
        out[f] = out.groupby(["season", "round"])[f].transform(_centre).fillna(0.0)
    out["has_of1"] = 1.0
    out["pass_hard"] = out["pass_hard"].fillna(0.5)
    return out[keys + [f for f in OF1_FEATURES if f != "of1_start_x_hard"] + ["pass_hard"]]


def _passing_difficulty(w: pd.DataFrame, ov: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per race: passing difficulty known before it, regulation-era aware.
    Uses the Jolpica overtake counts (2019 onward) when built, else OpenF1's.
    For each race, only earlier races count; a circuit's earlier visits under the
    same rules dominate, earlier-era visits enter scaled by the era factor
    measured on races before this one (eras.current_estimates)."""
    from ..config import era_of
    from . import build as B
    from . import eras as E
    ov = B.read("overtakes_by_race") if ov is None else ov
    if ov.empty:
        ov = w.rename(columns={"overtakes_per_lap": "per_lap"})[["season", "round", "circuit_id", "per_lap"]].dropna()
        ov["era"] = ov["season"].map(era_of)
    ov = ov.sort_values(["season", "round"]).reset_index(drop=True)
    rows = []
    for s, r, c in w.sort_values(["season", "round"])[["season", "round", "circuit_id"]].itertuples(index=False):
        past = ov[(ov["season"] < s) | ((ov["season"] == s) & (ov["round"] < r))]
        if past.empty:
            rows.append((s, r, np.nan))
            continue
        est = E.current_estimates(past, "per_lap", era_of(int(s)))
        v = est["estimate"].get(c, np.nan)
        if pd.isna(v):
            v = past[past["era"] == past["era"].max()]["per_lap"].mean()
        rows.append((s, r, 1 - min(max(v / EASY_PASSING, 0), 1)))
    return pd.DataFrame(rows, columns=["season", "round", "pass_hard"])


def attach(feats: pd.DataFrame, of1: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge the pre-race OpenF1 features onto model feature rows (0 where a race has none)."""
    of1 = table() if of1 is None else of1
    f = feats.drop(columns=[c for c in OF1_FEATURES + ["pass_hard"] if c in feats.columns])
    f = f.merge(of1, on=["season", "round", "driver_id"], how="left")
    f["of1_start_x_hard"] = f["log_start"] * (f["pass_hard"].fillna(0.5) - 0.5) * f["has_of1"].fillna(0)
    f[OF1_FEATURES] = f[OF1_FEATURES].fillna(0.0)
    return f.drop(columns=["pass_hard"])
