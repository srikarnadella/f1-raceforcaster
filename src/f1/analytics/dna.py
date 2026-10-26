"""Team and driver "DNA": where each car and driver gains or loses time, and
projections of team pace at a given circuit from those strengths.

Team DNA (per zone class, % vs field median; negative = faster):
    the team's fastest qualifying lap, split into straights and slow / medium /
    fast corners (telemetry.py), exponentially weighted over earlier weekends of
    the same regulation era (half-life HALF_LIFE weekends).

Circuit-fit projection of a team's one-lap gap at circuit c:
    gap(team, c) = sum_k share_k(c) * dna_k(team)
i.e. the share of the lap the circuit spends in each zone type times how good
the team is in that zone type. Validated walk-forward (each weekend projected
from earlier weekends only) against "same gap as the season so far" and "same
gap as last weekend" in `validate_projection`.

Driver DNA: teammate-relative (same car), so it isolates the driver:
zone deltas, qualifying vs race pace, opening-lap places gained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import era_of
from .telemetry import ZONE_CLASSES

HALF_LIFE = 5.0


def _order(df):
    return df.sort_values(["season", "round"])


def team_weekend(zones: pd.DataFrame, one_lap: pd.DataFrame, race_pace: pd.DataFrame,
                 long_runs: pd.DataFrame) -> pd.DataFrame:
    """One row per team per weekend: zone-class pace of the team's faster lap,
    qualifying gap, practice long-run gap and race pace gap (all % to the best team)."""
    keys = ["season", "round", "circuit_id", "constructor_id"]
    z = zones.copy()
    fast = z.loc[z.groupby(["season", "round", "constructor_id"])["lap_s"].idxmin().dropna()]
    fastest = fast[["season", "round", "constructor_id", "driver_id"]]
    z = z.merge(fastest, on=["season", "round", "constructor_id", "driver_id"])
    dnaw = z.pivot_table(index=keys, columns="kind", values="vs_median_pct").add_prefix("z_")
    if "speed_vs_median_pct" in z.columns:
        dnaw = dnaw.join(z.pivot_table(index=keys, columns="kind", values="speed_vs_median_pct").add_prefix("v_"))
    dnaw = dnaw.reset_index()
    q = one_lap[one_lap["session"] == "Q"].groupby(keys).agg(q_best=("best_s", "min"), team_name=("team_name", "first"),
                                                              team_colour=("team_colour", "first")).reset_index()
    q["quali_gap_pct"] = q.groupby(["season", "round"])["q_best"].transform(lambda s: (s / s.min() - 1) * 100)
    q["quali_vs_median_pct"] = q.groupby(["season", "round"])["q_best"].transform(lambda s: (s / s.median() - 1) * 100)
    out = q.merge(dnaw, on=keys, how="left")
    if len(race_pace):
        rp = race_pace[race_pace["session"] == "R"].groupby(keys)["race_pace_gap_pct"].min().reset_index()
        rp["race_pace_gap_pct"] = rp.groupby(["season", "round"])["race_pace_gap_pct"].transform(lambda s: s - s.min())
        out = out.merge(rp, on=keys, how="left")
    if len(long_runs):
        lr = long_runs.sort_values("long_laps", ascending=False)
        # session with the most long-run laps per weekend
        main = lr.groupby(["season", "round", "session"])["long_laps"].sum().reset_index().sort_values("long_laps")
        main = main.groupby(["season", "round"]).tail(1)[["season", "round", "session"]]
        lr = lr.merge(main, on=["season", "round", "session"])
        t = lr.groupby(keys)["long_run_gap_pct"].min().reset_index()
        t["longrun_gap_pct"] = t.groupby(["season", "round"])["long_run_gap_pct"].transform(lambda s: s - s.min())
        out = out.merge(t[keys + ["longrun_gap_pct"]], on=keys, how="left")
    out["era"] = out["season"].map(era_of)
    return _order(out).reset_index(drop=True)


def rolling_dna(tw: pd.DataFrame, half_life: float = HALF_LIFE) -> pd.DataFrame:
    """DNA *entering* each weekend: EWM of earlier weekends in the same era only."""
    a = 1 - 0.5 ** (1 / half_life)
    cols = [f"z_{k}" for k in ZONE_CLASSES] + [f"v_{k}" for k in ZONE_CLASSES] + ["quali_vs_median_pct"]
    rows: list[dict] = []
    state: dict[tuple, dict] = {}
    count: dict[tuple, int] = {}
    for (season, rnd), g in tw.groupby(["season", "round"], sort=True):
        era = int(g["era"].iat[0])
        for _, r in g.iterrows():
            k = (era, r.constructor_id)
            rows.append({"season": season, "round": rnd, "constructor_id": r.constructor_id,
                         "n_prior": count.get(k, 0), **{f"dna_{c}": state.get(k, {}).get(c, np.nan) for c in cols}})
        for _, r in g.iterrows():
            k = (era, r.constructor_id)
            s = state.setdefault(k, {})
            for c in cols:
                v = r.get(c)
                if pd.notna(v):
                    s[c] = v if c not in s else (1 - a) * s[c] + a * v
            count[k] = count.get(k, 0) + 1
    return pd.DataFrame(rows)


def latest_dna(tw: pd.DataFrame, half_life: float = HALF_LIFE) -> pd.DataFrame:
    """DNA after the most recent weekend (for projecting the next one)."""
    last = tw[tw["era"] == tw["era"].max()]
    a = 1 - 0.5 ** (1 / half_life)
    cols = [f"z_{k}" for k in ZONE_CLASSES] + [f"v_{k}" for k in ZONE_CLASSES] + ["quali_vs_median_pct"]
    out: dict[str, dict] = {}
    for _, r in _order(last).iterrows():
        s = out.setdefault(r.constructor_id, {"team_name": r.team_name, "team_colour": r.team_colour, "n": 0})
        s["team_name"], s["team_colour"] = r.team_name, r.team_colour
        for c in cols:
            if pd.notna(r.get(c)):
                s[c] = r[c] if c not in s else (1 - a) * s[c] + a * r[c]
        s["n"] += 1
    return pd.DataFrame(out).T.rename_axis("constructor_id").reset_index()


def project(dna_row: pd.Series | dict, profile: pd.Series | dict, prefix: str = "dna_") -> float:
    """Projected lap-time difference to the field median (%) at a circuit."""
    v = sum(float(profile[f"share_{k}"]) * float(dna_row[f"{prefix}z_{k}"]) for k in ZONE_CLASSES
            if pd.notna(dna_row.get(f"{prefix}z_{k}")))
    return v


def validate_projection(tw: pd.DataFrame, weekend: pd.DataFrame, min_prior: int = 3) -> tuple[pd.DataFrame, dict]:
    """Walk-forward: project each team's quali gap from earlier weekends' DNA and
    this circuit's profile *as measured at the circuit's previous visit*
    (so this weekend's telemetry never enters its own projection)."""
    dna = rolling_dna(tw)
    prof_cols = [f"share_{k}" for k in ZONE_CLASSES]
    w = _order(weekend[["season", "round", "circuit_id"] + prof_cols].dropna())
    prev_prof = w.groupby("circuit_id")[prof_cols].shift(1)
    w = pd.concat([w[["season", "round", "circuit_id"]], prev_prof], axis=1).dropna()
    m = tw.merge(dna, on=["season", "round", "constructor_id"]).merge(w, on=["season", "round", "circuit_id"])
    m = m[m["n_prior"] >= min_prior].copy()
    if m.empty:
        return m.assign(actual=np.nan, proj_gap=np.nan, blend_gap=np.nan, season_avg_gap=np.nan, last_gap=np.nan), {}
    m["proj_vs_median"] = [project(r, r) for _, r in m.iterrows()]
    # all three predictors are re-centred to "gap to the best team" per weekend
    def gap(s):
        return s - s.min()
    m["proj_gap"] = m.groupby(["season", "round"])["proj_vs_median"].transform(gap)
    m["season_avg_gap"] = m.groupby(["season", "round"])["dna_quali_vs_median_pct"].transform(gap)
    m["actual"] = m["quali_gap_pct"]
    # blend: DNA projection adjusted toward the team's recent overall level
    m["blend_vs_median"] = 0.5 * m["proj_vs_median"] + 0.5 * m["dna_quali_vs_median_pct"]
    m["blend_gap"] = m.groupby(["season", "round"])["blend_vs_median"].transform(gap)
    prev = tw[["season", "round", "constructor_id", "quali_gap_pct"]].copy()
    prev["last_gap"] = prev.sort_values(["season", "round"]).groupby("constructor_id")["quali_gap_pct"].shift(1)
    m = m.merge(prev[["season", "round", "constructor_id", "last_gap"]], on=["season", "round", "constructor_id"], how="left")
    res = {}
    for name in ["proj_gap", "blend_gap", "season_avg_gap", "last_gap"]:
        ok = m[name].notna() & m["actual"].notna()
        err = (m.loc[ok, name] - m.loc[ok, "actual"]).abs()
        rk = pd.Series([g[name].corr(g["actual"], method="spearman")
                        for _, g in m[ok].groupby(["season", "round"])], dtype=float)
        res[name] = {"mae_pct": round(float(err.mean()), 3), "rank_spearman": round(float(rk.mean()), 3),
                     "n": int(ok.sum())}
    return m, res


def driver_dna(zones: pd.DataFrame, one_lap: pd.DataFrame, race_pace: pd.DataFrame,
               story: pd.DataFrame, results: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    """Teammate-relative driver traits (negative time = faster than the teammate)."""
    z = zones if season is None else zones[zones["season"] == season]
    tm = z.groupby(["season", "round", "constructor_id", "kind"])["time_s"].transform("mean")
    n = z.groupby(["season", "round", "constructor_id", "kind"])["time_s"].transform("count")
    z = z.assign(tm_delta_pct=np.where(n == 2, (z["time_s"] / tm - 1) * 2 * 100, np.nan))
    out = z.pivot_table(index=["driver_id", "code"], columns="kind", values="tm_delta_pct", aggfunc="median").add_prefix("tm_")
    out["weekends"] = z.groupby(["driver_id", "code"])[["season", "round"]].apply(lambda g: len(g.drop_duplicates()))
    q = one_lap[(one_lap["session"] == "Q")]
    q = q if season is None else q[q["season"] == season]
    tq = q.groupby(["season", "round", "constructor_id"])["best_s"].transform("mean")
    nq = q.groupby(["season", "round", "constructor_id"])["best_s"].transform("count")
    q = q.assign(tm_q=np.where(nq == 2, (q["best_s"] / tq - 1) * 2 * 100, np.nan))
    out = out.join(q.groupby(["driver_id", "code"])["tm_q"].median().rename("tm_quali_pct"), how="outer")
    if len(race_pace):
        r = race_pace[race_pace["session"] == "R"]
        r = r if season is None else r[r["season"] == season]
        tr = r.groupby(["season", "round", "constructor_id"])["race_pace_s"].transform("mean")
        nr = r.groupby(["season", "round", "constructor_id"])["race_pace_s"].transform("count")
        r = r.assign(tm_r=np.where(nr == 2, (r["race_pace_s"] / tr - 1) * 2 * 100, np.nan))
        out = out.join(r.groupby(["driver_id", "code"])["tm_r"].median().rename("tm_race_pct"), how="outer")
        # positive = relatively better on Sunday than on Saturday
        out["race_minus_quali"] = out["tm_quali_pct"] - out["tm_race_pct"]
    if len(story) and len(results):
        s = story if season is None else story[story["season"] == season]
        l1 = s[s["lap"] == 1][["season", "round", "driver_id", "position"]].rename(columns={"position": "pos_lap1"})
        g = results[["season", "round", "driver_id", "grid"]]
        l1 = l1.merge(g, on=["season", "round", "driver_id"])
        l1 = l1[l1["grid"] > 0]
        l1["lap1_gain"] = l1["grid"] - l1["pos_lap1"]
        out = out.join(l1.groupby("driver_id")["lap1_gain"].mean().rename("lap1_gain"), on="driver_id")
    return out.reset_index()
