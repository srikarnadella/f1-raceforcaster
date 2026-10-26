"""Regulation eras: keep them apart, measure how a new rule set changes each
track, and predict tracks the new rules have not raced yet.

Eras (config.ERA_STARTS): 2014-16, 2017-21, 2022-25 (ground effect), 2026-.

1. Overtakes per race from Jolpica lap positions (2019 onward, every era):
   pairs of drivers who swap places between consecutive laps while both are on
   green laps (not lap 1, no pit in/out lap, no SC/VSC lap). The same counting
   rule in every era, so eras are comparable.
2. Era multiplier per metric: for circuits raced in both eras, the ratio
   new-era mean / old-era mean at the same circuit; the era factor is their
   geometric mean, with a bootstrap 90% interval over circuits. "Overtakes are
   1.5x per track under the 2026 rules" is this number.
3. Track-profile model (overtakes): Poisson regression of passes per lap on the
   circuit's layout (share of the lap on straights and in slow / medium / fast
   corners, top speed, braking) plus an era term, so it can predict a circuit
   the current rules have not raced yet.
4. Current-era estimate per circuit and metric: the circuit's own current-era
   mean when it exists, shrunk toward its previous-era mean x era factor; with
   no current-era race, the scaled previous-era mean; with no history at all,
   the track-layout model (it is worse than a circuit's own record, so it only
   fills the gaps: walk-forward check on 2026).
5. Walk-forward check over the current era's races: each race predicted from
   everything before it, against "same as last visit" and "previous era only".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import LAPS_DIR, era_of

ERA_NAMES = {2014: "2014–16 hybrid", 2017: "2017–21 wide cars", 2022: "2022–25 ground effect", 2026: "2026 rules"}
LAYOUT = ["share_straight", "share_slow", "share_medium", "share_fast", "top_speed_kmh", "braking_pct"]
SHRINK = 1.0  # weight (in races) of the prior when blending with a circuit's own current-era races


def era_name(season: int) -> str:
    return ERA_NAMES.get(era_of(season), str(era_of(season)))


# ------------------------------------------------------------------ overtakes from Jolpica laps

def count_overtakes(laps: pd.DataFrame) -> dict:
    """Passes in one race from Jolpica laps (definition in analytics/overtaking.py)."""
    from ..laps import categorise
    from .overtaking import count_passes, green_set
    l = categorise(laps)
    green = green_set(l, ["clean", "traffic", "slow"])  # Jolpica's own classes; 'slow' is still green
    pos = l.pivot_table(index="lap", columns="driver_id", values="position")
    n, _, _ = count_passes(pos, green)
    return {"overtakes": n, "race_laps": int(pos.index.max()) if len(pos) else 0,
            "green_laps": int(l.loc[l["category"].isin(["clean", "traffic", "slow"]), "lap"].nunique())}


OVERTAKE_METHOD = "count_passes-v1"  # bump when the pass definition or lap classes change


def overtakes_table(results: pd.DataFrame, cache: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per race with stored laps: season, round, circuit_id, era, overtakes, per lap."""
    if cache is not None and ("method" not in cache or (cache["method"] != OVERTAKE_METHOD).any()):
        cache = None  # counted by an older definition: recount everything
    done = set(zip(cache["season"], cache["round"])) if cache is not None and len(cache) else set()
    circ = results.drop_duplicates(["season", "round"]).set_index(["season", "round"])["circuit_id"]
    rows = [] if cache is None else cache.to_dict("records")
    for p in sorted(LAPS_DIR.glob("*_*.csv")):
        if p.name == "pace_summary.csv":
            continue
        s, r = map(int, p.stem.split("_"))
        if (s, r) in done or (s, r) not in circ.index:
            continue
        try:
            o = count_overtakes(pd.read_csv(p))
        except Exception:
            continue
        rows.append({"season": s, "round": r, "circuit_id": circ[(s, r)], "era": era_of(s), **o,
                     "method": OVERTAKE_METHOD})
    t = pd.DataFrame(rows)
    if len(t):
        t["per_lap"] = t["overtakes"] / t["race_laps"].clip(lower=1)
        t = t.sort_values(["season", "round"]).reset_index(drop=True)
    return t


# ------------------------------------------------------------------ era factors

FACTOR_SHRINK = 10.0  # pseudo-circuits of "no change" (factor 1) the measured factor is shrunk toward


def era_factor(df: pd.DataFrame, metric: str, new: int, old: int, n_boot: int = 2000, seed: int = 0,
               shrink: float | None = None) -> dict:
    """Geometric-mean ratio of a circuit metric between two eras over circuits raced in both."""
    a = df[df["era"] == new].groupby("circuit_id")[metric].mean()
    b = df[df["era"] == old].groupby("circuit_id")[metric].mean()
    j = pd.concat([a, b], axis=1, keys=["new", "old"]).dropna()
    j = j[(j["new"] > 0) & (j["old"] > 0)]
    if len(j) < 3:
        return {"metric": metric, "new_era": new, "old_era": old, "factor": np.nan, "lo": np.nan, "hi": np.nan,
                "circuits": int(len(j))}
    lr = np.log(j["new"] / j["old"]).to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.exp(lr[rng.integers(0, len(lr), (n_boot, len(lr)))].mean(1))
    # shrink toward "the new rules change nothing" while few circuits have been raced under both:
    # three early races once said 1.5x and over-scaled every track (walk-forward 2026)
    k = FACTOR_SHRINK if shrink is None else shrink
    return {"metric": metric, "new_era": new, "old_era": old,
            "factor": float(np.exp(lr.sum() / (len(lr) + k))), "raw_factor": float(np.exp(lr.mean())),
            "lo": float(np.percentile(boot, 5)), "hi": float(np.percentile(boot, 95)), "circuits": int(len(j)),
            "per_circuit": {c: float(v) for c, v in (j["new"] / j["old"]).items()}}


# ------------------------------------------------------------------ track-profile model

def fit_profile_model(ov: pd.DataFrame, layout: pd.DataFrame, alpha: float = 0.5):
    """Poisson regression: passes per lap ~ era + circuit layout. Returns (model, columns, scaler)."""
    from sklearn.linear_model import PoissonRegressor
    d = ov.merge(layout[LAYOUT], left_on="circuit_id", right_index=True, how="inner").dropna(subset=LAYOUT)
    if len(d) < 10:
        return None
    eras = sorted(d["era"].unique())
    mu, sd = d[LAYOUT].mean(), d[LAYOUT].std() + 1e-9
    X = _design(d, eras, mu, sd)
    m = PoissonRegressor(alpha=alpha / len(d), max_iter=1000).fit(X, d["per_lap"], sample_weight=d["race_laps"])
    return {"model": m, "eras": eras, "mu": mu, "sd": sd}


def _design(d, eras, mu, sd):
    Z = ((d[LAYOUT] - mu) / sd).to_numpy()
    E = np.column_stack([(d["era"] == e).to_numpy(float) for e in eras[1:]]) if len(eras) > 1 else np.zeros((len(d), 0))
    return np.column_stack([Z, E])


def predict_profile(pm, circuits: pd.DataFrame, era: int) -> pd.Series:
    if pm is None:
        return pd.Series(np.nan, index=circuits.index)
    d = circuits[LAYOUT].copy().dropna()
    use_era = era if era in pm["eras"] else pm["eras"][-1]
    d["era"] = use_era
    return pd.Series(pm["model"].predict(_design(d, pm["eras"], pm["mu"], pm["sd"])), index=d.index)


# ------------------------------------------------------------------ current-era estimates

def current_estimates(df: pd.DataFrame, metric: str, era: int, prior: pd.Series | None = None) -> pd.DataFrame:
    """Per circuit: current-era estimate of `metric` and how it was made."""
    eras = sorted(e for e in df["era"].unique() if e < era)
    old = eras[-1] if eras else None
    f = era_factor(df, metric, era, old) if old is not None else {"factor": np.nan}
    cur = df[df["era"] == era].groupby("circuit_id")[metric].agg(["mean", "count"])
    prev = df[df["era"] == old].groupby("circuit_id")[metric].mean() if old is not None else pd.Series(dtype=float)
    idx = sorted(set(cur.index) | set(prev.index) | (set(prior.index) if prior is not None else set()))
    rows = []
    for c in idx:
        scaled = prev.get(c, np.nan) * f["factor"] if pd.notna(f["factor"]) else np.nan
        if pd.isna(scaled) and pd.notna(prev.get(c, np.nan)):
            scaled = prev.get(c)   # no era factor yet (first races of an era): the previous era as is
        # the circuit's own previous-era record wins; the layout model only fills circuits with no history
        # (walk-forward 2026: averaging the two was worse than the scaled record alone)
        layout = prior.get(c, np.nan) if prior is not None else np.nan
        p0 = scaled if pd.notna(scaled) else layout
        if c in cur.index and cur.loc[c, "count"] > 0:
            n, m = cur.loc[c, "count"], cur.loc[c, "mean"]
            est = (n * m + SHRINK * p0) / (n + SHRINK) if pd.notna(p0) else m
            how = f"{int(n)} race(s) under these rules" + (", shrunk toward the scaled prior" if pd.notna(p0) else "")
        else:
            est = p0
            how = ("previous era × era factor" if pd.notna(scaled) else
                   "track-layout model (no earlier visit)" if pd.notna(layout) else "no data")
        rows.append({"circuit_id": c, "estimate": est, "current_era_mean": cur["mean"].get(c, np.nan),
                     "current_era_races": int(cur["count"].get(c, 0)), "previous_era_mean": prev.get(c, np.nan),
                     "scaled_previous": scaled, "profile_model": prior.get(c, np.nan) if prior is not None else np.nan,
                     "method": how})
    out = pd.DataFrame(rows).set_index("circuit_id")
    out.attrs["factor"] = f
    return out


def validate(ov: pd.DataFrame, layout: pd.DataFrame, era: int) -> tuple[pd.DataFrame, dict]:
    """Walk-forward over the era's races: predict each race's passes per lap from earlier races only."""
    ov = ov.sort_values(["season", "round"]).reset_index(drop=True)
    rows = []
    for i, r in ov[ov["era"] == era].iterrows():
        past = ov.iloc[:i]
        pm = fit_profile_model(past, layout)
        prof = predict_profile(pm, layout, era)
        est = current_estimates(past, "per_lap", era, prof)
        c = r["circuit_id"]
        last = past[past["circuit_id"] == c]
        prev_era = past[(past["circuit_id"] == c) & (past["era"] < era)]
        rows.append({"season": r["season"], "round": r["round"], "circuit_id": c, "actual": r["per_lap"],
                     "model": est["estimate"].get(c, np.nan),
                     "last_visit": last["per_lap"].iat[-1] if len(last) else np.nan,
                     "previous_era_only": prev_era[prev_era["era"] == prev_era["era"].max()]["per_lap"].mean()
                     if len(prev_era) else np.nan})
    v = pd.DataFrame(rows)
    res: dict = {}
    if v.empty:
        return v, res
    # every method scored on the same races (those where all three can predict), plus the model on all
    common = v[["model", "last_visit", "previous_era_only", "actual"]].notna().all(axis=1)
    for k in ("model", "last_visit", "previous_era_only"):
        res[k] = {"mae_per_lap": round(float((v.loc[common, k] - v.loc[common, "actual"]).abs().mean()), 3),
                  "n": int(common.sum())}
    ok = v["model"].notna() & v["actual"].notna()
    res["model_all_races"] = {"mae_per_lap": round(float((v.loc[ok, "model"] - v.loc[ok, "actual"]).abs().mean()), 3),
                              "n": int(ok.sum())}
    return v, res
