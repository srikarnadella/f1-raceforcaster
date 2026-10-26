"""One weekend of OpenF1 laps: identity join, lap classification, pace tables.

Lap classes (every session):
  invalid   no time, pit-out lap, in-lap, deleted for track limits, or > 2x the
            session's median lap (ran through a red flag / stoppage)
  neutral   SC/VSC/red-flag lap: field median > 115% of the session's green median (race/sprint)
  lap1      race/sprint opening lap
  cool      practice/quali lap > 110% of the driver's session best (cool-down, traffic, aborted)
  push      flying lap within 102% of the driver's best, in a run of < 5 pace laps
  long      lap in a long run: >= 5 pace laps on one set of tyres (practice)
  traffic   race/sprint lap started < 1.5 s behind the car ahead
  clean     race/sprint green lap in clean air
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .. import ingest, openf1

RACE_SESSIONS = ("R", "S")
TRAFFIC_GAP = 1.5
LONG_RUN_MIN = 5
PACE_BAND = 1.07            # practice: pace laps are within 107% of the driver's best
COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]  # real compounds (pacefit adds UNKNOWN)
_DELETED = re.compile(r"CAR (\d+) .*?(?:TIME [\d:.]+ )?DELETED.*?LAP (\d+)")


def available(season: int, rnd: int) -> bool:
    return (openf1.meeting_dir(season, rnd) / "laps.csv").exists()


def identities(season: int, rnd: int, drivers: pd.DataFrame) -> pd.DataFrame:
    """driver_number -> code, driver_id, constructor_id, team_name, team_colour.
    Race entrants come from the result store; practice-only drivers (FP1 rookies)
    get their team's constructor_id via team_name."""
    d = drivers.sort_values("session").drop_duplicates("driver_number", keep="last")
    d = d.rename(columns={"name_acronym": "code"})[["driver_number", "code", "full_name", "team_name", "team_colour"]]
    try:
        res = ingest.load_results([season])
        res = res[res["round"] == rnd][["code", "driver_id", "constructor_id"]]
    except Exception:
        res = pd.DataFrame(columns=["code", "driver_id", "constructor_id"])
    if res.empty:  # upcoming race: use the latest round's entrants
        try:
            r = ingest.load_results([season])
            r = r[r["round"] == r["round"].max()]
            res = r[["code", "driver_id", "constructor_id"]]
        except Exception:
            pass
    d = d.merge(res, on="code", how="left")
    team_map = d.dropna(subset=["constructor_id"]).groupby("team_name")["constructor_id"].first()
    d["constructor_id"] = d["constructor_id"].fillna(d["team_name"].map(team_map))
    d["constructor_id"] = d["constructor_id"].fillna(d["team_name"].str.lower().str.replace(" ", "_"))
    d["driver_id"] = d["driver_id"].fillna(d["code"].str.lower())
    d["team_colour"] = "#" + d["team_colour"].fillna("888888").astype(str).str.lstrip("#")
    return d


def deleted_laps(rc: pd.DataFrame) -> set[tuple[str, int, int]]:
    out = set()
    for s, m in zip(rc.get("session", []), rc.get("message", [])):
        if isinstance(m, str) and "DELETED" in m and "REINSTATED" not in m:
            g = _DELETED.search(m)
            if g:
                out.add((s, int(g.group(1)), int(g.group(2))))
    return out


def load(season: int, rnd: int) -> dict:
    """{'laps': classified laps, 'sessions', 'drivers', 'race_control'} for one weekend."""
    d = openf1.meeting_dir(season, rnd)
    laps = pd.read_csv(d / "laps.csv")
    drivers = pd.read_csv(d / "drivers.csv")
    sessions = pd.read_csv(d / "sessions.csv")
    rc = pd.read_csv(d / "race_control.csv") if (d / "race_control.csv").exists() else pd.DataFrame(
        columns=["session", "message"])
    ids = identities(season, rnd, drivers)
    laps = laps.merge(ids, on="driver_number", how="left")
    laps["code"] = laps["code"].fillna(laps["driver_number"].astype(str))
    laps["compound"] = laps["compound"].where(laps["compound"].isin(COMPOUND_ORDER))
    laps = classify(laps, deleted_laps(rc))
    laps.insert(0, "round", rnd)
    laps.insert(0, "season", season)
    return {"laps": laps, "sessions": sessions, "drivers": ids, "race_control": rc}


def classify(laps: pd.DataFrame, deleted=frozenset()) -> pd.DataFrame:
    out = []
    for sess, l in laps.groupby("session", sort=False):
        l = l.sort_values(["driver_number", "lap"]).copy()
        nxt_out = l.groupby("driver_number")["pit_out"].shift(-1).astype("boolean").fillna(False).astype(bool)
        nxt_stint = l.groupby("driver_number")["stint"].shift(-1)
        in_lap = nxt_out | (nxt_stint.notna() & (nxt_stint != l["stint"]))
        dele = np.array([(sess, int(n), int(k)) in deleted for n, k in zip(l["driver_number"], l["lap"])], bool)
        # a "lap" that ran through a red flag or stoppage is timed but is not a lap (seen: 1,130 s in FP2)
        stopped = l["lap_s"] > 2.0 * l["lap_s"].median()
        invalid = l["lap_s"].isna() | l["pit_out"].astype(bool) | in_lap | dele | stopped
        cat = np.full(len(l), "", dtype=object)
        cat[invalid.to_numpy()] = "invalid"
        if sess in RACE_SESSIONS:
            cat = _race_classes(l, cat)
        else:
            cat = _practice_classes(l, cat)
        l["category"] = cat
        l["deleted"] = dele
        out.append(l)
    return pd.concat(out, ignore_index=True)


def _race_classes(l, cat):
    ok = cat == ""
    t = l["lap_s"].to_numpy(float)
    green_med = np.nanmedian(t[ok]) if ok.any() else np.nan
    field_med = l.groupby("lap")["lap_s"].transform("median").to_numpy()
    cat[ok & (field_med > 1.15 * green_med)] = "neutral"
    cat[(l["lap"] == 1).to_numpy() & (cat == "")] = "lap1"
    # gap to the car ahead when the lap started: line crossings in order on the same lap number
    start = pd.to_datetime(l["date_start"], utc=True, format="ISO8601")
    tmp = pd.DataFrame({"lap": l["lap"].to_numpy(), "t": start.to_numpy()}, index=l.index)
    tmp = tmp.sort_values(["lap", "t"])
    gap = tmp.groupby("lap")["t"].diff().dt.total_seconds()
    l["gap_ahead"] = gap.reindex(l.index)
    cat[(cat == "") & (l["gap_ahead"] < TRAFFIC_GAP).to_numpy()] = "traffic"
    cat[cat == ""] = "clean"
    return cat


def _practice_classes(l, cat):
    best = l[cat == ""].groupby("driver_number")["lap_s"].min()
    ratio = (l["lap_s"] / l["driver_number"].map(best)).to_numpy()
    cat[(cat == "") & (ratio > 1.10)] = "cool"
    pace = (cat == "") & (ratio <= PACE_BAND)
    run_n = pd.Series(pace, index=l.index).groupby([l["driver_number"], l["stint"].fillna(-1)]).transform("sum").to_numpy()
    cat[pace & (run_n >= LONG_RUN_MIN)] = "long"
    cat[pace & (run_n < LONG_RUN_MIN) & (ratio <= 1.02)] = "push"
    cat[cat == ""] = "cool"
    return cat


# ------------------------------------------------------------------ pace tables

def one_lap(laps: pd.DataFrame) -> pd.DataFrame:
    """Best lap, ideal lap (best sectors) and top speed per driver per session."""
    v = laps[laps["category"] != "invalid"]
    g = v.groupby(["session", "driver_id"])
    out = g.agg(code=("code", "first"), constructor_id=("constructor_id", "first"),
                team_name=("team_name", "first"), team_colour=("team_colour", "first"),
                best_s=("lap_s", "min"), s1=("s1", "min"), s2=("s2", "min"), s3=("s3", "min"),
                top_speed=("st", "max"), laps=("lap", "size")).reset_index()
    best_comp = v.loc[v.groupby(["session", "driver_id"])["lap_s"].idxmin(), ["session", "driver_id", "compound"]]
    out = out.merge(best_comp.rename(columns={"compound": "best_compound"}), on=["session", "driver_id"], how="left")
    out["ideal_s"] = out[["s1", "s2", "s3"]].sum(1, min_count=3)
    out["gap_pct"] = out.groupby("session")["best_s"].transform(lambda s: (s / s.min() - 1) * 100)
    out["ideal_gap_pct"] = out.groupby("session")["ideal_s"].transform(lambda s: (s / s.min() - 1) * 100)
    return out


def long_runs(laps: pd.DataFrame, session: str | None = None) -> pd.DataFrame:
    """Practice long-run pace per driver, corrected for compound and tyre age with a
    shared fit:  lap = driver + compound + deg[compound] * tyre_age.
    Fuel load is unknown in practice; it is absorbed by the driver term."""
    l = laps[(laps["category"] == "long") & laps["compound"].notna()]
    if session:
        l = l[l["session"] == session]
    if len(l) < 30 or l["driver_id"].nunique() < 4:
        return pd.DataFrame()
    fit = _fe_fit(l, fuel=False)
    if fit is None:
        return pd.DataFrame()
    eff, extra = fit
    g = l.groupby("driver_id")
    out = pd.DataFrame({"code": g["code"].first(), "constructor_id": g["constructor_id"].first(),
                        "team_name": g["team_name"].first(), "team_colour": g["team_colour"].first(),
                        "long_laps": g.size(), "main_compound": g["compound"].agg(lambda s: s.mode().iat[0]),
                        "raw_median_s": g["lap_s"].median()})
    out["long_run_s"] = eff.reindex(out.index)
    out["long_run_gap_pct"] = (out["long_run_s"] / out["long_run_s"].min() - 1) * 100
    out.attrs.update(extra)
    return out.reset_index()


def race_pace(laps: pd.DataFrame, session: str = "R") -> pd.DataFrame:
    """Race clean-air pace per driver corrected for fuel, compound and tyre age
    (lap = driver + compound + deg[compound] * tyre_age + fuel * lap)."""
    l = laps[(laps["session"] == session) & laps["category"].isin(["clean", "traffic"]) & laps["compound"].notna()]
    if len(l) < 100:
        return pd.DataFrame()
    use = l[l["category"] == "clean"]
    use = use if use.groupby("driver_id").size().median() >= 10 else l
    fit = _fe_fit(use, fuel=True)
    if fit is None:
        return pd.DataFrame()
    eff, extra = fit
    g = l.groupby("driver_id")
    out = pd.DataFrame({"code": g["code"].first(), "constructor_id": g["constructor_id"].first(),
                        "team_name": g["team_name"].first(), "team_colour": g["team_colour"].first(),
                        "green_laps": g.size(), "clean_laps": use.groupby("driver_id").size(),
                        "clean_median_s": l[l["category"] == "clean"].groupby("driver_id")["lap_s"].median()})
    out["race_pace_s"] = eff.reindex(out.index)
    ok = out["clean_laps"].fillna(0) >= 8
    out.loc[~ok, "race_pace_s"] = np.nan
    out["race_pace_gap_pct"] = (out["race_pace_s"] / out["race_pace_s"].min() - 1) * 100
    out.attrs.update(extra)
    return out.reset_index()


def _fe_fit(l, fuel: bool):
    """Driver race/long-run pace at reference conditions; see analytics/pacefit.py."""
    from .pacefit import fit
    return fit(l, fuel)


def degradation(laps: pd.DataFrame, fuel_s_per_lap: float = -0.06) -> pd.DataFrame:
    """Per team and compound: tyre degradation slope (s/lap of tyre age) from race
    green laps (fuel-corrected) and practice long runs (fuel unknown, so labelled)."""
    rows = []
    for (sess, team, comp), g in laps[laps["category"].isin(["clean", "traffic", "long"]) & laps["compound"].notna()].groupby(
            ["session", "constructor_id", "compound"]):
        g = g[g["lap_s"] < g["lap_s"].median() * 1.05]
        if len(g) < 8 or g["tyre_age"].nunique() < 4:
            continue
        # within-stint slope: demean each stint, then regress on tyre age (and lap for fuel in races)
        y = g["lap_s"] - g.groupby(["driver_id", "stint"])["lap_s"].transform("mean")
        a = g["tyre_age"] - g.groupby(["driver_id", "stint"])["tyre_age"].transform("mean")
        if sess in RACE_SESSIONS:
            # remove fuel burn (laps get faster as fuel burns: fuel_s_per_lap < 0)
            y = y - fuel_s_per_lap * (g["lap"] - g.groupby(["driver_id", "stint"])["lap"].transform("mean"))
        if (a ** 2).sum() == 0:
            continue
        slope = float((a * y).sum() / (a ** 2).sum())
        rows.append({"session": sess, "constructor_id": team, "team_name": g["team_name"].iat[0],
                     "team_colour": g["team_colour"].iat[0], "compound": comp, "deg_s_per_lap": round(slope, 4),
                     "laps": len(g), "fuel_corrected": sess in RACE_SESSIONS})
    return pd.DataFrame(rows)


def stints(laps: pd.DataFrame, session: str = "R") -> pd.DataFrame:
    l = laps[laps["session"] == session]
    g = l.groupby(["driver_id", "stint"])
    return g.agg(code=("code", "first"), team_colour=("team_colour", "first"), compound=("compound", "first"),
                 lap_start=("lap", "min"), lap_end=("lap", "max"), tyre_age_start=("tyre_age", "min"),
                 median_s=("lap_s", "median")).reset_index()


def race_story(laps: pd.DataFrame, session: str = "R") -> pd.DataFrame:
    """Position and gap to the leader at the end of every lap (from line-crossing times)."""
    l = laps[laps["session"] == session].copy()
    if l.empty:
        return l
    st = pd.to_datetime(l["date_start"], utc=True, format="ISO8601")
    l["t_end"] = st + pd.to_timedelta(l["lap_s"].fillna(0), unit="s")
    # a lap's end is the next lap's start when known (handles laps without a time)
    nxt = l.sort_values("lap").groupby("driver_id")["date_start"].shift(-1)
    l["t_end"] = pd.to_datetime(nxt, utc=True, format="ISO8601").fillna(l["t_end"])
    l = l[l["lap_s"].notna() | nxt.notna()]
    l["position"] = l.groupby("lap")["t_end"].rank(method="first").astype(int)
    lead = l.groupby("lap")["t_end"].transform("min")
    l["gap_to_leader_s"] = (l["t_end"] - lead).dt.total_seconds()
    return l[["driver_id", "code", "team_colour", "lap", "position", "gap_to_leader_s", "lap_s", "compound",
              "category"]].sort_values(["lap", "position"])


def pit_loss(laps: pd.DataFrame, session: str = "R") -> float:
    """Median time cost of a green-flag stop: (in-lap + out-lap) - 2 x the
    driver's median clean lap. Stops on neutralised laps are left out."""
    l = laps[laps["session"] == session].sort_values(["driver_id", "lap"])
    if l.empty:
        return np.nan
    field_med = l.groupby("lap")["lap_s"].transform("median")
    green_med = l.loc[l["category"] == "clean", "lap_s"].median()
    ref = l[l["category"] == "clean"].groupby("driver_id")["lap_s"].median()
    out = []
    for d, g in l.groupby("driver_id"):
        g = g.reset_index(drop=True)
        for i in g.index[g["pit_out"].astype(bool) & (g["lap"] > 1)]:
            if i == 0 or pd.isna(g.loc[i, "lap_s"]) or pd.isna(g.loc[i - 1, "lap_s"]):
                continue
            if field_med.loc[l.index[(l["driver_id"] == d)]].iloc[[i - 1, i]].max() > 1.10 * green_med:
                continue
            out.append(g.loc[i - 1, "lap_s"] + g.loc[i, "lap_s"] - 2 * ref.get(d, np.nan))
    out = [x for x in out if 5 < x < 60]
    return float(np.median(out)) if out else np.nan


def overtakes(story: pd.DataFrame, laps: pd.DataFrame, session: str = "R") -> dict:
    """On-track passes in one race (definition in analytics/overtaking.py)."""
    from .overtaking import count_passes, green_set
    l = laps[laps["session"] == session]
    pos = story.pivot_table(index="lap", columns="driver_id", values="position")
    n, made, lost = count_passes(pos, green_set(l, ["clean", "traffic"]))
    laps_n = int(pos.index.max()) if len(pos) else 0
    return {"overtakes": n, "race_laps": laps_n, "overtakes_per_lap": n / laps_n if laps_n else np.nan,
            "by_driver": made, "conceded": lost}


def driver_race_traits(laps: pd.DataFrame, fuel_s_per_lap: float = -0.06, session: str = "R") -> pd.DataFrame:
    """Per driver in one race: tyre wear (within-stint slope, fuel-corrected, s/lap of
    tyre age) and consistency (std of clean-air laps around each stint's own trend)."""
    l = laps[(laps["session"] == session) & laps["category"].isin(["clean", "traffic"]) & laps["tyre_age"].notna()]
    rows = []
    for d, g in l.groupby("driver_id"):
        slopes, resid, w = [], [], []
        for _, st in g.groupby("stint"):
            st = st[st["lap_s"] < st["lap_s"].median() * 1.05]
            if len(st) < 6 or st["tyre_age"].nunique() < 4:
                continue
            y = st["lap_s"] - fuel_s_per_lap * st["lap"]
            b, a = np.polyfit(st["tyre_age"], y, 1)
            slopes.append(b)
            w.append(len(st))
            resid.extend(list(y - (a + b * st["tyre_age"])))
        if slopes:
            rows.append({"driver_id": d, "deg_s_per_lap": float(np.average(slopes, weights=w)),
                         "consistency_s": float(np.std(resid)), "stint_laps": int(sum(w))})
    return pd.DataFrame(rows)
