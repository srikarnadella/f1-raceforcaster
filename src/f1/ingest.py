"""Build the persistent data store from Jolpica.

Outputs (committed to the repo):
  data/calendar.csv            one row per race weekend, session start times (UTC)
  data/results/<season>.csv    one row per driver per race: qualifying, sprint, race result

Finished seasons are written once. Rows for completed races are never rewritten
(historical result data is protected); a race is only added once its results exist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import jolpica, schemas
from .config import CALENDAR_CSV, FIRST_SEASON, RESULTS_DIR

RESULT_COLS = [
    "season", "round", "race_name", "date", "circuit_id", "format",
    "driver_id", "code", "constructor_id", "number",
    "quali_pos", "q1", "q2", "q3",
    "sprint_pos", "sprint_classified", "sprint_points",
    "grid", "position", "position_text", "classified", "status", "points", "laps",
]


def lap_seconds(t) -> float:
    """'1:31.373' -> 91.373; missing or blank -> NaN."""
    if not isinstance(t, str) or not t.strip():
        return np.nan
    parts = t.strip().split(":")
    try:
        secs = float(parts[-1])
        if len(parts) >= 2:
            secs += 60 * float(parts[-2])
        if len(parts) == 3:
            secs += 3600 * float(parts[0])
        return secs
    except ValueError:
        return np.nan


def _ts(block: dict | None):
    if not block:
        return pd.NaT
    return pd.Timestamp(f"{block['date']}T{block.get('time', '00:00:00Z')}")


def calendar_rows(season: int, ttl_hours: float | None) -> list[dict]:
    rows = []
    for r in jolpica.schedule(season, ttl_hours):
        rows.append({
            "season": season,
            "round": int(r["round"]),
            "race_name": r["raceName"],
            "circuit_id": r["Circuit"]["circuitId"],
            "country": r["Circuit"]["Location"]["country"],
            "format": "sprint" if "Sprint" in r else "conventional",
            "fp1_utc": _ts(r.get("FirstPractice")),
            "sprint_quali_utc": _ts(r.get("SprintQualifying") or r.get("SprintShootout")),
            "sprint_utc": _ts(r.get("Sprint")),
            "quali_utc": _ts(r.get("Qualifying")),
            "race_utc": _ts(r),
        })
    return rows


def _by_driver(races: list[dict], key: str) -> dict[tuple[int, str], dict]:
    out = {}
    for race in races:
        for row in race.get(key, []):
            out[(int(race["round"]), row["Driver"]["driverId"])] = row
    return out


def season_results(season: int, ttl_hours: float | None = None) -> pd.DataFrame:
    """Driver-race rows for every race of `season` that has results."""
    races = jolpica.fetch_races(f"{season}/results", "Results", ttl_hours)
    quali = _by_driver(jolpica.fetch_races(f"{season}/qualifying", "QualifyingResults", ttl_hours),
                       "QualifyingResults")
    sprint = (_by_driver(jolpica.fetch_races(f"{season}/sprint", "SprintResults", ttl_hours),
                         "SprintResults") if season >= 2021 else {})
    cal = {r["round"]: r for r in calendar_rows(season, ttl_hours)}
    rows = []
    for race in races:
        rnd = int(race["round"])
        for res in race["Results"]:
            did = res["Driver"]["driverId"]
            q = quali.get((rnd, did), {})
            s = sprint.get((rnd, did))
            ptext = res["positionText"]
            rows.append({
                "season": season, "round": rnd, "race_name": race["raceName"],
                "date": race["date"], "circuit_id": race["Circuit"]["circuitId"],
                "format": cal.get(rnd, {}).get("format", "conventional"),
                "driver_id": did, "code": res["Driver"].get("code", did[:3].upper()),
                "constructor_id": res["Constructor"]["constructorId"],
                "number": res.get("number"),
                "quali_pos": int(q["position"]) if q.get("position") else np.nan,
                "q1": lap_seconds(q.get("Q1")), "q2": lap_seconds(q.get("Q2")), "q3": lap_seconds(q.get("Q3")),
                "sprint_pos": int(s["position"]) if s else np.nan,
                "sprint_classified": bool(s and s["positionText"].isdigit()) if s else np.nan,
                "sprint_points": float(s["points"]) if s else 0.0,
                "grid": int(res["grid"]), "position": int(res["position"]),
                "position_text": ptext, "classified": ptext.isdigit(),
                "status": res["status"], "points": float(res["points"]),
                "laps": int(res["laps"]),
            })
    return pd.DataFrame(rows, columns=RESULT_COLS)


CORRECTION_DAYS = 14
CORRECTION_COLS = ["driver_id", "constructor_id", "position", "position_text", "status", "points", "grid", "laps",
                   "sprint_pos", "sprint_points"]


def _corrected_rounds(season: int, old: pd.DataFrame, new: pd.DataFrame, exclude=(), today=None) -> list[int]:
    """Rounds raced within CORRECTION_DAYS whose official result differs from the store."""
    import json
    from .config import DATA
    today = pd.Timestamp(today or pd.Timestamp.utcnow().date())
    out = []
    for rnd, n in new.groupby("round"):
        if rnd in exclude or rnd not in set(old["round"]):
            continue
        if (today - pd.Timestamp(n["date"].iloc[0])).days > CORRECTION_DAYS:
            continue
        o = old[old["round"] == rnd]
        def key(d):
            d = d[CORRECTION_COLS].copy()
            for c in ["position", "points", "grid", "laps", "sprint_pos", "sprint_points"]:
                d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
            return d.astype(str).sort_values("driver_id").reset_index(drop=True)
        a, b = key(o), key(n.reindex(columns=old.columns))
        if a.shape != b.shape or not a.equals(b):
            diff = [{"driver_id": d, "field": c, "old": a.loc[i, c], "new": b.loc[i, c]}
                    for i, d in enumerate(a["driver_id"]) if i < len(b)
                    for c in CORRECTION_COLS if a.loc[i, c] != b.loc[i, c]]
            with open(DATA / "corrections.jsonl", "a") as f:
                f.write(json.dumps({"season": season, "round": int(rnd), "detected": str(pd.Timestamp.utcnow()),
                                    "changes": diff}) + "\n")
            out.append(int(rnd))
    return out


def update(seasons: list[int] | None = None, current: int | None = None) -> dict:
    """Refresh the store. Finished seasons are fetched once; `current` is re-polled
    with a short cache TTL. Returns {season: n_races} for seasons that changed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    current = current or pd.Timestamp.utcnow().year
    seasons = seasons or list(range(FIRST_SEASON, current + 1))
    changed = {}
    cal = []
    for season in seasons:
        live = season >= current
        ttl = 1.0 if live else None
        cal += calendar_rows(season, ttl)
        path = RESULTS_DIR / f"{season}.csv"
        if path.exists() and not live:
            continue
        new = season_results(season, ttl)
        if path.exists():
            old = pd.read_csv(path)
            # a stored round is final unless (a) the API now has more rows for it
            # (results published part-way through) or (b) it is inside the stewards'
            # correction window and the official result changed. Every correction is
            # logged to data/corrections.jsonl; older rounds are never rewritten.
            n_old, n_new = old.groupby("round").size(), new.groupby("round").size()
            grow = [r for r in n_new.index if n_new[r] > n_old.get(r, 0)]
            grow += _corrected_rounds(season, old, new, exclude=grow)
            if not grow:
                continue
            new = pd.concat([old[~old["round"].isin(grow)], new[new["round"].isin(grow)]], ignore_index=True)
            new = new.sort_values(["round", "position"], kind="stable")
        schemas.validate(new, "results")
        new.to_csv(path, index=False)
        changed[season] = new["round"].nunique()
    if cal:
        cdf = pd.DataFrame(cal)
        if CALENDAR_CSV.exists():
            old = pd.read_csv(CALENDAR_CSV)
            old = old[~old["season"].isin(cdf["season"])]
            cdf = pd.concat([old, cdf], ignore_index=True)
        cdf = cdf.sort_values(["season", "round"])
        schemas.validate(cdf, "calendar")
        cdf.to_csv(CALENDAR_CSV, index=False)
    return changed


def load_results(seasons=None) -> pd.DataFrame:
    files = sorted(RESULTS_DIR.glob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if seasons is not None:
        df = df[df["season"].isin(seasons)]
    return df.sort_values(["season", "round", "position"]).reset_index(drop=True)


def load_calendar() -> pd.DataFrame:
    cal = pd.read_csv(CALENDAR_CSV)
    for c in [c for c in cal.columns if c.endswith("_utc")]:
        cal[c] = pd.to_datetime(cal[c], utc=True)
    return cal


def weekend_entry(season: int, rnd: int, ttl_hours: float = 0.5) -> pd.DataFrame:
    """Pre-race view of an upcoming race: qualifying (+ sprint) rows, no race result.

    Raises DataUnavailable until qualifying (and the sprint, if any) is published.
    """
    cal = {r["round"]: r for r in calendar_rows(season, ttl_hours)}
    if rnd not in cal:
        raise jolpica.DataUnavailable(f"{season} round {rnd} not on calendar")
    ev = cal[rnd]
    q = jolpica.fetch_races(f"{season}/{rnd}/qualifying", "QualifyingResults", ttl_hours)
    if not q or not q[0].get("QualifyingResults"):
        raise jolpica.DataUnavailable(f"no qualifying yet for {season} R{rnd}")
    sprint = {}
    if ev["format"] == "sprint":
        pages = jolpica.fetch_races(f"{season}/{rnd}/sprint", "SprintResults", ttl_hours)
        if not pages or not pages[0].get("SprintResults"):
            raise jolpica.DataUnavailable(f"no sprint result yet for {season} R{rnd}")
        sprint = {r["Driver"]["driverId"]: r for r in pages[0]["SprintResults"]}
    rows = []
    for qr in q[0]["QualifyingResults"]:
        did = qr["Driver"]["driverId"]
        s = sprint.get(did)
        rows.append({
            "season": season, "round": rnd, "race_name": ev["race_name"],
            "date": ev["race_utc"].strftime("%Y-%m-%d"), "circuit_id": ev["circuit_id"],
            "format": ev["format"], "driver_id": did,
            "code": qr["Driver"].get("code", did[:3].upper()),
            "constructor_id": qr["Constructor"]["constructorId"], "number": qr.get("number"),
            "quali_pos": int(qr["position"]),
            "q1": lap_seconds(qr.get("Q1")), "q2": lap_seconds(qr.get("Q2")), "q3": lap_seconds(qr.get("Q3")),
            "sprint_pos": int(s["position"]) if s else np.nan,
            "sprint_classified": bool(s["positionText"].isdigit()) if s else np.nan,
            "sprint_points": float(s["points"]) if s else 0.0,
        })
    return pd.DataFrame(rows).reindex(columns=RESULT_COLS)
