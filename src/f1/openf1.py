"""OpenF1 (api.openf1.org): free, no login, 2023 onward.

Per race weekend it gives lap times with sectors and speed traps for every
session (practice included), tyre stints with the real compound, weather,
race control messages (SC/VSC/red flags) and ~3.7 Hz car telemetry.

While any F1 session is live the API answers 401 to unauthenticated users;
that is raised as LiveSession (a DataUnavailable) so callers retry later.

Store, one folder per weekend: data/openf1/<season>_<rr>/
  sessions.csv      session code/key/times + weather and race-control summary
  drivers.csv       driver_number, acronym, team_name, team_colour per session
  laps.csv          every timed lap of every session, with compound and tyre age
  race_control.csv  race-control messages
  tel_Q.csv         telemetry of each driver's fastest qualifying lap (distance-indexed)
  map_Q.csv         x/y track outline from the pole lap
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd
import requests

from .config import CACHE, DATA
from .jolpica import DataUnavailable

BASE = "https://api.openf1.org/v1"
OPENF1_DIR = DATA / "openf1"
FIRST_SEASON = 2023
_MIN_INTERVAL = 0.4
_last_call = 0.0

SESSION_CODES = {"Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3",
                 "Sprint Qualifying": "SQ", "Sprint Shootout": "SQ", "Sprint": "S",
                 "Qualifying": "Q", "Race": "R"}
SESSION_ORDER = ["FP1", "FP2", "FP3", "SQ", "S", "Q", "R"]


class LiveSession(DataUnavailable):
    """OpenF1 is restricted while a session is live."""


def _get(path: str, ttl_hours: float | None = None, **params):
    global _last_call
    q = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE}/{path}?{q}"
    p = CACHE / "openf1" / (hashlib.sha1(url.encode()).hexdigest() + ".json")
    if p.exists() and (ttl_hours is None or time.time() - p.stat().st_mtime < ttl_hours * 3600):
        return json.loads(p.read_text())
    for attempt in range(6):
        wait = _MIN_INTERVAL - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        try:
            r = requests.get(url, timeout=90)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 401:
            raise LiveSession(r.text[:200])
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 404:
            data = []
        else:
            r.raise_for_status()
            data = r.json()
        if isinstance(data, dict):  # {"detail": "No results found."}
            data = []
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data))
        return data
    raise RuntimeError(f"OpenF1 failed repeatedly: {url}")


def meeting_dir(season: int, rnd: int):
    return OPENF1_DIR / f"{season}_{rnd:02d}"


def meetings(season: int, calendar: pd.DataFrame, ttl_hours: float | None = 6.0) -> dict[int, int]:
    """Map Jolpica round -> OpenF1 meeting_key by matching the race date (±1 day)."""
    sess = pd.DataFrame(_get("sessions", ttl_hours, year=season))
    if sess.empty:
        return {}
    races = sess[(sess["session_name"] == "Race") & ~sess.get("is_cancelled", False).astype(bool)].copy()
    races["day"] = pd.to_datetime(races["date_start"], utc=True).dt.normalize()
    cal = calendar[calendar["season"] == season]
    out = {}
    for rnd, ru in zip(cal["round"], cal["race_utc"]):
        d = abs(races["day"] - pd.Timestamp(ru).normalize())
        if len(d) and d.min() <= pd.Timedelta(days=1):
            out[int(rnd)] = int(races.loc[d.idxmin(), "meeting_key"])
    return out


def _frame(rows, cols=None):
    df = pd.DataFrame(rows)
    return df.reindex(columns=cols) if cols is not None else df


def _session_summary(sess, weather, rc):
    out = []
    for _, s in sess.iterrows():
        w = weather[weather["session_key"] == s.session_key] if len(weather) else weather
        m = rc[rc["session_key"] == s.session_key] if len(rc) else rc
        msg = m["message"].fillna("").str.upper() if len(m) else pd.Series(dtype=str)
        out.append({
            "session": s.code, "session_key": s.session_key, "session_name": s.session_name,
            "start_utc": s.date_start, "end_utc": s.date_end,
            "air_temp": w["air_temperature"].mean() if len(w) else np.nan,
            "track_temp": w["track_temperature"].mean() if len(w) else np.nan,
            "humidity": w["humidity"].mean() if len(w) else np.nan,
            "rain_share": (w["rainfall"] > 0).mean() if len(w) else np.nan,
            "wind_speed": w["wind_speed"].mean() if len(w) else np.nan,
            "sc": int(msg.str.contains("SAFETY CAR DEPLOYED").sum() - msg.str.contains("VIRTUAL SAFETY CAR DEPLOYED").sum()),
            "vsc": int(msg.str.contains("VIRTUAL SAFETY CAR DEPLOYED").sum()),
            "red_flags": int((m["flag"] == "RED").sum()) if len(m) else 0,
        })
    return pd.DataFrame(out)


def _laps_with_stints(laps, stints, sess):
    l = laps.merge(sess[["session_key", "code"]], on="session_key").rename(columns={"code": "session"})
    l = l.rename(columns={"lap_number": "lap", "lap_duration": "lap_s", "duration_sector_1": "s1",
                          "duration_sector_2": "s2", "duration_sector_3": "s3", "st_speed": "st",
                          "i1_speed": "i1", "i2_speed": "i2", "is_pit_out_lap": "pit_out"})
    l["compound"], l["stint"], l["tyre_age"] = None, np.nan, np.nan
    if len(stints):
        for st in stints.itertuples():
            m = ((l["session_key"] == st.session_key) & (l["driver_number"] == st.driver_number)
                 & (l["lap"] >= st.lap_start) & (l["lap"] <= (st.lap_end if pd.notna(st.lap_end) else 999)))
            l.loc[m, "compound"] = st.compound
            l.loc[m, "stint"] = st.stint_number
            age0 = st.tyre_age_at_start if pd.notna(st.tyre_age_at_start) else 0
            l.loc[m, "tyre_age"] = age0 + l.loc[m, "lap"] - st.lap_start
    cols = ["session", "driver_number", "lap", "lap_s", "s1", "s2", "s3", "i1", "i2", "st", "pit_out",
            "date_start", "compound", "stint", "tyre_age"]
    order = {c: i for i, c in enumerate(SESSION_ORDER)}
    return (l[cols].assign(_o=l["session"].map(order)).sort_values(["_o", "driver_number", "lap"])
            .drop(columns="_o").reset_index(drop=True))


def _distance(car: pd.DataFrame, lap_start: pd.Timestamp) -> pd.DataFrame:
    """t = seconds since the lap started (official timing line crossing);
    dist = integrated speed from the line (trapezoid; the first sample's speed
    covers the gap back to the line)."""
    car = car.sort_values("date").copy()
    t = pd.to_datetime(car["date"], utc=True, format="ISO8601")
    car["t"] = (t - lap_start).dt.total_seconds()
    v = car["speed"].to_numpy(float) / 3.6
    tt = car["t"].to_numpy()
    seg = np.r_[v[0] * max(tt[0], 0.0), 0.5 * (v[1:] + v[:-1]) * np.diff(tt)]
    car["dist"] = np.cumsum(seg)
    return car


def fetch_quali_telemetry(laps: pd.DataFrame, sess: pd.DataFrame, ttl_hours=None):
    """Fastest qualifying lap of every driver -> distance-indexed telemetry, plus
    the pole lap's x/y outline for track maps."""
    q = sess[sess["code"] == "Q"]
    if q.empty:
        return None, None
    key = int(q["session_key"].iloc[0])
    ql = laps[(laps["session"] == "Q") & laps["lap_s"].notna()]
    if ql.empty:
        return None, None
    best = ql.loc[ql.groupby("driver_number")["lap_s"].idxmin()].sort_values("lap_s")
    tels = []
    for b in best.itertuples():
        t0 = pd.Timestamp(b.date_start)
        t1 = t0 + pd.Timedelta(seconds=float(b.lap_s))
        car = _frame(_get("car_data", ttl_hours, session_key=key, driver_number=int(b.driver_number),
                          **{"date>": t0.isoformat(), "date<": t1.isoformat()}))
        if len(car) < 50:
            continue
        car = _distance(car, t0)
        car["driver_number"], car["lap_s"] = int(b.driver_number), float(b.lap_s)
        tels.append(car[["driver_number", "lap_s", "t", "dist", "speed", "throttle", "brake", "n_gear", "drs", "rpm"]])
    p = best.iloc[0]
    t0 = pd.Timestamp(p.date_start)
    loc = _frame(_get("location", ttl_hours, session_key=key, driver_number=int(p.driver_number),
                      **{"date>": t0.isoformat(), "date<": (t0 + pd.Timedelta(seconds=float(p.lap_s))).isoformat()}))
    tmap = None
    if len(loc):
        loc = loc.sort_values("date")
        tt = pd.to_datetime(loc["date"], utc=True, format="ISO8601")
        loc["t"] = (tt - tt.iloc[0]).dt.total_seconds()
        tmap = loc[["t", "x", "y"]].assign(driver_number=int(p.driver_number))
    tel = pd.concat(tels, ignore_index=True) if tels else None
    if tel is not None:
        tel[["t", "dist"]] = tel[["t", "dist"]].round(2)
    return tel, tmap


def fetch_weekend(season: int, rnd: int, meeting_key: int, ttl_hours: float | None = None,
                  telemetry: bool = True) -> dict:
    """Download one weekend into data/openf1/<season>_<rr>/. Returns sessions found."""
    sess = _frame(_get("sessions", ttl_hours, meeting_key=meeting_key))
    if sess.empty:
        raise DataUnavailable(f"no OpenF1 sessions for meeting {meeting_key}")
    sess["code"] = sess["session_name"].map(SESSION_CODES)
    sess = sess[sess["code"].notna()]
    laps = _frame(_get("laps", ttl_hours, meeting_key=meeting_key))
    if laps.empty:
        raise DataUnavailable(f"no OpenF1 laps yet for {season} R{rnd}")
    stints = _frame(_get("stints", ttl_hours, meeting_key=meeting_key))
    weather = _frame(_get("weather", ttl_hours, meeting_key=meeting_key))
    rc = _frame(_get("race_control", ttl_hours, meeting_key=meeting_key))
    drivers = _frame(_get("drivers", ttl_hours, meeting_key=meeting_key))
    out = meeting_dir(season, rnd)
    out.mkdir(parents=True, exist_ok=True)
    laps = _laps_with_stints(laps, stints, sess)
    laps.to_csv(out / "laps.csv", index=False)
    _session_summary(sess, weather, rc).to_csv(out / "sessions.csv", index=False)
    drv = drivers.merge(sess[["session_key", "code"]], on="session_key").rename(columns={"code": "session"})
    drv[["session", "driver_number", "name_acronym", "full_name", "team_name", "team_colour"]].to_csv(
        out / "drivers.csv", index=False)
    if len(rc):
        rc.merge(sess[["session_key", "code"]], on="session_key").rename(columns={"code": "session"})[
            ["session", "date", "lap_number", "category", "flag", "scope", "driver_number", "message"]].to_csv(
            out / "race_control.csv", index=False)
    if telemetry:
        tel, tmap = fetch_quali_telemetry(laps, sess, ttl_hours)
        if tel is not None:
            tel.to_csv(out / "tel_Q.csv", index=False)
        if tmap is not None:
            tmap.to_csv(out / "map_Q.csv", index=False)
    return {"sessions": sorted(set(laps["session"]), key=SESSION_ORDER.index), "laps": len(laps)}


def stored(season: int, rnd: int) -> set[str]:
    p = meeting_dir(season, rnd) / "sessions.csv"
    if not p.exists():
        return set()
    l = meeting_dir(season, rnd) / "laps.csv"
    return set(pd.read_csv(l, usecols=["session"])["session"]) if l.exists() else set()


def rebuild_telemetry(season: int, rnd: int) -> bool:
    """Re-derive tel_Q/map_Q from the raw API cache (no new requests once cached)."""
    d = meeting_dir(season, rnd)
    if not (d / "laps.csv").exists():
        return False
    sess = pd.read_csv(d / "sessions.csv").rename(columns={"session": "code"})
    laps = pd.read_csv(d / "laps.csv")
    tel, tmap = fetch_quali_telemetry(laps, sess)
    if tel is not None:
        tel.to_csv(d / "tel_Q.csv", index=False)
    if tmap is not None:
        tmap.to_csv(d / "map_Q.csv", index=False)
    return tel is not None
