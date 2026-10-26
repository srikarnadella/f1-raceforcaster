"""Turn every stored OpenF1 weekend into tidy analytics tables (data/analytics/).

All tables carry season, round, circuit_id. They are derived (rebuildable)
from data/openf1/ and data/results/, so the dashboard never calls an API.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .. import ingest, openf1
from ..config import DATA
from . import telemetry as T
from . import weekend as W

OUT = DATA / "analytics"


def weekends() -> list[tuple[int, int]]:
    out = []
    for d in sorted(openf1.OPENF1_DIR.glob("*_*")):
        if (d / "laps.csv").exists():
            s, r = d.name.split("_")
            out.append((int(s), int(r)))
    return out


def build_weekend(season: int, rnd: int, cal: pd.DataFrame) -> dict[str, pd.DataFrame]:
    w = W.load(season, rnd)
    laps, ids, sess = w["laps"], w["drivers"], w["sessions"]
    c = cal[(cal["season"] == season) & (cal["round"] == rnd)]
    meta = {"season": season, "round": rnd,
            "circuit_id": c["circuit_id"].iat[0] if len(c) else None,
            "race_name": c["race_name"].iat[0] if len(c) else None,
            "format": c["format"].iat[0] if len(c) else None}
    out = {}
    ol = W.one_lap(laps)
    out["one_lap"] = ol
    lr = []
    for s in [s for s in ("FP1", "FP2", "FP3") if s in set(laps["session"])]:
        t = W.long_runs(laps, s)
        if len(t):
            lr.append(t.assign(session=s))
    out["long_runs"] = pd.concat(lr, ignore_index=True) if lr else pd.DataFrame()
    rp, fuel, race_deg, race_off = [], -0.06, {}, {}
    for s in [s for s in W.RACE_SESSIONS if s in set(laps["session"])]:
        t = W.race_pace(laps, s)
        if len(t):
            rp.append(t.assign(session=s))
            if s == "R":
                fuel = t.attrs.get("fuel_s_per_lap", fuel)
                race_deg = t.attrs.get("deg_s_per_lap", {})
                race_off = t.attrs.get("compound_offset_s", {})
    out["race_pace"] = pd.concat(rp, ignore_index=True) if rp else pd.DataFrame()
    out["degradation"] = W.degradation(laps, fuel)
    has_race = "R" in set(laps["session"])
    out["stints"] = W.stints(laps, "R") if has_race else pd.DataFrame()
    out["race_story"] = W.race_story(laps, "R") if has_race else pd.DataFrame()
    ov = W.overtakes(out["race_story"], laps) if has_race and len(out["race_story"]) else {}
    drivers_seen = set(ov.get("by_driver", {})) | set(ov.get("conceded", {}))
    out["overtakes"] = pd.DataFrame([{"driver_id": d, "overtakes": ov["by_driver"].get(d, 0),
                                      "conceded": ov["conceded"].get(d, 0)} for d in drivers_seen])
    out["driver_traits"] = W.driver_race_traits(laps, fuel) if has_race else pd.DataFrame()
    from .ml import run_programs
    out["run_programmes"] = run_programs(laps)
    cat = laps.groupby(["session", "category"]).size().rename("laps").reset_index()
    out["lap_classes"] = cat
    tel = T.load(season, rnd)
    prof = {}
    if tel is not None and tel["driver_number"].nunique() >= 5:
        try:
            zt, cls, z = T.zone_table(tel, ids)
            out["zones"] = cls
            out["corners"] = zt[zt["kind"] != "straight"][["code", "driver_id", "constructor_id", "team_name",
                                                            "team_colour", "corner", "kind", "time_s", "min_speed",
                                                            "delta_to_pole_s", "apex_delta_kmh"]]
            prof = T.circuit_profile(tel)
        except Exception as e:  # malformed telemetry for one weekend should not stop the build
            warnings.warn(f"telemetry {season} R{rnd}: {e}")
    r = sess[sess["session"] == "R"]
    q = sess[sess["session"] == "Q"]
    prof.update({
        "race_track_temp": r["track_temp"].iat[0] if len(r) else np.nan,
        "quali_track_temp": q["track_temp"].iat[0] if len(q) else np.nan,
        "race_rain_share": r["rain_share"].iat[0] if len(r) else np.nan,
        "quali_rain_share": q["rain_share"].iat[0] if len(q) else np.nan,
        "race_sc": r["sc"].iat[0] if len(r) else np.nan, "race_vsc": r["vsc"].iat[0] if len(r) else np.nan,
        "race_red": r["red_flags"].iat[0] if len(r) else np.nan,
        "race_deg_soft": race_deg.get("SOFT", np.nan), "race_deg_medium": race_deg.get("MEDIUM", np.nan),
        "race_deg_hard": race_deg.get("HARD", np.nan),
        "race_off_medium": race_off.get("MEDIUM", np.nan), "race_off_hard": race_off.get("HARD", np.nan),
        "pit_loss_s": W.pit_loss(laps) if has_race else np.nan,
        "overtakes": ov.get("overtakes", np.nan), "race_laps": ov.get("race_laps", np.nan),
        "overtakes_per_lap": ov.get("overtakes_per_lap", np.nan),
        "race_stops_median": (out["stints"].groupby("driver_id").size().median() - 1) if len(out["stints"]) else np.nan,
    })
    out["weekend"] = pd.DataFrame([meta | prof])
    for k, v in out.items():
        if len(v) and k != "weekend":
            for col, val in reversed(list(meta.items())):
                if col not in v.columns:
                    v.insert(0, col, val)
                elif col == "circuit_id":
                    v[col] = val
    return out


def build_all(verbose: bool = False) -> dict[str, pd.DataFrame]:
    cal = ingest.load_calendar()
    tables: dict[str, list] = {}
    for season, rnd in weekends():
        try:
            t = build_weekend(season, rnd, cal)
        except Exception as e:
            warnings.warn(f"weekend {season} R{rnd} skipped: {e}")
            continue
        for k, v in t.items():
            if len(v):
                tables.setdefault(k, []).append(v)
        if verbose:
            print(season, rnd, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for k, parts in tables.items():
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(OUT / f"{k}.csv", index=False, float_format="%.5g")
        res[k] = df
    return res


def read(name: str) -> pd.DataFrame:
    p = OUT / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()
