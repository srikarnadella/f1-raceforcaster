"""Trigger A: pre-race forecast. Idempotent; exits 0 when data is not ready yet.

Usage: python scripts/run_forecast.py 2026_19
"""
import json
import sys

import _path  # noqa: F401

from f1 import ingest, predlog, registry, state
from f1.forecast import forecast
from f1.jolpica import DataUnavailable
from f1.config import DATA


def main(key: str):
    season, rnd = map(int, key.split("_"))
    ev = ingest.load_calendar().query("season == @season and round == @rnd").iloc[0]
    reg = registry.ensure_initialized()
    roles = [("champion", reg["champion"])] + ([("shadow", reg["shadow"])] if reg.get("shadow") else [])
    if all(any(e["model_version"] == v for e in predlog.find(season, rnd, role)) for role, v in roles):
        print(f"{key}: already forecast")
        state.mark("forecast", season, rnd, "done")
        return
    ingest.update(seasons=[season], current=season)
    results = ingest.load_results()
    try:
        entry = ingest.weekend_entry(season, rnd)
    except DataUnavailable as ex:
        print(f"{key}: not ready ({ex}); will retry next tick")
        return
    for role, version in roles:
        spec = reg["versions"][version]["spec"]
        pred, fh = forecast(season, rnd, spec, entry=entry, results=results)
        try:
            e = predlog.log_forecast(pred, season=season, rnd=rnd, race_name=ev.race_name, race_utc=ev.race_utc,
                                     fmt=ev.format, role=role, model_version=version,
                                     git_commit=registry.git_commit(), data_hash=registry.data_hash(),
                                     feature_hash=fh)
        except predlog.LateForecast as ex:
            print(f"{key}: {ex}; not logged")
            state.mark("forecast", season, rnd, "missed (late)")
            return
        print(f"{key}: logged {role} {version}" if e else f"{key}: {role} {version} already logged")
        if role == "champion":
            out = {"season": season, "round": rnd, "race_name": ev.race_name, "race_utc": ev.race_utc.isoformat(),
                   "model_version": version, "logged_at": e["logged_at"] if e else None,
                   "drivers": pred.round(4).to_dict("records")}
            (DATA / "dashboard").mkdir(parents=True, exist_ok=True)
            (DATA / "dashboard" / "next_race.json").write_text(json.dumps(out, indent=1, default=float))
    state.mark("forecast", season, rnd, "done")


if __name__ == "__main__":
    main(sys.argv[1])
