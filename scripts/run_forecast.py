"""Trigger A: pre-race forecast. Idempotent; exits 0 when data is not ready yet.

Usage: python scripts/run_forecast.py 2026_19
"""
import json
import sys
from pathlib import Path


from f1 import ingest, predlog, registry, state
from f1.forecast import forecast
from f1.jolpica import DataUnavailable
from f1.config import DATA

# Challengers from scripts/model_lab.py, logged live next to the champion so they
# build a real (pre-race, timestamped) record. They are never used for promotion
# directly: promotion stays with the weekly gate.
CHALLENGERS = ["v1+practice", "nn+practice"]


def refresh_openf1(season: int, rnd: int):
    """Pull this weekend's practice/quali/sprint from OpenF1 and rebuild the
    analytics tables the practice-informed challengers read. Best effort: the
    champion does not depend on it."""
    import subprocess
    from f1 import openf1
    try:
        m = openf1.meetings(season, ingest.load_calendar(), ttl_hours=1)
        if rnd in m:
            print(openf1.fetch_weekend(season, rnd, m[rnd], ttl_hours=0.25))
        subprocess.run([sys.executable, str(Path(__file__).parent / "build_analytics.py")], check=True,
                       capture_output=True)
    except Exception as ex:  # OpenF1 live-session lock, network, partial data
        print(f"OpenF1 refresh skipped: {ex}")


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
    refresh_openf1(season, rnd)
    from model_lab import specs
    lab = specs(reg["versions"][reg["champion"]]["spec"])
    roles += [("challenger", f"lab:{c}") for c in CHALLENGERS]
    for role, version in roles:
        spec = lab[version[4:]] if version.startswith("lab:") else reg["versions"][version]["spec"]
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
        if role == "challenger":
            continue
        if role == "champion":
            out = {"season": season, "round": rnd, "race_name": ev.race_name, "race_utc": ev.race_utc.isoformat(),
                   "model_version": version, "logged_at": e["logged_at"] if e else None,
                   "drivers": pred.round(4).to_dict("records")}
            (DATA / "dashboard").mkdir(parents=True, exist_ok=True)
            (DATA / "dashboard" / "next_race.json").write_text(json.dumps(out, indent=1, default=float))
    state.mark("forecast", season, rnd, "done")


if __name__ == "__main__":
    main(sys.argv[1])
