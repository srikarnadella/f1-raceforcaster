"""Idempotency state (state/last_run.json) and the trigger schedule logic."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, UTC

import pandas as pd

from .config import LAST_RUN

FORECAST_DELAY = timedelta(hours=1)    # after the last pre-race session starts + its length
QUALI_LENGTH = timedelta(hours=1)
REVIEW_DELAY = timedelta(hours=3)      # race length + results publication
GIVE_UP_AFTER = timedelta(days=2, hours=12)  # ~Tuesday after a Sunday race


def key(season, rnd) -> str:
    return f"{int(season)}_{int(rnd):02d}"


def load() -> dict:
    if LAST_RUN.exists():
        return json.loads(LAST_RUN.read_text())
    return {"forecast": {}, "review": {}, "notes": {}}


def save(st: dict):
    LAST_RUN.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN.write_text(json.dumps(st, indent=1, sort_keys=True) + "\n")


def mark(job: str, season, rnd, info):
    st = load()
    st.setdefault(job, {})[key(season, rnd)] = info
    save(st)


def last_pre_race_session(ev) -> pd.Timestamp:
    times = [t for t in (ev["quali_utc"], ev.get("sprint_utc")) if pd.notna(t) and t < ev["race_utc"]]
    return max(times) if times else ev["race_utc"] - timedelta(days=1)


def due_jobs(calendar: pd.DataFrame, now: datetime | None = None, st: dict | None = None) -> dict:
    """Which job (if any) is due now. Returns {"forecast": (s, r) | None, "review": ..., "overdue": [...]}"""
    now = pd.Timestamp(now or datetime.now(UTC))
    st = st or load()
    out: dict = {"forecast": None, "review": None, "overdue": []}
    for _, ev in calendar.sort_values(["season", "round"]).iterrows():
        k = key(ev["season"], ev["round"])
        ready = last_pre_race_session(ev) + QUALI_LENGTH + FORECAST_DELAY
        if out["forecast"] is None and ready <= now < ev["race_utc"] and k not in st.get("forecast", {}):
            out["forecast"] = (int(ev["season"]), int(ev["round"]))
        rready = ev["race_utc"] + REVIEW_DELAY
        if rready <= now < ev["race_utc"] + timedelta(days=10) and k not in st.get("review", {}):
            if out["review"] is None:
                out["review"] = (int(ev["season"]), int(ev["round"]))
            if now > ev["race_utc"] + GIVE_UP_AFTER:
                out["overdue"].append(k)
    return out
