"""Minimal cached, rate-limited client for the Jolpica (Ergast-compatible) F1 API.

Free and unauthenticated. Limits: ~4 req/s burst, 500 req/hour, so every page
is cached on disk. Pages for finished seasons never change; pages for the
current season expire after `ttl_hours`.
"""
from __future__ import annotations

import hashlib
import json
import time

import requests

from .config import CACHE

BASE = "https://api.jolpi.ca/ergast/f1"
PAGE = 100
_MIN_INTERVAL = 0.35
_last_call = 0.0


class DataUnavailable(Exception):
    """Raised when the API has no data (yet) for a request."""


def _cache_path(url: str):
    d = CACHE / "jolpica"
    d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.sha1(url.encode()).hexdigest() + ".json")


def _get(url: str, ttl_hours: float | None) -> dict:
    global _last_call
    p = _cache_path(url)
    if p.exists() and (ttl_hours is None or time.time() - p.stat().st_mtime < ttl_hours * 3600):
        return json.loads(p.read_text())
    for attempt in range(5):
        wait = _MIN_INTERVAL - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        r = requests.get(url, timeout=30)
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        data = r.json()["MRData"]
        p.write_text(json.dumps(data))
        return data
    raise RuntimeError(f"rate limited repeatedly: {url}")


def fetch_races(path: str, table_key: str, ttl_hours: float | None = None) -> list[dict]:
    """Fetch every page of `path` (e.g. '2024/results') and merge races.

    Ergast paginates by result row, so a race can be split across pages;
    rows of the same round are merged back together.
    """
    races: dict[str, dict] = {}
    offset = 0
    while True:
        data = _get(f"{BASE}/{path}.json?limit={PAGE}&offset={offset}", ttl_hours)
        for race in data["RaceTable"]["Races"]:
            key = race["round"]
            if key in races:
                races[key][table_key].extend(race.get(table_key, []))
            else:
                races[key] = race
        offset += PAGE
        if offset >= int(data["total"]):
            break
    return [races[k] for k in sorted(races, key=int)]


def schedule(season: int, ttl_hours: float | None = None) -> list[dict]:
    data = _get(f"{BASE}/{season}.json?limit=100", ttl_hours)
    return data["RaceTable"]["Races"]
