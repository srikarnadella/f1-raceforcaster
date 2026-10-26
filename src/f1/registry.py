"""Versioned model registry with champion pointer, shadow mode and rollback.

models/registry.json holds every version's spec (models are small and refit
deterministically from spec + data, so the spec plus git commit plus data hash
is the artifact). Versions are never deleted.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from datetime import datetime, UTC

from .config import RESULTS_DIR, REGISTRY, ROOT
from .models import DEFAULT_SPEC

SHADOW_RACES = 6
ROLLBACK_MARGIN = 0.02
# owner decisions (2026-10-26): no automatic rollback on fewer than 3 live races, and the shadow is
# always a version with a live (pre-race, scored) record
ROLLBACK_MIN_RACES = 3


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "uncommitted"


def data_hash(analytics: bool = False) -> str:
    """Fingerprint of the data a model was trained on: the result store, plus (when the
    model reads them) the analytics tables the race simulator and practice features use."""
    h = hashlib.sha256()
    files = sorted(RESULTS_DIR.glob("*.csv"))
    if analytics:
        from .analytics.build import OUT
        files += [OUT / f for f in ("profiles.csv", "weekend.csv", "overtakes_by_race.csv", "prerace_features.csv",
                                    "one_lap.csv", "race_pace.csv") if (OUT / f).exists()]
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def load() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {"champion": None, "shadow": None, "shadow_since": None, "versions": {}, "history": []}


def save(reg: dict):
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=1, default=float) + "\n")


def ensure_initialized(backtest: dict | None = None) -> dict:
    reg = load()
    if not reg["versions"]:
        register(reg, "v1", DEFAULT_SPEC, note="initial champion", backtest=backtest)
        reg["champion"] = "v1"
        reg["history"].append({"date": now(), "event": "init", "version": "v1"})
        save(reg)
    return reg


def register(reg: dict, version: str, spec: dict, note: str = "", parent: str | None = None,
             backtest: dict | None = None):
    if version in reg["versions"]:
        raise ValueError(f"version {version} exists; versions are immutable")
    reg["versions"][version] = {"spec": copy.deepcopy(spec), "created": now(), "git_commit": git_commit(),
                                "data_hash": data_hash(), "parent": parent, "note": note,
                                "backtest": backtest}


def next_version(reg: dict) -> str:
    return f"v{max(int(v[1:]) for v in reg['versions']) + 1}"


def champion_spec(reg: dict | None = None) -> dict:
    reg = reg or ensure_initialized()
    return reg["versions"][reg["champion"]]["spec"]


def live_races(version: str) -> int:
    """Pre-race (not late) forecasts of `version` that have been scored."""
    from . import predlog
    return len({(s["season"], s["round"]) for s in predlog.read_scores()
                if s["model_version"] == version and not s["late"]})


def promote(reg: dict, version: str, season_round: tuple[int, int], reason: str):
    old = reg["champion"]
    reg["champion"] = version
    # the fallback must have a live record: if the outgoing champion was never scored live (promoted
    # and replaced between races), keep the existing shadow instead
    keep = reg.get("shadow") if reg.get("shadow") and live_races(old) == 0 else None
    reg["shadow"] = keep or old
    reg["shadow_since"] = list(season_round)   # first live race the new champion forecasts
    reg["history"].append({"date": now(), "event": "promote", "version": version, "previous": old,
                           "shadow": reg["shadow"], "reason": reason})


def rollback(reg: dict, reason: str):
    bad = reg["champion"]
    reg["champion"] = reg["shadow"]
    reg["shadow"] = None
    reg["shadow_since"] = None
    reg["history"].append({"date": now(), "event": "rollback", "version": reg["champion"], "from": bad,
                           "reason": reason})
    return bad
