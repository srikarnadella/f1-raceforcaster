"""Append-only prediction log and score log.

predictions/log.jsonl is the source of truth for live performance. Entries are
only ever appended; nothing here edits or deletes a line. Writes are idempotent:
an entry with the same (season, round, role, model_version) is not written twice.
A forecast logged after the race start is refused unless explicitly marked late,
and late entries never count as live accuracy.
"""
from __future__ import annotations

import json
from datetime import datetime, UTC

import pandas as pd

from .config import PRED_LOG, SCORES_LOG

DRIVER_COLS = ["driver_id", "code", "constructor_id", "start_pos", "p_win", "p_top3", "p_top10",
               "exp_pos", "pred_rank", "p_dnf", "score"]


class LateForecast(Exception):
    pass


def _read(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _append(path, entry: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=float, sort_keys=True) + "\n")


def read_log() -> list[dict]:
    return _read(PRED_LOG)


def read_scores() -> list[dict]:
    return _read(SCORES_LOG)


def _key(e):
    return (e["season"], e["round"], e["role"], e["model_version"])


def find(season, rnd, role=None) -> list[dict]:
    return [e for e in read_log() if e["season"] == season and e["round"] == rnd
            and (role is None or e["role"] == role)]


def log_forecast(pred: pd.DataFrame, *, season: int, rnd: int, race_name: str, race_utc: pd.Timestamp,
                 fmt: str, role: str, model_version: str, git_commit: str, data_hash: str,
                 feature_hash: str, allow_late: bool = False, now: datetime | None = None) -> dict | None:
    """Append a forecast. Returns the entry, or None if already logged."""
    now = now or datetime.now(UTC)
    late = now >= race_utc.to_pydatetime()
    if late and not allow_late:
        raise LateForecast(f"{season} R{rnd}: race started at {race_utc}, now {now}")
    entry = {
        "logged_at": now.isoformat(timespec="seconds"), "season": int(season), "round": int(rnd),
        "race_name": race_name, "race_utc": race_utc.isoformat(), "format": fmt, "role": role,
        "model_version": model_version, "git_commit": git_commit, "data_hash": data_hash,
        "feature_hash": feature_hash, "late": bool(late),
        "drivers": pred[[c for c in DRIVER_COLS if c in pred]].round(6).to_dict("records"),
    }
    if any(_key(e) == _key(entry) for e in read_log()):
        return None
    _append(PRED_LOG, entry)
    return entry


def log_score(entry: dict, metrics: dict, baselines: dict) -> dict | None:
    rec = {"season": entry["season"], "round": entry["round"], "role": entry["role"],
           "model_version": entry["model_version"], "late": entry["late"], "logged_at": entry["logged_at"],
           "scored_at": datetime.now(UTC).isoformat(timespec="seconds"),
           "metrics": metrics, "baselines": baselines}
    if any(_key(s) == _key(rec) for s in read_scores()):
        return None
    _append(SCORES_LOG, rec)
    return rec


def entry_frame(entry: dict) -> pd.DataFrame:
    return pd.DataFrame(entry["drivers"])
