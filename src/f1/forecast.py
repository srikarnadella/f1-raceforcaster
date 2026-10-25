"""Pre-race forecast for one race, using only data available before it."""
from __future__ import annotations

import pandas as pd

from . import ingest
from .features import build_features, feature_hash
from .models import RaceModel


def history_before(results: pd.DataFrame, season: int, rnd: int) -> pd.DataFrame:
    before = (results["season"] < season) | ((results["season"] == season) & (results["round"] < rnd))
    return results[before]


def forecast(season: int, rnd: int, spec: dict, entry: pd.DataFrame | None = None,
             results: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str]:
    """Returns (per-driver forecast, feature hash). `entry` defaults to the live
    qualifying/sprint data for the weekend."""
    results = ingest.load_results() if results is None else results
    history = history_before(results, season, rnd)
    if entry is None:
        entry = ingest.weekend_entry(season, rnd)
    feats = build_features(history, entry, **spec.get("form", {}))
    race = feats[(feats["season"] == season) & (feats["round"] == rnd)]
    train = feats[feats["position"].notna()]
    model = RaceModel(spec).fit(train)
    pred = model.predict(race)
    cols = spec["strength"]["features"] + spec["dnf"]["features"]
    return pred, feature_hash(race, cols)
