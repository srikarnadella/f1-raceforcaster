"""PROTECTED (CLAUDE.md section 9): frozen walk-forward evaluation harness.

For every test race, in date order, the model is fit on all strictly earlier
races and predicts the next one. No random k-fold, ever. The folds below
(test window, minimum training history) are part of the frozen protocol.
"""
from __future__ import annotations

import time

import pandas as pd

from eval.metrics import pooled_calibration, score_race, summarize

TEST_FROM_SEASON = 2019     # frozen: test window starts here
MIN_TRAIN_SEASONS = 3       # frozen: models always see >= 3 full seasons


def walk_forward(model_factory, feats: pd.DataFrame, test_from: int = TEST_FROM_SEASON,
                 test_to: int | None = None, verbose: bool = False):
    """Returns (per_race_metrics, stacked_predictions)."""
    feats = feats[feats["position"].notna()]
    keys = feats[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    keys = keys[keys["season"] >= test_from]
    if test_to is not None:
        keys = keys[keys["season"] <= test_to]
    first = int(feats["season"].min())
    assert test_from - first >= MIN_TRAIN_SEASONS, "not enough training history"
    rows, preds = [], []
    t0 = time.time()
    for season, rnd in keys.itertuples(index=False):
        before = (feats["season"] < season) | ((feats["season"] == season) & (feats["round"] < rnd))
        train = feats[before]
        race = feats[(feats["season"] == season) & (feats["round"] == rnd)]
        test_view = race.drop(columns=["position", "position_text", "classified", "status", "points", "laps", "grid"])
        model = model_factory().fit(train)
        pred = model.predict(test_view)
        m = score_race(pred, race)
        m.update(season=int(season), round=int(rnd), n=len(race))
        rows.append(m)
        preds.append(pred.merge(race[["driver_id", "position"]], on="driver_id"))
        if verbose and len(rows) % 25 == 0:
            print(f"  {len(rows)} races, {time.time() - t0:.0f}s")
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def evaluate(model_factory, feats, **kw) -> dict:
    per_race, preds = walk_forward(model_factory, feats, **kw)
    return {"summary": summarize(per_race), "calibration": pooled_calibration(preds),
            "per_race": per_race, "preds": preds}
