"""Leakage tests. The promotion gate requires these to pass."""
import numpy as np
import pandas as pd
import pytest

from eval import harness
from f1.features import (DNF_FEATURES, FORM_ONLY_FEATURES, OUTCOME_COLS, STRENGTH_FEATURES,
                         build_features)
from f1.forecast import history_before
from f1.models import DEFAULT_SPEC

ALL_FEATS = sorted(set(STRENGTH_FEATURES + DNF_FEATURES + FORM_ONLY_FEATURES))
TARGETS = [(2024, 10), (2025, 6), (2026, 3)]


def _race(df, s, r):
    return df[(df.season == s) & (df["round"] == r)].sort_values("driver_id").reset_index(drop=True)


@pytest.mark.parametrize("season,rnd", TARGETS)
def test_features_identical_with_future_removed(results, season, rnd):
    full = build_features(results)
    race = _race(results, season, rnd)
    trunc = build_features(history_before(results, season, rnd), race.drop(columns=OUTCOME_COLS))
    a, b = _race(full, season, rnd), _race(trunc, season, rnd)
    pd.testing.assert_frame_equal(a[ALL_FEATS], b[ALL_FEATS])


@pytest.mark.parametrize("season,rnd", TARGETS)
def test_own_outcome_does_not_move_features(results, season, rnd):
    base = _race(build_features(results), season, rnd)
    shuffled = results.copy()
    m = (shuffled.season == season) & (shuffled["round"] == rnd)
    rng = np.random.default_rng(0)
    for c in ["position", "points", "classified", "grid", "laps"]:
        shuffled.loc[m, c] = rng.permutation(shuffled.loc[m, c].to_numpy())
    pert = _race(build_features(shuffled), season, rnd)
    pd.testing.assert_frame_equal(base[ALL_FEATS], pert[ALL_FEATS])


def test_no_outcome_columns_as_features():
    for f in ALL_FEATS + DEFAULT_SPEC["strength"]["features"] + DEFAULT_SPEC["dnf"]["features"]:
        assert f not in OUTCOME_COLS


def test_harness_trains_only_on_past(results):
    feats = build_features(results)
    seen = []

    class Spy:
        def fit(self, train):
            self.last = tuple(train.sort_values(["season", "round"])[["season", "round"]].iloc[-1])
            return self

        def predict(self, race):
            key = (int(race.season.iloc[0]), int(race["round"].iloc[0]))
            assert self.last < key, f"trained on {self.last} for {key}"
            assert not set(OUTCOME_COLS) & set(race.columns), "outcome columns visible at predict time"
            seen.append(key)
            out = race[["season", "round", "driver_id", "code", "constructor_id", "start_pos"]].copy()
            n = len(out)
            out["p_win"], out["p_top3"], out["p_top10"], out["exp_pos"] = 1 / n, 3 / n, 10 / n, (n + 1) / 2
            out["pred_rank"] = range(1, n + 1)
            return out

    harness.walk_forward(Spy, feats, test_from=2025, test_to=2025)
    assert len(seen) == 24


def test_race_pace_features_leak_free(results):
    from f1.features import RACE_PACE_FEATURES
    season, rnd = 2026, 10
    rng = np.random.default_rng(0)
    keys = results[results.season >= 2025][["season", "round", "driver_id"]]
    pace = keys.assign(race_pace_gap=rng.random(len(keys)) * 2)
    base = _race(build_features(results, race_pace=pace), season, rnd)
    m = (pace.season == season) & (pace["round"] == rnd)
    pace2 = pace.copy()
    pace2.loc[m, "race_pace_gap"] = 9.0          # change this race's own pace
    later = (pace2.season == season) & (pace2["round"] > rnd)
    pace2.loc[later, "race_pace_gap"] = 7.0      # and every later race
    pert = _race(build_features(results, race_pace=pace2), season, rnd)
    pd.testing.assert_frame_equal(base[RACE_PACE_FEATURES], pert[RACE_PACE_FEATURES])
    nxt = _race(build_features(results, race_pace=pace2), season, rnd + 1)
    assert not np.allclose(nxt["drv_race_pace"], _race(build_features(results, race_pace=pace), season, rnd + 1)["drv_race_pace"])
