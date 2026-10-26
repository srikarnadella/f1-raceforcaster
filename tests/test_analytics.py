"""Analytics layer: leak-free pre-race features, pace fits that recover known
effects, telemetry that agrees with official timing, a correct neural gradient."""
import numpy as np
import pandas as pd
import pytest

from f1.analytics import build as B
from f1.analytics import dna as D
from f1.analytics import prerace
from f1.analytics import telemetry as T
from f1.analytics import weekend as W

STORED = B.weekends()
needs_data = pytest.mark.skipif(len(STORED) < 2, reason="needs >= 2 stored OpenF1 weekends")


@pytest.fixture(scope="module")
def tables():
    import warnings
    warnings.simplefilter("ignore")
    cal = B.ingest.load_calendar()
    parts = {}
    for s, r in STORED:
        for k, v in B.build_weekend(s, r, cal).items():
            if len(v):
                parts.setdefault(k, []).append(v)
    return {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}


def _rows(of1, s, r):
    return of1[(of1.season == s) & (of1["round"] == r)].sort_values("driver_id").reset_index(drop=True)


@needs_data
def test_prerace_features_ignore_race_and_future(tables):
    s, r = STORED[-2]
    base = _rows(prerace.table(tables), s, r)
    t = {k: v.copy() for k, v in tables.items()}
    own = lambda d: (d.season == s) & (d["round"] == r)
    later = lambda d: (d.season > s) | ((d.season == s) & (d["round"] > r))
    # this weekend's race (and sprint pace is pre-race, so only R) and everything after it
    t["one_lap"] = t["one_lap"][~(own(t["one_lap"]) & (t["one_lap"].session == "R"))]
    t["race_pace"] = t["race_pace"][~(own(t["race_pace"]) & (t["race_pace"].session == "R"))]
    for k in ("one_lap", "long_runs", "zones"):
        m = later(t[k])
        num = t[k].select_dtypes("number").columns.difference(["season", "round", "driver_number", "corner"])
        t[k].loc[m, num] = t[k].loc[m, num] * 3 + 1
    pert = _rows(prerace.table(t), s, r)
    pd.testing.assert_frame_equal(base, pert)


@needs_data
def test_rolling_dna_never_sees_its_own_weekend(tables):
    tw = D.team_weekend(tables["zones"], tables["one_lap"], pd.DataFrame(), pd.DataFrame())
    s, r = STORED[-1]
    a = D.rolling_dna(tw)
    tw2 = tw.copy()
    m = (tw2.season == s) & (tw2["round"] == r)
    tw2.loc[m, ["z_straight", "z_slow", "quali_vs_median_pct"]] = 99.0
    b = D.rolling_dna(tw2)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.synthetic
def test_fixed_effects_fit_recovers_known_effects():
    rng = np.random.default_rng(0)
    rows = []
    gap = {"a": 0.0, "b": 0.3, "c": 0.8, "d": 0.5, "e": 1.1, "f": 0.2}
    off = {"SOFT": 0.0, "MEDIUM": 0.5, "HARD": 0.9}
    deg = {"SOFT": 0.12, "MEDIUM": 0.07, "HARD": 0.03}
    # drivers stop on different laps and run compounds in different orders, as in real races
    plans = [("SOFT", "MEDIUM", "HARD"), ("MEDIUM", "HARD", "SOFT"), ("HARD", "SOFT", "MEDIUM")]
    for i, d in enumerate(gap):
        lens = rng.integers(8, 22, 3)
        lap = 0
        for comp, n in zip(plans[i % 3], lens):
            for age in range(1, n + 1):
                lap += 1
                t = 90 + gap[d] + off[comp] + deg[comp] * age - 0.05 * lap + rng.normal(0, 0.05)
                rows.append({"driver_id": d, "compound": comp, "tyre_age": age, "lap": lap, "lap_s": t})
    eff, extra = W._fe_fit(pd.DataFrame(rows), fuel=True)
    g = eff - eff.min()
    for d in gap:
        assert abs(g[d] - gap[d]) < 0.05
    assert abs(extra["fuel_s_per_lap"] + 0.05) < 0.01
    for c in deg:
        assert abs(extra["deg_s_per_lap"][c] - deg[c]) < 0.02
    assert abs(extra["compound_offset_s"]["HARD"] - 0.9) < 0.1


@needs_data
def test_telemetry_matches_official_timing():
    s, r = next((s, r) for s, r in reversed(STORED) if T.load(s, r) is not None)
    w = W.load(s, r)
    tel = T.load(s, r)
    chk = T.sector_check(tel, w["laps"])
    assert len(chk) > 10
    assert chk[["err_s1", "err_s2", "err_s3"]].abs().max().max() < 0.01
    zt, cls, _ = T.zone_table(tel, w["drivers"])
    total = cls.groupby("driver_id")["delta_to_pole_s"].sum()
    gap = cls.groupby("driver_id")["lap_s"].first() - cls["lap_s"].min()
    assert np.allclose(total, gap.reindex(total.index), atol=1e-6)


def test_neural_ranker_gradient(results):
    from scipy.optimize import check_grad
    from scipy import optimize
    from f1.analytics import ml
    from f1.features import STRENGTH_FEATURES, build_features
    f = build_features(results[results.season >= 2024])
    f = f[f.position.notna()]
    captured = {}
    real = optimize.minimize

    def spy(fun, x0, **kw):
        captured["f"], captured["x0"] = fun, x0
        return real(fun, x0, **{**kw, "options": {"maxiter": 2}})
    ml.minimize = spy
    try:
        ml.NeuralPL(STRENGTH_FEATURES, hidden=4, n_models=1).fit(f)
    finally:
        ml.minimize = real
    fun, x0 = captured["f"], captured["x0"] + 0.1
    assert check_grad(lambda x: fun(x)[0], lambda x: fun(x)[1], x0) < 1e-4


@pytest.mark.synthetic
def test_era_factor_and_current_estimates():
    from f1.analytics import eras as E
    rows = []
    for c, base in [("a", 0.4), ("b", 0.8), ("c", 0.2), ("d", 0.6)]:
        rows.append({"circuit_id": c, "era": 2022, "per_lap": base})
        if c != "d":  # d not yet raced under the new rules
            rows.append({"circuit_id": c, "era": 2026, "per_lap": base * 1.5})
    df = pd.DataFrame(rows)
    f = E.era_factor(df, "per_lap", 2026, 2022, shrink=0)
    assert abs(f["factor"] - 1.5) < 1e-9 and f["circuits"] == 3
    shrunk = E.era_factor(df, "per_lap", 2026, 2022)
    assert 1.0 < shrunk["factor"] < 1.5 and abs(shrunk["raw_factor"] - 1.5) < 1e-9   # few circuits: pulled toward 1
    monkey = E.FACTOR_SHRINK
    E.FACTOR_SHRINK = 0.0
    est = E.current_estimates(df, "per_lap", 2026)
    E.FACTOR_SHRINK = monkey
    assert abs(est.loc["d", "estimate"] - 0.6 * 1.5) < 1e-9          # unseen: previous era x factor
    assert abs(est.loc["a", "estimate"] - 0.6) < 1e-9                # seen: own value (prior agrees)
    assert est.loc["d", "current_era_races"] == 0


@pytest.mark.synthetic
def test_passing_difficulty_uses_only_earlier_races():
    w = pd.DataFrame({"season": [2025, 2026, 2026], "round": [1, 1, 2], "circuit_id": ["x", "x", "y"]})
    ov = pd.DataFrame({"season": [2025, 2026, 2026], "round": [1, 1, 2], "circuit_id": ["x", "x", "y"],
                       "era": [2022, 2026, 2026], "per_lap": [0.4, 0.8, 0.2]})
    base = prerace._passing_difficulty(w, ov)
    ov2 = ov.copy()
    ov2.loc[ov2["round"] == 2, "per_lap"] = 9.0     # a race's own passes must not move its own feature
    pd.testing.assert_frame_equal(base, prerace._passing_difficulty(w, ov2))
    ov3 = ov.copy()
    ov3.loc[(ov3.season == 2026) & (ov3["round"] == 1), "per_lap"] = 0.05  # an earlier race does
    assert not base.equals(prerace._passing_difficulty(w, ov3))


@needs_data
def test_race_simulator_sees_no_outcome_of_its_race(results):
    """The simulator's prediction for a race must not move when that race's own outcome
    (race laps, traits, overtakes, result) or anything after it changes."""
    from f1.analytics.simmodel import RaceSimModel, SimInputs
    from f1.analytics.racesim import SimParams
    from f1.features import build_features
    from f1 import ingest
    res = ingest.load_results()
    s, r = 2025, 10
    feats = build_features(res)
    race = feats[(feats.season == s) & (feats["round"] == r)].drop(columns=["position"])
    prm = SimParams(n_sims=300)
    base = RaceSimModel(SimInputs(results=res), prm).predict(race)
    t = {k: B.read(k) for k in ("one_lap", "race_pace", "driver_traits", "race_story", "overtakes", "weekend")}
    later = lambda d: (d.season > s) | ((d.season == s) & (d["round"] >= r))
    for k in ("race_pace", "driver_traits", "overtakes"):
        num = t[k].select_dtypes("number").columns.difference(["season", "round"])
        t[k].loc[later(t[k]), num] = t[k].loc[later(t[k]), num] * 2 + 3
    t["one_lap"] = t["one_lap"][~(later(t["one_lap"]) & (t["one_lap"].session == "R"))]
    res2 = res.copy()
    m = later(res2) & (res2.season == s) & (res2["round"] == r)
    res2.loc[m, "position"] = res2.loc[m, "position"].to_numpy()[::-1]
    pert = RaceSimModel(SimInputs(tables=t, results=res2), prm).predict(race)
    pd.testing.assert_frame_equal(base[["driver_id", "p_win", "p_top3"]].reset_index(drop=True),
                                  pert[["driver_id", "p_win", "p_top3"]].reset_index(drop=True))


@needs_data
def test_practice_features_attach_to_model_rows(results):
    """build_features(openf1=True) must attach every OpenF1 feature (guards the live challengers)."""
    from f1.analytics.prerace import OF1_FEATURES
    from f1.features import build_features
    f = build_features(results[results.season >= 2023], openf1=True)
    assert set(OF1_FEATURES) <= set(f.columns)
    assert f.loc[f.season >= 2024, "has_of1"].mean() > 0.9
