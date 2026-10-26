"""Self-contained checks of the analytics methods on synthetic inputs with known answers."""
import numpy as np
import pandas as pd

from f1.analytics import affinity, circuits, racesim, strategy


def test_soft_impute_recovers_a_low_rank_matrix():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(30, 2)) @ rng.normal(size=(2, 20))
    M = truth.copy()
    hide = rng.random(M.shape) < 0.2
    M[hide] = np.nan
    _, _, L = affinity.soft_impute(M, rank=2)
    assert np.abs(L[hide] - truth[hide]).mean() < 0.05 * np.abs(truth).mean()


def test_strategy_optimiser_finds_the_known_optimum():
    # no degradation on hard, heavy on soft: one long hard stint plus the minimum soft stint is best
    plans = strategy.best_strategies(n_laps=50, pit_loss=20.0, offset={"SOFT": 0.0, "MEDIUM": 0.3, "HARD": 0.5},
                                     deg={"SOFT": 0.2, "MEDIUM": 0.1, "HARD": 0.0}, max_stops=2)
    best = plans.iloc[0]
    assert best["stops"] == 1 and set(best["compounds"]) == {"SOFT", "HARD"}
    assert min(best["stints"]) == strategy.MIN_STINT or best["delta_s"] == 0
    # a huge pit loss can never make two stops better than one
    many = strategy.best_strategies(50, 200.0, {"SOFT": 0.0, "MEDIUM": 0.3, "HARD": 0.5},
                                    {"SOFT": 0.2, "MEDIUM": 0.1, "HARD": 0.0}, max_stops=2)
    assert many.iloc[0]["stops"] == 1


def _field(n=10, fast=0):
    return pd.DataFrame({"start_pos": np.arange(1, n + 1), "pace_s": np.where(np.arange(n) == fast, 0.0, 0.5),
                         "sigma": 0.3, "deg_mult": 1.0, "start_skill": 0.0, "attack": 0.0, "defence": 0.0,
                         "p_dnf": 0.05, "base_lap_s": 90.0, "constructor_id": [f"t{i}" for i in range(n)]})


def test_race_simulator_rewards_pace_and_respects_passing():
    prm = racesim.SimParams(n_sims=400)
    pos = racesim.simulate(_field(fast=5), racesim.Circuit(laps=40, passing_ease=0.8), prm, seed=1)
    win = (pos == 1).mean(0)
    assert win[5] == win.max() and win[5] > 0.3          # the fastest car wins most, even from P6
    hard = racesim.simulate(_field(fast=5), racesim.Circuit(laps=40, passing_ease=0.02), prm, seed=1)
    # where passing is impossible, the grid order holds far more often
    moved_easy = np.abs(pos - np.arange(1, 11)).mean()
    moved_hard = np.abs(hard - np.arange(1, 11)).mean()
    assert moved_hard < moved_easy
    assert sorted(pos[0]) == list(range(1, 11))          # every simulated race is a complete order


def test_circuit_clustering_separates_obvious_groups():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(12):
        power = i < 6
        rows.append({"share_straight": 0.7 if power else 0.4, "share_fast": 0.05, "share_medium": 0.1,
                     "share_slow": 0.15 if power else 0.45, "avg_speed_kmh": 240 if power else 180,
                     "top_speed_kmh": 340 if power else 300, "full_throttle_pct": 80 if power else 55,
                     "braking_pct": 10 if power else 20, "gear_changes_per_km": 6 if power else 11,
                     "race_deg": 0.05, "race_track_temp": 35, "sc_rate": 1, "overtakes_per_lap": 0.6,
                     "pit_loss_s": 22, "race_name": f"c{i}"})
    prof = pd.DataFrame(rows, index=[f"c{i}" for i in range(12)])
    prof[circuits.PROFILE] = prof[circuits.PROFILE] + rng.normal(0, 0.01, (12, len(circuits.PROFILE)))
    cl = circuits.cluster(prof, k_range=range(2, 3))
    assert cl.loc[[f"c{i}" for i in range(6)], "cluster"].nunique() == 1
    assert cl.loc["c0", "cluster"] != cl.loc["c11", "cluster"]
