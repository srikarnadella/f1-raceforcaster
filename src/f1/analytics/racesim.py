"""Agent-based race simulator: driver and car profiles race each other lap by lap.

Each simulated race (vectorised over S simulations):
  start    cars line up 0.25 s per grid slot; lap 1 adds a shuffle (noise plus
           the driver's start skill).
  pace     lap time = base + race pace (car + driver) + a per-race pace error
           (our estimate is never exact: drawn once per simulated race, partly
           shared by teammates) + compound offset
           + tyre wear x age (car and driver tyre skill scale it) + noise
           (the driver's consistency scales it).
  traffic  a car that would end the lap ahead of the car in front only gets by
           with probability sigmoid(c0 + c1 * pace advantage + c2 * (passing
           ease - 0.5) + c3 * (attack - defence)); otherwise it is held up
           (finishes the lap 0.25 s behind, plus dirty air). Cars in the pits
           are passed freely.
  strategy each car follows the circuit's best plan with its stops jittered.
  chaos    safety cars arrive at the circuit's rate per lap, bunch the field
           and make a stop half price; cars retire at their car's DNF rate.
Win, podium, top-10 and expected position come from the finishing orders.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import SEED

COMPOUND_IDX = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}


@dataclass
class SimParams:
    w_quali: float = 0.6        # weight of this weekend's qualifying gap in race pace (vs profile prior)
    lap_noise: float = 0.35     # s, base lap-to-lap noise
    c0: float = -1.0
    c1: float = 3.0             # per second of pace advantage
    c2: float = 3.0             # passing ease
    c3: float = 0.15            # attack - defence (passes per race vs teammate)
    start_noise: float = 0.7    # s, lap-1 shuffle
    pit_jitter: float = 2.5     # laps
    pace_unc: float = 0.35      # % of lap: how wrong our pace estimate can be on the day (drawn once per race)
    team_share: float = 0.6     # share of that uncertainty common to both cars of a team
    n_sims: int = 1000


@dataclass
class Circuit:
    laps: int = 57
    pit_loss: float = 22.0
    sc_per_race: float = 1.0
    passing_ease: float = 0.5
    deg: dict = field(default_factory=lambda: {"SOFT": 0.08, "MEDIUM": 0.05, "HARD": 0.035})
    offset: dict = field(default_factory=lambda: {"SOFT": 0.0, "MEDIUM": 0.35, "HARD": 0.7})
    plan: list = field(default_factory=lambda: [("MEDIUM", 0.45), ("HARD", 0.55)])  # (compound, share of race)


def simulate(drv: pd.DataFrame, circ: Circuit, prm: SimParams, seed: int = SEED) -> np.ndarray:
    """drv columns: start_pos, pace_s (s/lap vs best), sigma (s), deg_mult, start_skill, attack, defence, p_dnf.
    Returns (S, N) finishing positions (1 = winner)."""
    rng = np.random.default_rng(seed)
    S, N, L = prm.n_sims, len(drv), int(circ.laps)
    base_s = drv["base_lap_s"].to_numpy(float) if "base_lap_s" in drv else np.full(N, 90.0)
    team = pd.factorize(drv["constructor_id"])[0] if "constructor_id" in drv else np.arange(N)
    unc = prm.pace_unc / 100 * base_s
    t_err = rng.normal(0, 1, (S, team.max() + 1))[:, team]
    d_err = rng.normal(0, 1, (S, N))
    pace = drv["pace_s"].to_numpy(float)[None, :] + unc[None, :] * (np.sqrt(prm.team_share) * t_err
                                                                   + np.sqrt(1 - prm.team_share) * d_err)
    sig = drv["sigma"].to_numpy(float)
    degm = drv["deg_mult"].to_numpy(float)
    att, dfn = drv["attack"].to_numpy(float), drv["defence"].to_numpy(float)
    deg = np.array([circ.deg.get(c, 0.05) for c in COMPOUND_IDX])
    off = np.array([circ.offset.get(c, 0.35) for c in COMPOUND_IDX])
    # strategy: stop laps per car per sim
    shares = np.cumsum([s for _, s in circ.plan])[:-1]
    comps = np.array([COMPOUND_IDX[c] for c, _ in circ.plan])
    stops = np.clip(np.round(shares[None, None, :] * L + rng.normal(0, prm.pit_jitter, (S, N, len(shares)))), 2, L - 2)
    stops = np.sort(stops, axis=2)
    # start: 0.25 s per grid slot plus a lap-1 shuffle
    cum = 0.25 * (drv["start_pos"].to_numpy(float) - 1)[None, :] + rng.normal(0, prm.start_noise, (S, N)) \
        - 0.3 * drv["start_skill"].to_numpy(float)[None, :]
    stint = np.zeros((S, N), int)
    age = np.ones((S, N))
    alive = np.ones((S, N), bool)
    haz = 1 - (1 - drv["p_dnf"].to_numpy(float)) ** (1 / L)
    sc_p = circ.sc_per_race / L
    ease = circ.passing_ease - 0.5
    rows = np.arange(S)[:, None]
    for lap in range(1, L + 1):
        comp = comps[np.minimum(stint, len(comps) - 1)]
        t = pace + off[comp] + deg[comp] * degm[None, :] * age + rng.normal(0, 1, (S, N)) * sig[None, :]
        sc = rng.random(S) < sc_p
        # stop now if planned, or if a safety car comes within 10 laps of the next planned stop
        nxt = np.take_along_axis(stops, np.minimum(stint, stops.shape[2] - 1)[:, :, None], axis=2)[:, :, 0]
        has_more = stint < stops.shape[2]
        pit = has_more & ((nxt == lap) | (sc[:, None] & (nxt - lap <= 10) & (nxt > lap)))
        t = t + np.where(pit, circ.pit_loss * np.where(sc[:, None], 0.5, 1.0), 0.0)
        stint = stint + pit
        age = np.where(pit, 1, age + 1)
        # overtaking, front to back in the order at the start of the lap
        order = np.argsort(np.where(alive, cum, np.inf), axis=1)
        new = cum + t
        for k in range(1, N):
            j, i = order[:, k - 1], order[:, k]
            nj, ni = new[rows[:, 0], j], new[rows[:, 0], i]
            would = ni < nj + 0.2
            adv = t[rows[:, 0], j] - t[rows[:, 0], i]
            p = 1 / (1 + np.exp(-(prm.c0 + prm.c1 * adv + prm.c2 * ease + prm.c3 * (att[i] - dfn[j]))))
            free = pit[rows[:, 0], j] & ~pit[rows[:, 0], i]
            ok = free | (rng.random(S) < p)
            stuck = would & ~ok & alive[rows[:, 0], i] & alive[rows[:, 0], j]
            new[rows[:, 0], i] = np.where(stuck, nj + 0.25 + 0.1, ni)
        cum = new
        if sc.any():  # bunch the field behind the safety car
            o2 = np.argsort(np.where(alive, cum, np.inf), axis=1)
            lead = np.take_along_axis(cum, o2[:, :1], axis=1)
            rank = np.empty_like(o2)
            rank[rows, o2] = np.arange(N)[None, :]
            cum = np.where(sc[:, None] & alive, lead + 0.6 * rank, cum)
        alive &= rng.random((S, N)) >= haz[None, :]
    final = np.where(alive, cum, 1e9 + rng.random((S, N)))
    order = np.argsort(final, axis=1)
    pos = np.empty_like(order)
    pos[rows, order] = np.arange(1, N + 1)[None, :]
    return pos
