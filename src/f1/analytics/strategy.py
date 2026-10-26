"""Race strategy optimiser.

Stint cost with compound c over L laps on fresh tyres (age 1..L):
    L * offset_c + deg_c * L (L + 1) / 2
Total = sum of stints + stops * pit_loss. Fuel burn is the same for every
strategy, so it drops out. Dry races must use two different compounds.
Inputs come from earlier visits to the circuit (fitted race degradation and
compound offsets, measured pit loss), so this is a projection, not a fact:
it ignores safety cars and tyre cliffs (degradation is linear).

Track position (an explicit assumption, shown separately): every stop drops a
car into traffic it must re-pass. Where passing is hard that costs time, so
each stop is charged TRACK_POSITION_S * (1 - passing ease), with passing ease
= overtakes per lap / 0.8 capped at 1 (0.8 ~ the easiest tracks, Monza / Spa).
Per-circuit degradation is shrunk toward the field-wide median with the
weight of one visit, because one race's fit is noisy.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

DRY = ["SOFT", "MEDIUM", "HARD"]
MIN_STINT = 5
TRACK_POSITION_S = 12.0
EASY_PASSING = 0.8


def stint_cost(L, off, deg):
    return L * off + deg * L * (L + 1) / 2


def best_strategies(n_laps: int, pit_loss: float, offset: dict, deg: dict, max_stops: int = 3,
                    top: int = 3, stop_penalty: float = 0.0) -> pd.DataFrame:
    """Fastest plans for 1..max_stops stops, ranked by total time relative to the best."""
    rows = []
    comps = [c for c in DRY if c in offset and c in deg and pd.notna(offset[c]) and pd.notna(deg[c])]
    for stops in range(1, max_stops + 1):
        k = stops + 1
        best_for_k, seen = [], set()
        for seq in itertools.product(comps, repeat=k):
            if len(set(seq)) < 2 or tuple(sorted(seq)) in seen:
                continue  # stint order does not change linear-model time; keep one ordering
            seen.add(tuple(sorted(seq)))
            # optimal split for a fixed sequence: brute force on a coarse grid then refine
            best = None
            for cuts in _splits(n_laps, k):
                t = sum(stint_cost(L, offset[c], max(deg[c], 0.0)) for L, c in zip(cuts, seq))
                if best is None or t < best[0]:
                    best = (t, cuts)
            if best:
                best_for_k.append({"stops": stops, "plan": " → ".join(f"{c[0]}{L}" for c, L in zip(seq, best[1])),
                                   "compounds": seq, "stints": best[1], "pace_time_s": best[0] + stops * pit_loss,
                                   "time_s": best[0] + stops * (pit_loss + stop_penalty)})
        rows += sorted(best_for_k, key=lambda r: r["time_s"])[:top]
    out = pd.DataFrame(rows).sort_values("time_s").reset_index(drop=True)
    out["delta_s"] = out["time_s"] - out["time_s"].min()
    out["pace_only_delta_s"] = out["pace_time_s"] - out["pace_time_s"].min()
    return out


def _splits(n, k):
    if k == 2:
        for a in range(MIN_STINT, n - MIN_STINT + 1):
            yield (a, n - a)
    elif k == 3:
        for a in range(MIN_STINT, n - 2 * MIN_STINT + 1):
            for b in range(MIN_STINT, n - a - MIN_STINT + 1):
                yield (a, b, n - a - b)
    else:  # 4 stints: coarser grid keeps it fast
        for a in range(MIN_STINT, n - 3 * MIN_STINT + 1, 2):
            for b in range(MIN_STINT, n - a - 2 * MIN_STINT + 1, 2):
                for c in range(MIN_STINT, n - a - b - MIN_STINT + 1, 2):
                    yield (a, b, c, n - a - b - c)


def for_circuit(prof: pd.Series, field: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Plans from a circuit profile row (circuits.profiles); `field` = all profiles,
    used for the degradation prior."""
    n = prof.get("race_laps")
    if pd.isna(n) or pd.isna(prof.get("pit_loss_s")):
        return None
    deg = {"SOFT": prof.get("race_deg_soft"), "MEDIUM": prof.get("race_deg_medium"), "HARD": prof.get("race_deg_hard")}
    off = {"SOFT": 0.0, "MEDIUM": prof.get("race_off_medium"), "HARD": prof.get("race_off_hard")}
    # a compound nobody raced at this circuit has no fitted values: borrow a typical step
    if pd.isna(off["MEDIUM"]):
        off["MEDIUM"] = 0.35
    if pd.isna(off["HARD"]):
        off["HARD"] = off["MEDIUM"] + 0.35
    for c in deg:
        col = f"race_deg_{c.lower()}"
        prior = float(field[col].median()) if field is not None and col in field else 0.05
        w = float(prof.get("weekends", 1) or 1)
        deg[c] = prior if pd.isna(deg[c]) else (w * deg[c] + prior) / (w + 1)
    ease = min(float(prof.get("overtakes_per_lap", EASY_PASSING) or 0) / EASY_PASSING, 1.0)
    out = best_strategies(int(n), float(prof["pit_loss_s"]), off, deg,
                          stop_penalty=TRACK_POSITION_S * (1 - ease))
    out.attrs.update(deg=deg, offset=off, pit_loss=float(prof["pit_loss_s"]), passing_ease=ease)
    return out


def validate(weekend: pd.DataFrame, penalty: float | None = None) -> tuple[pd.DataFrame, dict]:
    """Walk-forward: each dry race's optimal stop count, projected from earlier
    visits only, vs the field's actual median stop count. Baseline: same stop
    count as the circuit's previous dry race."""
    from .circuits import profiles
    global TRACK_POSITION_S
    old = TRACK_POSITION_S
    if penalty is not None:
        TRACK_POSITION_S = penalty
    try:
        w = weekend.sort_values(["season", "round"]).reset_index(drop=True)
        rows = []
        for i, r in w.iterrows():
            if r.get("race_rain_share", 0) or pd.isna(r.get("race_stops_median")):
                continue
            prior = w.iloc[:i]
            if not (prior["circuit_id"] == r["circuit_id"]).any():
                continue
            prof = profiles(prior)
            if r["circuit_id"] not in prof.index:
                continue
            plans = for_circuit(prof.loc[r["circuit_id"]], prof)
            if plans is None or plans.empty:
                continue
            last = prior[(prior["circuit_id"] == r["circuit_id"]) & (prior["race_rain_share"].fillna(0) == 0)]
            rows.append({"season": r["season"], "round": r["round"], "circuit_id": r["circuit_id"],
                         "actual_stops": r["race_stops_median"], "pred_stops": int(plans["stops"].iat[0]),
                         "last_visit_stops": last["race_stops_median"].iat[-1] if len(last) else np.nan,
                         "plan": plans["plan"].iat[0]})
    finally:
        TRACK_POSITION_S = old
    v = pd.DataFrame(rows)
    if v.empty:
        return v, {}
    v["actual_round"] = v["actual_stops"].round()
    # both methods scored on the same races: those with an earlier dry visit to copy
    ok = v["last_visit_stops"].notna()
    c = v[ok]
    res = {"n": int(ok.sum()),
           "exact_model": round(float((c["pred_stops"] == c["actual_round"]).mean()), 3),
           "mae_model": round(float((c["pred_stops"] - c["actual_stops"]).abs().mean()), 3),
           "exact_last_visit": round(float((c["last_visit_stops"].round() == c["actual_round"]).mean()), 3),
           "mae_last_visit": round(float((c["last_visit_stops"] - c["actual_stops"]).abs().mean()), 3)}
    return v, res
