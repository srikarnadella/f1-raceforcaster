"""PROTECTED (CLAUDE.md sections 8-9): the promotion gate.

A candidate is promoted only if ALL hold against the current champion on the
same walk-forward races:
  1. composite skill (mean relative improvement over winner log-loss, podium
     Brier, top-10 Brier, order Spearman) >= +1%
  2. no single composite target worse by more than 2%
  3. paired bootstrap over races: candidate composite > champion in >= 80% of resamples
  4. calibration not degraded: pooled ECE (win, top-3, top-10) rises by at most 0.005 each
  5. leakage tests pass
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd

from eval.metrics import COMPOSITE, DIRECTION

MIN_COMPOSITE = 0.01
MAX_TARGET_REGRESSION = -0.02
MIN_BOOTSTRAP_WIN = 0.80
MAX_ECE_INCREASE = 0.005
N_BOOT = 2000


def rel_improvement(cand: float, champ: float, metric: str) -> float:
    if DIRECTION[metric]:
        return (cand - champ) / max(abs(champ), 1e-9)
    return (champ - cand) / max(abs(champ), 1e-9)


def composite(cand_means: dict, champ_means: dict) -> tuple[float, dict]:
    rel = {m: rel_improvement(cand_means[m], champ_means[m], m) for m in COMPOSITE}
    return float(np.mean(list(rel.values()))), rel


def paired_bootstrap(cand: pd.DataFrame, champ: pd.DataFrame, n: int = N_BOOT, seed: int = 0) -> float:
    key = ["season", "round"]
    j = cand.merge(champ, on=key, suffixes=("_c", "_k"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(j), (n, len(j)))
    wins = 0
    for row in idx:
        s = j.iloc[row]
        c = {m: s[f"{m}_c"].mean() for m in COMPOSITE}
        k = {m: s[f"{m}_k"].mean() for m in COMPOSITE}
        wins += composite(c, k)[0] > 0
    return wins / n


def leakage_tests_pass() -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_leakage.py"], capture_output=True)
    return r.returncode == 0


def decide(cand: dict, champ: dict, run_leakage: bool = True) -> dict:
    """cand/champ: outputs of eval.harness.evaluate on the same races."""
    comp, rel = composite(cand["summary"], champ["summary"])
    boot = paired_bootstrap(cand["per_race"], champ["per_race"])
    cal_ok = all(cand["calibration"][k] <= champ["calibration"][k] + MAX_ECE_INCREASE for k in champ["calibration"])
    leak_ok = leakage_tests_pass() if run_leakage else True
    checks = {
        "composite": comp >= MIN_COMPOSITE,
        "no_target_regression": min(rel.values()) >= MAX_TARGET_REGRESSION,
        "bootstrap": boot >= MIN_BOOTSTRAP_WIN,
        "calibration": cal_ok,
        "leakage": leak_ok,
    }
    return {"promote": all(checks.values()), "checks": checks, "composite": comp, "relative": rel,
            "bootstrap_win_rate": boot, "calibration_cand": cand["calibration"],
            "calibration_champ": champ["calibration"]}
