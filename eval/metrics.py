"""PROTECTED (CLAUDE.md section 9): metric definitions.

Do not modify in an autonomous improvement cycle. Changes need a human-approved
proposal in proposals/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

EPS = 1e-4  # probability floor for log-loss

# metric -> True if higher is better
DIRECTION = {
    "winner_logloss": False, "winner_brier": False, "top1_hit": True, "winner_rank": False,
    "podium_brier": False, "top10_brier": False,
    "spearman": True, "kendall": True, "mae_pos": False,
}
# Promotion objective. Changed by the project owner on 2026-10-25: optimise for the race winner first and the
# podium second; top-10 and the full finishing order are still predicted and reported but no longer decide
# promotions. (Previously an equal-weight mean of winner log-loss, podium Brier, top-10 Brier and Spearman.)
COMPOSITE_WEIGHTS = {"winner_logloss": 0.6, "podium_brier": 0.4}
COMPOSITE = list(COMPOSITE_WEIGHTS)
REPORTED_ONLY = ["top10_brier", "spearman", "kendall", "mae_pos"]


def score_race(pred: pd.DataFrame, actual: pd.DataFrame) -> dict:
    """pred: one row per driver with p_win, p_top3, p_top10, exp_pos, pred_rank.
    actual: driver_id, position (official order, including unclassified)."""
    m = pred.merge(actual[["driver_id", "position"]], on="driver_id", how="inner")
    pos = m["position"].to_numpy()
    win = (pos == 1).astype(float)
    pw = np.clip(m["p_win"].to_numpy(), EPS, 1)
    pw_norm = pw / pw.sum()
    winner_p = pw_norm[win == 1][0] if win.any() else EPS
    rank_by_pwin = (-m["p_win"].to_numpy()).argsort().argsort() + 1
    sp = spearmanr(m["pred_rank"], pos).statistic
    kt = kendalltau(m["pred_rank"], pos).statistic
    return {
        "winner_logloss": float(-np.log(winner_p)),
        "winner_brier": float(((m["p_win"].to_numpy() - win) ** 2).sum()),
        "top1_hit": float(rank_by_pwin[win == 1][0] == 1) if win.any() else 0.0,
        "winner_rank": float(rank_by_pwin[win == 1][0]) if win.any() else float(len(m)),
        "podium_brier": float(((m["p_top3"] - (pos <= 3)) ** 2).mean()),
        "top10_brier": float(((m["p_top10"] - (pos <= 10)) ** 2).mean()),
        "spearman": 0.0 if np.isnan(sp) else float(sp),
        "kendall": 0.0 if np.isnan(kt) else float(kt),
        "mae_pos": float(np.abs(m["exp_pos"] - pos).mean()),
    }


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    err = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            err += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(err)


def calibration_table(p, y, bins: int = 10) -> pd.DataFrame:
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = [{"bin": b, "p_mean": p[idx == b].mean(), "y_rate": y[idx == b].mean(), "n": int((idx == b).sum())}
            for b in range(bins) if (idx == b).any()]
    return pd.DataFrame(rows)


def pooled_calibration(preds: pd.DataFrame) -> dict:
    """preds: stacked per-driver predictions with a `position` column."""
    pos = preds["position"].to_numpy()
    return {
        "ece_win": ece(preds["p_win"], pos == 1),
        "ece_top3": ece(preds["p_top3"], pos <= 3),
        "ece_top10": ece(preds["p_top10"], pos <= 10),
    }


def rps(cum_pred: np.ndarray, actual_rank: int) -> float:
    """Ranked probability score for one competitor. cum_pred[k] = P(final rank <= k+1)."""
    K = len(cum_pred)
    obs = (np.arange(1, K + 1) >= actual_rank).astype(float)
    return float(((cum_pred - obs) ** 2).sum() / max(K - 1, 1))


def summarize(per_race: pd.DataFrame) -> dict:
    return {k: float(per_race[k].mean()) for k in DIRECTION if k in per_race}


def bootstrap_ci(per_race: pd.DataFrame, metric: str, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = per_race[metric].to_numpy()
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
