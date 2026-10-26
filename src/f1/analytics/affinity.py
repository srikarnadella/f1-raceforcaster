"""Track affinity: which drivers are unusually strong or weak at which tracks.

Measure (car cancels out): each driver's best qualifying lap vs the teammate
(%, negative = faster), 2014 onward, from the Jolpica store. For every
driver-circuit pair the *affinity* is the driver's mean gap at that circuit
minus the driver's mean gap everywhere (so it says "better or worse than
their own usual level"), shrunk toward 0 by n / (n + SHRINK) because a single
visit is noisy.

Unsupervised layers on the driver x circuit affinity matrix:
  factors     low-rank matrix factorisation with missing cells (soft-impute /
              iterative SVD). Drivers and circuits land in one latent space:
              a driver near a circuit tends to over-perform there.
  co-clusters spectral co-clustering: groups of drivers that share groups of
              tracks they like.
  types       driver x circuit-type matrix (types from circuits.py) and
              k-means driver groups on it (street / power / high-speed ...).
  check       10% of observed cells held out: does the factor model predict
              them better than "same as their usual level" (0)?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SEED

SHRINK = 2.0
MIN_DRIVER_RACES = 25
MIN_CIRCUIT_DRIVERS = 8


def observations(results: pd.DataFrame, since: int = 2014) -> pd.DataFrame:
    r = results[results["season"] >= since].copy()
    r["best_q"] = r[["q1", "q2", "q3"]].min(axis=1)
    g = r.groupby(["season", "round", "constructor_id"])["best_q"]
    n, tot = g.transform("count"), g.transform("sum")
    other = (tot - r["best_q"]) / (n - 1).where(n > 1)
    r["tm_gap"] = ((r["best_q"] / other - 1) * 100).where(n == 2)
    # a lap ruined by a crash, rain timing or a penalty is not information about the track
    r = r[r["tm_gap"].abs() < 2.0]
    return r[["season", "round", "circuit_id", "driver_id", "code", "constructor_id", "tm_gap"]]


def affinity_matrix(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(long table with n and shrunk affinity, wide driver x circuit matrix with NaN for unseen)."""
    keep_d = obs["driver_id"].value_counts()
    keep_d = keep_d[keep_d >= MIN_DRIVER_RACES].index
    o = obs[obs["driver_id"].isin(keep_d)]
    base = o.groupby("driver_id")["tm_gap"].mean()
    cell = o.groupby(["driver_id", "circuit_id"])["tm_gap"].agg(["mean", "count"]).reset_index()
    cell["raw"] = cell["mean"] - cell["driver_id"].map(base)
    cell["affinity"] = cell["raw"] * cell["count"] / (cell["count"] + SHRINK)
    nd = cell.groupby("circuit_id")["driver_id"].nunique()
    cell = cell[cell["circuit_id"].isin(nd[nd >= MIN_CIRCUIT_DRIVERS].index)]
    wide = cell.pivot(index="driver_id", columns="circuit_id", values="affinity")
    return cell, wide


def soft_impute(M: np.ndarray, rank: int = 2, iters: int = 200, tol: float = 1e-6):
    """Low-rank completion of a matrix with NaNs. Returns (U*S, V, completed)."""
    mask = ~np.isnan(M)
    X = np.where(mask, M, 0.0)
    Z = X.copy()
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        L = (U[:, :rank] * s[:rank]) @ Vt[:rank]
        newZ = np.where(mask, M, L)
        if np.linalg.norm(newZ - Z) < tol * (np.linalg.norm(Z) + 1e-9):
            Z = newZ
            break
        Z = newZ
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    return U[:, :rank] * s[:rank], Vt[:rank].T, (U[:, :rank] * s[:rank]) @ Vt[:rank]


def holdout_check(wide: pd.DataFrame, rank: int = 2, frac: float = 0.1, seed: int = SEED) -> dict:
    """Hide some observed cells, complete the matrix, compare with predicting 0."""
    M = wide.to_numpy(float)
    obs = np.argwhere(~np.isnan(M))
    rng = np.random.default_rng(seed % 2**32)
    errs: dict[str, list[float]] = {"factor_model": [], "usual_level (0)": []}
    for rep in range(5):
        pick = obs[rng.random(len(obs)) < frac]
        M2 = M.copy()
        M2[pick[:, 0], pick[:, 1]] = np.nan
        _, _, L = soft_impute(M2, rank)
        truth = M[pick[:, 0], pick[:, 1]]
        errs["factor_model"] += list(np.abs(L[pick[:, 0], pick[:, 1]] - truth))
        errs["usual_level (0)"] += list(np.abs(truth))
    return {k: round(float(np.mean(v)), 4) for k, v in errs.items()} | {"cells": int(len(obs)), "rank": rank}


def factors(wide: pd.DataFrame, rank: int = 2):
    U, V, L = soft_impute(wide.to_numpy(float), rank)
    d = pd.DataFrame(U, index=wide.index, columns=[f"f{i + 1}" for i in range(rank)])
    c = pd.DataFrame(V, index=wide.columns, columns=[f"f{i + 1}" for i in range(rank)])
    # put both on a comparable scale for a biplot
    c = c * (np.abs(d.to_numpy()).mean() / (np.abs(c.to_numpy()).mean() + 1e-9))
    return d, c, pd.DataFrame(L, index=wide.index, columns=wide.columns)


def coclusters(completed: pd.DataFrame, k: int = 4, seed: int = 0):
    from sklearn.cluster import SpectralCoclustering
    X = completed.to_numpy(float)
    X = X - X.min() + 1e-3  # the method needs positive entries; "affinity" = -gap so faster is larger
    X = (-completed.to_numpy(float))
    X = X - X.min() + 1e-3
    m = SpectralCoclustering(n_clusters=k, random_state=seed).fit(X)
    return (pd.Series(m.row_labels_, index=completed.index, name="cocluster"),
            pd.Series(m.column_labels_, index=completed.columns, name="cocluster"))


def by_type(cell: pd.DataFrame, types: dict, k: int = 4, seed: int = 0):
    """Driver x circuit-type affinity (count-weighted) and k-means driver groups on it."""
    from sklearn.cluster import KMeans
    c = cell.assign(ctype=cell["circuit_id"].map(types)).dropna(subset=["ctype"])
    t = c.groupby(["driver_id", "ctype"]).apply(lambda g: np.average(g["affinity"], weights=g["count"]),
                                                include_groups=False).unstack()
    t = t.dropna(thresh=max(2, t.shape[1] - 1))
    X = t.fillna(0).to_numpy()
    k = min(k, len(t) - 1)
    km = KMeans(k, n_init=20, random_state=seed).fit(X)
    cent = pd.DataFrame(km.cluster_centers_, columns=t.columns)
    names = {}
    for i in range(k):
        best = cent.loc[i].idxmin()      # most negative = fastest vs own level
        names[i] = f"strongest at {best}" if cent.loc[i].min() < -0.03 else "even across track types"
    seen: dict[str, int] = {}
    for i in list(names):
        seen[names[i]] = seen.get(names[i], 0) + 1
        if seen[names[i]] > 1:
            names[i] = f"{names[i]} ({seen[names[i]]})"
    return t, pd.Series([names[x] for x in km.labels_], index=t.index, name="track_group")


def temporal_check(obs: pd.DataFrame, types: dict, shrink: float = 8.0) -> dict:
    """Does a driver's record at a circuit (or circuit type) from earlier visits predict the
    next visit, beyond their usual level? Residual = gap minus the driver's running mean."""
    o = obs.sort_values(["season", "round"]).copy()
    o["ctype"] = o["circuit_id"].map(types)
    o["res"] = o["tm_gap"] - o.groupby("driver_id")["tm_gap"].transform(lambda s: s.shift(1).expanding().mean())
    out = {}
    for lvl, name in (("circuit_id", "same circuit"), ("ctype", "same circuit type")):
        prev = o.groupby(["driver_id", lvl])["res"].transform(lambda s: s.shift(1).expanding().sum())
        n = o.groupby(["driver_id", lvl]).cumcount()
        pred = (prev / (n + shrink)).where(n > 0)
        ok = pred.notna() & o["res"].notna()
        out[name] = {"mae_model": round(float((pred[ok] - o.loc[ok, "res"]).abs().mean()), 4),
                     "mae_usual_level": round(float(o.loc[ok, "res"].abs().mean()), 4),
                     "correlation": round(float(np.corrcoef(pred[ok], o.loc[ok, "res"])[0, 1]), 3),
                     "visits": int(ok.sum())}
    return out


def significance(obs: pd.DataFrame, cell: pd.DataFrame, key: str = "circuit_id") -> pd.DataFrame:
    """t-statistic of each driver's residual at a circuit (vs their own mean), visits >= 3."""
    base = obs.groupby("driver_id")["tm_gap"].mean()
    o = obs.assign(res=obs["tm_gap"] - obs["driver_id"].map(base))
    g = o.groupby(["driver_id", key])["res"].agg(["mean", "std", "count"]).reset_index()
    g["t"] = g["mean"] / (g["std"] / np.sqrt(g["count"])).replace(0, np.nan)
    return cell.merge(g[["driver_id", key, "t"]], on=["driver_id", key], how="left")
