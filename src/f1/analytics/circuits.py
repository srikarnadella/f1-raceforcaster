"""Unsupervised circuit typing.

Each circuit is described by where its pole lap spends time (straights, fast /
medium / slow corners), speeds, braking, gear changes, race tyre degradation,
track temperature and safety-car rate. Standardised profiles are clustered
with k-means (k picked by silhouette) and projected to 2-D with PCA. Cluster
names come from each centroid's strongest trait, so they describe the data
rather than a preset list.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from ..config import SEED

PROFILE = ["share_straight", "share_fast", "share_medium", "share_slow", "avg_speed_kmh", "top_speed_kmh",
           "full_throttle_pct", "braking_pct", "gear_changes_per_km", "race_deg", "race_track_temp", "sc_rate",
           "overtakes_per_lap", "pit_loss_s"]
TRAITS = {  # feature -> description when the centroid is high on it
    "share_straight": "power / long straights", "top_speed_kmh": "power / long straights",
    "share_slow": "stop-go, slow corners", "braking_pct": "heavy braking",
    "share_fast": "high-speed flowing", "avg_speed_kmh": "high-speed flowing",
    "share_medium": "medium-speed technical", "gear_changes_per_km": "twisty / technical",
    "race_deg": "high tyre degradation", "race_track_temp": "hot track", "sc_rate": "safety-car prone",
    "overtakes_per_lap": "overtaking-friendly", "pit_loss_s": "expensive pit stops",
    "full_throttle_pct": "power / long straights",
}


def profiles(weekend: pd.DataFrame) -> pd.DataFrame:
    w = weekend.copy()
    w["race_deg"] = w[["race_deg_soft", "race_deg_medium", "race_deg_hard"]].median(axis=1)
    w["sc_rate"] = w[["race_sc", "race_vsc"]].sum(axis=1, min_count=1)
    w = w.sort_values(["season", "round"])
    # layout traits from the most recent telemetry; conditions averaged over years
    cond_cols = ["race_deg", "race_track_temp", "sc_rate", "overtakes_per_lap", "pit_loss_s", "race_laps",
                 "race_deg_soft", "race_deg_medium", "race_deg_hard", "race_off_medium", "race_off_hard"]
    layout = [c for c in PROFILE if c not in cond_cols]
    last = w.dropna(subset=["share_straight"]).groupby("circuit_id")[layout].last()
    cond = w.groupby("circuit_id")[[c for c in cond_cols if c in w.columns]].mean()
    # tyre fits only from dry races (rain makes compound offsets and wear meaningless)
    tyre = ["race_deg", "race_deg_soft", "race_deg_medium", "race_deg_hard", "race_off_medium", "race_off_hard"]
    dry = w[w["race_rain_share"].fillna(0) == 0]
    cond[tyre] = dry.groupby("circuit_id")[tyre].mean().reindex(cond.index)
    name = w.groupby("circuit_id")["race_name"].last()
    n = w.groupby("circuit_id").size().rename("weekends")
    return last.join(cond).join(name).join(n)


def cluster(prof: pd.DataFrame, k_range=range(3, 7)) -> pd.DataFrame:
    p = prof.dropna(subset=["share_straight"]).copy()
    X = p[PROFILE].fillna(p[PROFILE].median())
    Z = ((X - X.mean()) / (X.std() + 1e-9)).to_numpy()
    best: tuple | None = None
    for k in k_range:
        if k >= len(p):
            break
        km = KMeans(k, n_init=20, random_state=SEED % 2**31).fit(Z)
        s = silhouette_score(Z, km.labels_)
        if best is None or s > best[0]:
            best = (s, k, km)
    assert best is not None, "need more circuits than clusters"
    s, k, km = best
    p["cluster"] = km.labels_
    pca = PCA(2, random_state=0).fit(Z)
    xy = pca.transform(Z)
    p["pc1"], p["pc2"] = xy[:, 0], xy[:, 1]
    cent = pd.DataFrame(km.cluster_centers_, columns=PROFILE)
    names, used = {}, set()
    for c in range(k):
        order = cent.loc[c].sort_values(ascending=False).index
        picks = []
        for f in order:
            t = TRAITS.get(f, f)
            if t not in picks:
                picks.append(t)
            if len(picks) == 2:
                break
        nm = picks[0] if picks[0] not in used else f"{picks[0]} + {picks[1]}"
        used.add(nm)
        names[c] = nm
    p["cluster_name"] = p["cluster"].map(names)
    p.attrs.update(k=int(k), silhouette=round(float(s), 3),
                   pc_loadings={f"pc{i + 1}": dict(zip(PROFILE, np.round(pca.components_[i], 3))) for i in range(2)},
                   explained=list(np.round(pca.explained_variance_ratio_, 3)))
    return p


def similar(prof: pd.DataFrame, circuit_id: str, n: int = 3) -> pd.Series:
    X = prof[PROFILE].fillna(prof[PROFILE].median())
    Z = (X - X.mean()) / (X.std() + 1e-9)
    if circuit_id not in Z.index:
        return pd.Series(dtype=float)
    d = np.sqrt(((Z - Z.loc[circuit_id]) ** 2).sum(1)).drop(circuit_id)
    return d.sort_values().head(n)
