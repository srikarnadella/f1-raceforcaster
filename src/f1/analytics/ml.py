"""Machine learning on top of the analytics tables.

Supervised
  NeuralPL          listwise neural ranker: a small MLP scores every driver,
                    trained on the Plackett-Luce likelihood of the whole
                    finishing order (the "listwise ranking model" CLAUDE.md
                    section 5 names). Linear skip connection + tanh hidden
                    layer, L-BFGS, an ensemble of seeds. Drop-in for the
                    linear PlackettLuce inside models.RaceModel.
  race_pace_model   gradient boosting: projects each driver's Sunday race pace
                    gap from Saturday information (quali, practice long runs,
                    team degradation history, circuit type, temperature).
Unsupervised
  run_programs      Gaussian mixture on practice stints -> run programmes
                    (quali sim / race sim / short or aero runs).
  car_map           PCA of team zone DNA -> 2-D "car character" map.
  anomalies         isolation forest on team-weekend performance vectors ->
                    weekends where a team was far from its usual self.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from ..config import SEED


# ------------------------------------------------------------------ neural listwise ranker

def _pack(feats, features):
    df = feats[feats["classified"].astype(bool)].sort_values(["season", "round", "position"])
    X = df[features].to_numpy(float)
    keys = df["season"].to_numpy() * 100 + df["round"].to_numpy()
    _, starts, counts = np.unique(keys, return_index=True, return_counts=True)
    R, M, F = len(starts), counts.max(), X.shape[1]
    return df, X, starts, counts, R, M, F


class NeuralPL:
    def __init__(self, features, hidden=8, l2=1.0, l2_hidden=4.0, depth=3, season_half_life=3.0,
                 n_models=3, seed=SEED):
        self.features, self.h, self.l2, self.l2h = list(features), hidden, l2, l2_hidden
        self.depth, self.hl, self.n_models, self.seed = depth, season_half_life, n_models, seed

    def fit(self, feats: pd.DataFrame):
        from ..models import season_weights
        df, X, starts, counts, R, M, F = _pack(feats, self.features)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        X = (X - self.mu) / self.sd
        Xp = np.zeros((R, M, F))
        mask = np.zeros((R, M), bool)
        for i, (s, c) in enumerate(zip(starts, counts)):
            Xp[i, :c] = X[s:s + c]
            mask[i, :c] = True
        w = season_weights(df["season"].iloc[starts], int(df["season"].max()), self.hl)
        D = min(self.depth, M)
        tri = np.arange(M)[None, :] >= np.arange(D)[:, None]
        valid = mask[:, :D]
        H = self.h
        sizes = [F, F * H, H, H]  # skip w0, W1, b1, v

        def unpack(p):
            o = np.cumsum([0] + sizes)
            return p[o[0]:o[1]], p[o[1]:o[2]].reshape(F, H), p[o[2]:o[3]], p[o[3]:o[4]]

        n = w.sum()

        def f(p):
            w0, W1, b1, v = unpack(p)
            A = np.tanh(Xp @ W1 + b1)                                     # (R, M, H)
            s = Xp @ w0 + A @ v                                           # (R, M)
            sm = np.where(mask, s, -np.inf)
            z = np.where(tri[None], sm[:, None, :], -np.inf)
            lse = np.where(valid, logsumexp(z, axis=2), 0.0)
            ll = (np.where(valid, s[:, :D] - lse, 0.0).sum(1) * w).sum()
            P = np.exp(z - lse[:, :, None]) * valid[:, :, None]
            first = np.zeros_like(s)
            first[:, :D] = valid
            gs = (first - P.sum(1)) * w[:, None] * mask                   # d ll / d s
            g_w0 = np.einsum("rm,rmf->f", gs, Xp)
            g_v = np.einsum("rm,rmh->h", gs, A)
            gA = gs[:, :, None] * v[None, None, :] * (1 - A ** 2)          # (R, M, H)
            g_W1 = np.einsum("rmf,rmh->fh", Xp, gA)
            g_b1 = gA.sum((0, 1))
            reg = self.l2 * (w0 @ w0) + self.l2h * ((W1 ** 2).sum() + v @ v)
            loss = -ll / n + reg / n
            grad = np.concatenate([-g_w0 / n + 2 * self.l2 * w0 / n, (-g_W1 / n + 2 * self.l2h * W1 / n).ravel(),
                                   -g_b1 / n, -g_v / n + 2 * self.l2h * v / n])
            return loss, grad

        rng = np.random.default_rng(self.seed)
        self.params = []
        for _ in range(self.n_models):
            p0 = np.concatenate([np.zeros(F), rng.normal(0, 0.3, F * H), np.zeros(H), rng.normal(0, 0.3, H)])
            res = minimize(f, p0, jac=True, method="L-BFGS-B", options={"maxiter": 300})
            self.params.append(unpack(res.x))
        return self

    def score(self, race: pd.DataFrame) -> np.ndarray:
        X = (race[self.features].to_numpy(float) - self.mu) / self.sd
        return np.mean([X @ w0 + np.tanh(X @ W1 + b1) @ v for w0, W1, b1, v in self.params], axis=0)

    def coef(self) -> dict:
        w0 = np.mean(np.stack([p[0] for p in self.params]), axis=0)
        return {f: round(float(v), 4) for f, v in zip(self.features, w0)}


# ------------------------------------------------------------------ race pace projection

PACE_INPUTS = ["quali_gap_pct", "fp_longrun_gap", "team_deg_prior", "team_racepace_prior", "team_rmq_prior",
               "race_track_temp", "share_straight", "share_slow", "share_fast", "is_sprint"]


def race_pace_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Driver-race rows: Saturday inputs -> Sunday clean-air race pace gap (target).
    Team priors come from earlier weekends only."""
    ol, lr, rp, dg, wk = (tables[k] for k in ("one_lap", "long_runs", "race_pace", "degradation", "weekend"))
    keys = ["season", "round", "driver_id"]
    q = ol[ol["session"] == "Q"][keys + ["constructor_id", "gap_pct", "code"]].rename(columns={"gap_pct": "quali_gap_pct"})
    y = rp[rp["session"] == "R"][keys + ["race_pace_gap_pct"]]
    f = q.merge(y, on=keys, how="left")
    if len(lr):
        tot = lr.groupby(["season", "round", "session"])["long_laps"].sum().reset_index().sort_values("long_laps")
        main = tot.groupby(["season", "round"]).tail(1)[["season", "round", "session"]]
        m = lr.merge(main, on=["season", "round", "session"])[keys + ["long_run_gap_pct"]]
        f = f.merge(m.rename(columns={"long_run_gap_pct": "fp_longrun_gap"}), on=keys, how="left")
    # team priors: EWM over earlier weekends (shifted so a weekend never sees itself)
    team = f.groupby(["season", "round", "constructor_id"]).agg(t_rp=("race_pace_gap_pct", "min"),
                                                                t_q=("quali_gap_pct", "min")).reset_index()
    team["t_rmq"] = team["t_rp"] - team["t_q"]
    rdeg = dg[dg["session"] == "R"].groupby(["season", "round", "constructor_id"])["deg_s_per_lap"].mean().rename("t_deg")
    team = team.merge(rdeg.reset_index(), on=["season", "round", "constructor_id"], how="left").sort_values(["season", "round"])
    for src, dst in [("t_rp", "team_racepace_prior"), ("t_rmq", "team_rmq_prior"), ("t_deg", "team_deg_prior")]:
        team[dst] = team.groupby("constructor_id")[src].transform(lambda s: s.shift(1).ewm(halflife=4, ignore_na=True).mean())
    f = f.merge(team[["season", "round", "constructor_id", "team_racepace_prior", "team_rmq_prior", "team_deg_prior"]],
                on=["season", "round", "constructor_id"], how="left")
    w = wk[["season", "round", "circuit_id", "format", "race_track_temp", "share_straight", "share_slow", "share_fast"]].copy()
    w["is_sprint"] = (w["format"] == "sprint").astype(float)
    # track temperature on race day is a forecastable condition; use the quali value as the Saturday proxy
    w["race_track_temp"] = wk["quali_track_temp"].fillna(wk["race_track_temp"])
    return f.merge(w.drop(columns="format"), on=["season", "round"], how="left").sort_values(["season", "round"])


def race_pace_model(frame: pd.DataFrame, test_from: tuple[int, int] = (2024, 1)):
    """Walk-forward by weekend. Returns (predictions, metrics vs the 'race pace =
    quali gap' baseline)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    d = frame[frame["race_pace_gap_pct"].notna() & (frame["race_pace_gap_pct"] < 5) & (frame["quali_gap_pct"] < 5)]
    keys = d[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    keys = keys[(keys["season"] > test_from[0]) | ((keys["season"] == test_from[0]) & (keys["round"] >= test_from[1]))]
    preds = []
    for s, r in keys.itertuples(index=False):
        tr = d[(d["season"] < s) | ((d["season"] == s) & (d["round"] < r))]
        te = d[(d["season"] == s) & (d["round"] == r)]
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=0)
        m.fit(tr[PACE_INPUTS], tr["race_pace_gap_pct"])
        p = te[["season", "round", "driver_id", "code", "constructor_id", "quali_gap_pct", "race_pace_gap_pct"]].copy()
        p["pred"] = m.predict(te[PACE_INPUTS])
        p["pred"] -= p["pred"].min()
        preds.append(p)
    P = pd.concat(preds, ignore_index=True)

    def stats(col):
        err = (P[col] - P["race_pace_gap_pct"]).abs().mean()
        sp = np.nanmean([g[col].corr(g["race_pace_gap_pct"], method="spearman")
                         for _, g in P.groupby(["season", "round"])])
        return {"mae_pct": round(float(err), 3), "spearman": round(float(sp), 3)}
    return P, {"gbm": stats("pred"), "quali_gap_baseline": stats("quali_gap_pct"), "races": int(len(keys))}


def fit_race_pace(frame: pd.DataFrame):
    from sklearn.ensemble import HistGradientBoostingRegressor
    d = frame[frame["race_pace_gap_pct"].notna() & (frame["race_pace_gap_pct"] < 5) & (frame["quali_gap_pct"] < 5)]
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=0).fit(d[PACE_INPUTS], d["race_pace_gap_pct"])


# ------------------------------------------------------------------ unsupervised

def run_programs(laps: pd.DataFrame, k: int = 3, seed: int = 0) -> pd.DataFrame:
    """Cluster practice stints by (pace laps, pace vs driver best, lap-time spread,
    tyre age at start). Clusters are named from their centroids."""
    from sklearn.mixture import GaussianMixture
    p = laps[laps["session"].isin(["FP1", "FP2", "FP3"]) & laps["lap_s"].notna() & ~laps["pit_out"].astype(bool)]
    best = p.groupby(["session", "driver_id"])["lap_s"].transform("min")
    p = p.assign(ratio=p["lap_s"] / best)
    p = p[p["ratio"] < 1.10]
    st = p.groupby(["season", "round", "session", "driver_id", "code", "stint"]).agg(
        n=("lap", "size"), pace=("ratio", "median"), spread=("ratio", "std"), age0=("tyre_age", "min"),
        compound=("compound", "first")).reset_index()
    st = st[st["n"] >= 2].copy()
    if len(st) < 3 * k:
        st["programme"] = "unknown"
        return st
    Z = np.column_stack([np.log(st["n"]), st["pace"], st["spread"].fillna(0), st["age0"].fillna(0).clip(upper=20)])
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    lab = GaussianMixture(k, random_state=seed, n_init=3).fit_predict(Z)
    st["cl"] = lab
    c = st.groupby("cl").agg(n=("n", "mean"), pace=("pace", "mean"))
    race = c["n"].idxmax()
    quali = c.drop(race)["pace"].idxmin()
    names = {race: "race sim", quali: "quali sim"}
    st["programme"] = st["cl"].map(names).fillna("short / setup runs")
    return st.drop(columns="cl")


def car_map(tw: pd.DataFrame, season: int) -> pd.DataFrame:
    """PCA of each team's season-average zone DNA."""
    from sklearn.decomposition import PCA
    cols = ["z_straight", "z_fast", "z_medium", "z_slow"]
    t = tw[tw["season"] == season].sort_values(["season", "round"]).groupby(["constructor_id"]).agg(
        {**{c: "mean" for c in cols}, "team_name": "last", "team_colour": "last", "quali_gap_pct": "mean"}).dropna(subset=cols)
    if len(t) < 3:
        return pd.DataFrame()
    Z = t[cols].to_numpy()
    pca = PCA(2).fit(Z - Z.mean(0))
    xy = pca.transform(Z - Z.mean(0))
    t["pc1"], t["pc2"] = xy[:, 0], xy[:, 1]
    t.attrs["loadings"] = {f"pc{i + 1}": dict(zip(cols, np.round(pca.components_[i], 3))) for i in range(2)}
    return t.reset_index()


def anomalies(tw: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """Isolation forest on how a team's weekend differed from its rolling norm."""
    from sklearn.ensemble import IsolationForest
    t = tw.sort_values(["season", "round"]).copy()
    cols = ["quali_gap_pct", "race_pace_gap_pct", "longrun_gap_pct"]
    cols = [c for c in cols if c in t.columns]
    for c in cols:
        norm = t.groupby(["season", "constructor_id"])[c].transform(lambda s: s.shift(1).rolling(5, min_periods=2).median())
        t[f"d_{c}"] = t[c] - norm
    dcols = [f"d_{c}" for c in cols]
    m = t.dropna(subset=dcols[:1]).copy()
    X = m[dcols].fillna(0).to_numpy()
    iso = IsolationForest(contamination=contamination, random_state=0).fit(X)
    m["anomaly_score"] = -iso.score_samples(X)
    m["anomaly"] = iso.predict(X) == -1
    return m


def race_pace_shap(frame: pd.DataFrame, sample: int = 600, seed: int = 0) -> pd.DataFrame | None:
    """SHAP values of the boosted race-pace model (fitted on every race with a known
    outcome): how much each Saturday input pushed each driver's projected race-pace gap
    up or down. Long table: season, round, driver_id, code, feature, value, shap."""
    try:
        import shap
    except ImportError:
        return None
    d = frame[frame["race_pace_gap_pct"].notna() & (frame["race_pace_gap_pct"] < 5) & (frame["quali_gap_pct"] < 5)]
    if len(d) < 200:
        return None
    m = fit_race_pace(frame)
    d = d.sample(min(sample, len(d)), random_state=seed)
    sv = shap.TreeExplainer(m).shap_values(d[PACE_INPUTS])
    keys = d[["season", "round", "driver_id", "code"]].reset_index(drop=True)
    vals = d[PACE_INPUTS].reset_index(drop=True)
    rows = [keys.assign(feature=f, value=vals[f], shap=sv[:, i]) for i, f in enumerate(PACE_INPUTS)]
    return pd.concat(rows, ignore_index=True)
