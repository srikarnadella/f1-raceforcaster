"""Models: one pace score per driver -> finishing-order simulation -> every target.

- PlackettLuce: linear score s = x.w fit by maximum likelihood on the finishing
  order of classified drivers (optionally only the first `depth` stages).
- DNFModel: logistic regression for P(not classified).
- simulate: Gumbel-max sampling of Plackett-Luce orders for survivors, DNFs at
  the back. Win, podium, top-10, expected position and the championship all come
  from these samples, so they are coherent by construction.
- Baselines (uniform, pole, grid) share the same predict() output shape.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

from .config import SEED
from .features import DNF_FEATURES, FORM_ONLY_FEATURES, STRENGTH_FEATURES

DEFAULT_SPEC: dict[str, Any] = {
    "name": "pl-dnf",
    "strength": {"features": STRENGTH_FEATURES, "l2": 1.0, "depth": 3, "season_half_life": 3.0},
    "dnf": {"features": DNF_FEATURES, "C": 1.0, "season_half_life": 4.0},
    "form": {"form_half_life": 6.0, "dnf_half_life": 12.0},
    "n_sims": 10000,
}


def season_weights(seasons: pd.Series, latest: int, half_life: float | None) -> np.ndarray:
    if not half_life:
        return np.ones(len(seasons))
    return 0.5 ** ((latest - seasons.to_numpy()) / half_life)


def era_weights(seasons: pd.Series, era_weight: float | None) -> np.ndarray:
    """Down-weight races from earlier regulation eras: era_weight ** (eras back from the
    latest training era). 1 = no separation, 0 = train on the latest era only."""
    from .config import era_of
    if era_weight is None or era_weight == 1:
        return np.ones(len(seasons))
    eras = sorted({era_of(int(x)) for x in seasons})
    back = np.array([len(eras) - 1 - eras.index(era_of(int(x))) for x in seasons])
    return np.where(back == 0, 1.0, float(era_weight) ** back)


class PlackettLuce:
    def __init__(self, features, l2=1.0, depth=20, season_half_life=None, era_weight=None):
        self.features, self.l2, self.depth, self.hl = list(features), l2, depth, season_half_life
        self.era_weight = era_weight

    def fit(self, feats: pd.DataFrame):
        df = feats[feats["classified"].astype(bool)].sort_values(["season", "round", "position"])
        X = df[self.features].to_numpy(float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        X = (X - self.mu) / self.sd
        keys = df["season"].to_numpy() * 100 + df["round"].to_numpy()
        _, starts, counts = np.unique(keys, return_index=True, return_counts=True)
        R, M, F = len(starts), counts.max(), X.shape[1]
        Xp = np.zeros((R, M, F))
        mask = np.zeros((R, M), bool)
        for i, (s, c) in enumerate(zip(starts, counts)):
            Xp[i, :c] = X[s:s + c]
            mask[i, :c] = True
        latest = int(df["season"].max())
        w = season_weights(df["season"].iloc[starts], latest, self.hl) * era_weights(df["season"].iloc[starts],
                                                                                     self.era_weight)
        D = min(self.depth, M)
        tri = np.arange(M)[None, :] >= np.arange(D)[:, None]            # (D, M): j >= k
        valid = mask[:, :D]                                               # stage k exists

        def f(beta):
            s = Xp @ beta                                                 # (R, M)
            sm = np.where(mask, s, -np.inf)
            z = np.where(tri[None], sm[:, None, :], -np.inf)              # (R, D, M)
            lse = logsumexp(z, axis=2)                                    # (R, D)
            lse = np.where(valid, lse, 0.0)
            ll = (np.where(valid, s[:, :D] - lse, 0.0).sum(1) * w).sum()
            P = np.exp(z - lse[:, :, None]) * valid[:, :, None]           # (R, D, M)
            first = np.zeros_like(s)
            first[:, :D] = valid
            gs = (first - P.sum(1)) * w[:, None]                          # d ll / d s
            grad = np.einsum("rm,rmf->f", gs, Xp)
            n = w.sum()
            return -ll / n + self.l2 * beta @ beta / n, -grad / n + 2 * self.l2 * beta / n

        res = minimize(f, np.zeros(F), jac=True, method="L-BFGS-B")
        self.beta = res.x
        return self

    def score(self, race: pd.DataFrame) -> np.ndarray:
        X = (race[self.features].to_numpy(float) - self.mu) / self.sd
        return X @ self.beta

    def coef(self) -> dict:
        return dict(zip(self.features, np.round(self.beta, 4)))


class DNFModel:
    def __init__(self, features, C=1.0, season_half_life=None):
        self.features, self.C, self.hl = list(features), C, season_half_life

    def fit(self, feats: pd.DataFrame):
        y = (~feats["classified"].astype(bool)).astype(int)
        w = season_weights(feats["season"], int(feats["season"].max()), self.hl)
        self.m = LogisticRegression(C=self.C, max_iter=1000).fit(feats[self.features].to_numpy(float), y,
                                                                   sample_weight=w)
        return self

    def predict(self, race: pd.DataFrame) -> np.ndarray:
        return self.m.predict_proba(race[self.features].to_numpy(float))[:, 1]


def simulate_orders(scores, p_dnf, n_sims, rng, win_temperature: float = 1.0) -> np.ndarray:
    """(n_sims, n) array of finishing positions (1 = winner).

    win_temperature != 1 draws the winner from scores x win_temperature and the
    rest of the order from the plain scores, so only the first stage is
    sharpened; every target still comes from the same sampled orders."""
    n = len(scores)
    key = scores[None, :] + rng.gumbel(size=(n_sims, n))
    dnf = rng.random((n_sims, n)) < p_dnf[None, :]
    if win_temperature != 1.0:
        wkey = np.where(dnf, -np.inf, scores[None, :] * win_temperature + rng.gumbel(size=(n_sims, n)))
        key[np.arange(n_sims), wkey.argmax(1)] = 1e6
    key = np.where(dnf, -1e6 + rng.random((n_sims, n)), key)
    order = np.argsort(-key, axis=1)
    pos = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    pos[rows, order] = np.arange(1, n + 1)[None, :]
    return pos


def summarize(race: pd.DataFrame, pos: np.ndarray) -> pd.DataFrame:
    out = race[["season", "round", "driver_id", "code", "constructor_id", "start_pos"]].copy()
    out["p_win"] = (pos == 1).mean(0)
    out["p_top3"] = (pos <= 3).mean(0)
    out["p_top10"] = (pos <= 10).mean(0)
    out["exp_pos"] = pos.mean(0)
    return _rank(out)


def _rank(out):
    out = out.sort_values(["exp_pos", "p_win"], ascending=[True, False]).reset_index(drop=True)
    out["pred_rank"] = np.arange(1, len(out) + 1)
    return out


class RaceModel:
    """Champion-style model: Plackett-Luce strength + DNF model + simulation."""

    def __init__(self, spec: dict[str, Any] | None = None):
        self.spec: dict[str, Any] = spec or DEFAULT_SPEC

    def fit(self, feats: pd.DataFrame):
        if self.spec.get("by_era"):
            # separate models per regulation era: fitted lazily for the era of the race asked for
            self._train = feats
            self._sub: dict[str, RaceModel] = {}
            base = {k: v for k, v in self.spec.items() if k != "by_era"}
            self._base = RaceModel(base).fit(feats)
            return self
        s, d = self.spec["strength"], self.spec["dnf"]
        if s.get("hidden"):
            ranker: type
            if s.get("backend") == "torch":
                from .analytics.torch_ranker import TorchNeuralPL
                ranker = TorchNeuralPL
            else:
                from .analytics.ml import NeuralPL
                ranker = NeuralPL
            self.strength = ranker(s["features"], s["hidden"], s["l2"], s.get("l2_hidden", 4.0), s["depth"],
                                   s.get("season_half_life"), s.get("n_models", 3)).fit(feats)
        else:
            self.strength = PlackettLuce(s["features"], s["l2"], s["depth"], s.get("season_half_life"),
                                         s.get("era_weight")).fit(feats)
        self.dnf = DNFModel(d["features"], d["C"], d.get("season_half_life")).fit(feats)
        self.win_temperature = s.get("win_temperature", 1.0)
        if s.get("adaptive_win_temperature"):
            self.win_temperature = self._fit_win_temperature(feats, int(s["adaptive_win_temperature"]))
        return self

    def _fit_win_temperature(self, feats: pd.DataFrame, window: int) -> float:
        """How sharp should the winner draw be right now? Pick the temperature that
        best explains who won the last `window` training races (winner probability
        = softmax of temperature x score). Uses training races only, so it adapts
        to how predictable the current era is (2023 and 2026 sharper than 2021)."""
        tr = feats[feats["position"].notna()]
        keys = tr[["season", "round"]].drop_duplicates().tail(window)
        recent = tr.merge(keys, on=["season", "round"])
        grid = np.arange(0.8, 3.01, 0.1)
        ll = np.zeros(len(grid))
        for _, g in recent.groupby(["season", "round"]):
            if not (g["position"] == 1).any():
                continue
            sc = self.strength.score(g)
            w = int(np.flatnonzero(g["position"].to_numpy() == 1)[0])
            for i, t in enumerate(grid):
                z = t * sc
                ll[i] += z[w] - logsumexp(z)
        return float(grid[int(np.argmax(ll))])

    def predict(self, race: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
        if self.spec.get("by_era"):
            from .config import era_of
            era = str(era_of(int(race["season"].iloc[0])))
            patch = self.spec["by_era"].get(era)
            if not patch:
                return self._base.predict(race, seed)
            if era not in self._sub:
                base = {k: v for k, v in self.spec.items() if k != "by_era"}
                self._sub[era] = RaceModel(_merge(base, patch)).fit(self._train)
            return self._sub[era].predict(race, seed)
        rng = np.random.default_rng(seed + int(race["season"].iloc[0]) * 100 + int(race["round"].iloc[0]))
        # temperature > 1 sharpens the order distribution (the backtest found the favourite under-confident)
        sc = self.strength.score(race) * self.spec["strength"].get("temperature", 1.0)
        pd_ = self.dnf.predict(race)
        out = summarize(race, simulate_orders(sc, pd_, self.spec["n_sims"], rng,
                                              getattr(self, "win_temperature", 1.0)))
        extra = pd.DataFrame({"driver_id": race["driver_id"].to_numpy(), "score": sc, "p_dnf": pd_})
        out = out.merge(extra, on="driver_id")
        sim = self.spec.get("simulator")
        if sim and int(race["season"].iloc[0]) >= sim.get("from_season", 2023):
            out = _blend_simulator(out, race, sim, seed)
        return out


_SIM_INPUTS = None


def _blend_simulator(out: pd.DataFrame, race: pd.DataFrame, sim: dict, seed: int) -> pd.DataFrame:
    """Average the champion's probabilities with the lap-by-lap race simulator's
    (analytics/racesim.py), weight `sim["weight"]` on the simulator. Only where
    driver profiles exist (OpenF1 era); inputs come from earlier races only."""
    global _SIM_INPUTS
    from .analytics.racesim import SimParams
    from .analytics.simmodel import RaceSimModel, SimInputs
    if _SIM_INPUTS is None:
        _SIM_INPUTS = SimInputs()
    prm = SimParams(**{k: v for k, v in sim.items() if k in SimParams.__dataclass_fields__})
    sp = RaceSimModel(_SIM_INPUTS, prm).predict(race, seed)
    w = float(sim.get("weight", 0.5))
    m = out.merge(sp[["driver_id", "p_win", "p_top3", "p_top10", "exp_pos"]], on="driver_id", suffixes=("", "_sim"))
    for c in ("p_win", "p_top3", "p_top10", "exp_pos"):
        m[c] = (1 - w) * m[c] + w * m[f"{c}_sim"]
    m = m.drop(columns=[c for c in m.columns if c.endswith("_sim")])
    return _rank(m)


def _merge(base: dict, patch: dict) -> dict:
    import copy
    out = copy.deepcopy(base)
    for k, v in patch.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def form_only_spec(spec: dict) -> dict:
    """Variant that scores drivers without weekend data (for future races)."""
    s = {k: v for k, v in spec.items() if k not in ("simulator", "by_era")}  # future races: no grid, no qualifying
    s["strength"] = {**spec["strength"],
                     "features": [f for f in spec["strength"]["features"] if f in FORM_ONLY_FEATURES]}
    s["dnf"] = {**spec["dnf"], "features": [f for f in spec["dnf"]["features"] if f != "log_start"]}
    return s


# ---------------------------------------------------------------- baselines

class UniformBaseline:
    def fit(self, feats):
        return self

    def predict(self, race, seed=SEED):
        n = len(race)
        out = race[["season", "round", "driver_id", "code", "constructor_id", "start_pos"]].copy()
        out["p_win"], out["p_top3"], out["p_top10"] = 1 / n, min(3 / n, 1), min(10 / n, 1)
        out["exp_pos"] = (n + 1) / 2
        out = out.sample(frac=1, random_state=seed).reset_index(drop=True)  # no information in the order
        out["pred_rank"] = np.arange(1, n + 1)
        return out


class GridBaseline:
    """Empirical finishing distribution given start position (Laplace-smoothed),
    estimated on training races only. Predicted order = start order."""

    def __init__(self, max_pos=24):
        self.max_pos = max_pos

    def fit(self, feats):
        K = self.max_pos
        t = np.ones((K, K)) * 0.5
        st = feats["start_pos"].clip(1, K).astype(int).to_numpy() - 1
        fp = feats["position"].clip(1, K).astype(int).to_numpy() - 1
        np.add.at(t, (st, fp), 1)
        self.table = t / t.sum(1, keepdims=True)
        return self

    def predict(self, race, seed=SEED):
        n = len(race)
        dist = self.table[race["start_pos"].clip(1, self.max_pos).astype(int).to_numpy() - 1][:, :n]
        dist = dist / dist.sum(1, keepdims=True)
        out = race[["season", "round", "driver_id", "code", "constructor_id", "start_pos"]].copy()
        out["p_win"] = dist[:, 0] / dist[:, 0].sum()
        out["p_top3"] = dist[:, :3].sum(1)
        out["p_top10"] = dist[:, :10].sum(1)
        out["exp_pos"] = out["start_pos"].to_numpy(float)
        out = out.sort_values("start_pos", kind="stable").reset_index(drop=True)
        out["pred_rank"] = np.arange(1, n + 1)
        return out


class PoleBaseline(GridBaseline):
    """'Pole sitter wins': P(win) = historical pole win rate for the pole sitter,
    remainder spread evenly. Other targets as the grid baseline."""

    def predict(self, race, seed=SEED):
        out = super().predict(race, seed)
        n = len(out)
        p = self.table[0, 0] / self.table[0, :n].sum()
        out["p_win"] = np.where(out["pred_rank"] == 1, p, (1 - p) / (n - 1))
        return out


BASELINES = {"uniform": UniformBaseline, "grid": GridBaseline, "pole": PoleBaseline}
