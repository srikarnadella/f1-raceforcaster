"""PyTorch listwise ranker: the same model as ml.NeuralPL, trained with autograd.

score(driver) = x.w0 + v . tanh(W1 x + b1)   (linear skip + one hidden layer)
loss          = - sum over races and the first `depth` finishing places of
                log P(that driver | the drivers still unplaced)   (Plackett-Luce)
                + L2 on the skip and hidden weights, races weighted by season age.

The stage log-normalisers come from one reverse cumulative logsumexp over the
finishing order, so a whole season of races trains as one batched tensor op.
Deterministic (fixed seeds, deterministic algorithms, CPU). Selected in a spec
with strength.backend = "torch"; torch is optional (requirements-torch.txt).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SEED


class TorchNeuralPL:
    def __init__(self, features, hidden=8, l2=1.0, l2_hidden=4.0, depth=3, season_half_life=3.0,
                 n_models=3, seed=SEED, epochs=400, lr=0.03):
        self.features, self.h, self.l2, self.l2h = list(features), hidden, l2, l2_hidden
        self.depth, self.hl, self.n_models, self.seed = depth, season_half_life, n_models, seed
        self.epochs, self.lr = epochs, lr

    def fit(self, feats: pd.DataFrame):
        import torch
        from ..models import season_weights
        from .ml import _pack
        torch.use_deterministic_algorithms(True)
        df, X, starts, counts, R, M, F = _pack(feats, self.features)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        X = (X - self.mu) / self.sd
        Xp = np.zeros((R, M, F), np.float32)
        mask = np.zeros((R, M), bool)
        for i, (s, c) in enumerate(zip(starts, counts)):
            Xp[i, :c] = X[s:s + c]
            mask[i, :c] = True
        w = season_weights(df["season"].iloc[starts], int(df["season"].max()), self.hl).astype(np.float32)
        Xt, mt, wt = torch.from_numpy(Xp), torch.from_numpy(mask), torch.from_numpy(w)
        D = min(self.depth, M)
        n = float(w.sum())
        self.nets = []
        for k in range(self.n_models):
            torch.manual_seed(self.seed % 2**31 + k)
            skip = torch.nn.Linear(F, 1, bias=False)
            torch.nn.init.zeros_(skip.weight)
            hid = torch.nn.Linear(F, self.h)
            out = torch.nn.Linear(self.h, 1, bias=False)
            for p in (hid.weight, out.weight):
                torch.nn.init.normal_(p, 0, 0.3)
            torch.nn.init.zeros_(hid.bias)
            params = [*skip.parameters(), *hid.parameters(), *out.parameters()]
            opt = torch.optim.Adam(params, lr=self.lr)
            for _ in range(self.epochs):
                opt.zero_grad()
                s = (skip(Xt) + out(torch.tanh(hid(Xt)))).squeeze(-1)           # (R, M), rows in finishing order
                s = torch.where(mt, s, torch.full_like(s, -1e9))
                # log of sum over drivers at or behind each place: reverse cumulative logsumexp
                lse = torch.flip(torch.logcumsumexp(torch.flip(s, [1]), dim=1), [1])
                ll = ((s - lse)[:, :D] * mt[:, :D]).sum(1)
                loss = -(ll * wt).sum() / n + (self.l2 * skip.weight.pow(2).sum()
                                               + self.l2h * (hid.weight.pow(2).sum() + out.weight.pow(2).sum())) / n
                loss.backward()
                opt.step()
            self.nets.append((skip, hid, out))
        return self

    def score(self, race: pd.DataFrame) -> np.ndarray:
        import torch
        X = torch.from_numpy(((race[self.features].to_numpy(float) - self.mu) / self.sd).astype(np.float32))
        with torch.no_grad():
            s = [(skip(X) + out(torch.tanh(hid(X)))).squeeze(-1).numpy() for skip, hid, out in self.nets]
        return np.mean(s, axis=0).astype(float)

    def coef(self) -> dict:
        w0 = np.mean([skip.weight.detach().numpy()[0] for skip, _, _ in self.nets], axis=0)
        return {f: round(float(v), 4) for f, v in zip(self.features, w0)}
