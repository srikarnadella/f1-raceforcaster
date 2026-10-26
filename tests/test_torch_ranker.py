"""PyTorch listwise ranker (skipped when torch is not installed)."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from f1.analytics.torch_ranker import TorchNeuralPL  # noqa: E402


def _races(n_races=60, n=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        x = rng.normal(size=(n, 2))
        skill = 1.5 * x[:, 0] - 0.5 * x[:, 1] + np.sin(2 * x[:, 0])      # partly non-linear
        order = np.argsort(-(skill + rng.gumbel(size=n)))
        pos = np.empty(n, int)
        pos[order] = np.arange(1, n + 1)
        for i in range(n):
            rows.append({"season": 2024 + r // 30, "round": r % 30 + 1, "driver_id": f"d{i}", "a": x[i, 0],
                         "b": x[i, 1], "position": pos[i], "classified": True, "skill": skill[i]})
    return pd.DataFrame(rows)


def test_torch_ranker_learns_the_true_ordering():
    d = _races()
    m = TorchNeuralPL(["a", "b"], hidden=4, n_models=2, epochs=300).fit(d)
    test = _races(n_races=5, seed=1)
    corr = np.corrcoef(m.score(test), test["skill"])[0, 1]
    assert corr > 0.9


def test_torch_ranker_is_deterministic():
    d = _races(n_races=20)
    a = TorchNeuralPL(["a", "b"], hidden=4, n_models=1, epochs=100).fit(d).score(d)
    b = TorchNeuralPL(["a", "b"], hidden=4, n_models=1, epochs=100).fit(d).score(d)
    assert np.allclose(a, b)
