import numpy as np
import pandas as pd

from f1.features import OUTCOME_COLS
from f1.forecast import forecast
from f1.models import DEFAULT_SPEC, PlackettLuce, simulate_orders


def test_simulated_positions_are_permutations():
    pos = simulate_orders(np.linspace(2, 0, 20), np.full(20, 0.1), 500, np.random.default_rng(0))
    assert (np.sort(pos, 1) == np.arange(1, 21)).all()


def test_forecast_marginals_coherent(results):
    race = results[(results.season == 2025) & (results["round"] == 10)]
    pred, h = forecast(2025, 10, DEFAULT_SPEC, entry=race.drop(columns=OUTCOME_COLS), results=results)
    assert abs(pred.p_win.sum() - 1) < 1e-9
    assert abs(pred.p_top3.sum() - 3) < 1e-9
    assert abs(pred.p_top10.sum() - 10) < 1e-9
    assert (pred.p_win <= pred.p_top3 + 1e-12).all() and (pred.p_top3 <= pred.p_top10 + 1e-12).all()
    assert abs(pred.exp_pos.sum() - len(pred) * (len(pred) + 1) / 2) < 1e-6
    assert len(h) == 16


def test_plackett_luce_recovers_signal():
    rng = np.random.default_rng(1)
    rows = []
    for r in range(150):
        x = rng.normal(size=12)
        order = np.argsort(-(1.5 * x + rng.gumbel(size=12)))
        pos = np.empty(12, int)
        pos[order] = np.arange(1, 13)
        rows += [{"season": 2020, "round": r, "x": x[i], "position": pos[i], "classified": True} for i in range(12)]
    pl = PlackettLuce(["x"], l2=0.01, depth=12).fit(pd.DataFrame(rows))
    assert abs(pl.beta[0] / pl.sd[0] - 1.5) < 0.3
