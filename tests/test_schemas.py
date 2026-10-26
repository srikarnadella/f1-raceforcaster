"""Data contracts reject bad tables (synthetic, no data store needed)."""
import pandas as pd
import pytest

from f1 import schemas


def _results():
    return pd.DataFrame({"season": [2026, 2026], "round": [1, 1], "driver_id": ["a", "b"],
                         "constructor_id": ["x", "x"], "circuit_id": ["c", "c"], "position": [1, 2],
                         "grid": [1, 2], "points": [25.0, 18.0], "quali_pos": [1.0, 2.0], "q1": [80.0, 80.5]})


def test_valid_results_pass():
    schemas.validate(_results(), "results")


@pytest.mark.parametrize("break_it, why", [
    (lambda d: d.assign(position=[1, 1]), "two winners"),
    (lambda d: pd.concat([d, d.iloc[[0]]]), "duplicate driver row"),
    (lambda d: d.assign(points=[25.0, -3.0]), "negative points"),
    (lambda d: d.assign(q1=[80.0, 1129.7]), "a 19-minute qualifying lap"),
])
def test_broken_results_fail(break_it, why):
    with pytest.raises(ValueError, match="data contract 'results' failed"):
        schemas.validate(break_it(_results()), "results")


def test_zone_shares_must_cover_the_lap():
    w = pd.DataFrame({"season": [2026], "round": [1], "share_straight": [0.5], "share_fast": [0.1],
                      "share_medium": [0.1], "share_slow": [0.1], "pit_loss_s": [22.0]})
    with pytest.raises(ValueError, match="weekend"):
        schemas.validate(w, "weekend")


def test_unknown_table_passes_through():
    df = pd.DataFrame({"x": [1]})
    assert schemas.validate(df, "no_such_contract") is df
