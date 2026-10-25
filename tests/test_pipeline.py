from datetime import datetime, timezone

import numpy as np
import pandas as pd

from f1 import championship, ingest, state
from f1.ingest import lap_seconds
from f1.laps import categorise


def test_lap_seconds():
    assert lap_seconds("1:31.373") == 91.373
    assert lap_seconds("1:42:00.785") == 6120.785
    assert np.isnan(lap_seconds("")) and np.isnan(lap_seconds(None))


def test_store_integrity(results):
    assert results.groupby(["season", "round", "driver_id"]).size().max() == 1
    winners = results[results.position == 1].groupby(["season", "round"]).size()
    assert (winners == 1).all()
    assert results.quali_pos.notna().mean() > 0.98


def test_due_jobs():
    cal = ingest.load_calendar()
    ev = cal[(cal.season == 2026) & (cal["round"] == 18)].iloc[0]
    st = {"forecast": {}, "review": {}}
    before_race = (ev.race_utc - pd.Timedelta(hours=2)).to_pydatetime()
    assert state.due_jobs(cal, before_race, st)["forecast"] == (2026, 18)
    after = (ev.race_utc + pd.Timedelta(hours=4)).to_pydatetime()
    assert state.due_jobs(cal, after, st)["review"] == (2026, 18)
    assert state.due_jobs(cal, after, {"forecast": {}, "review": {"2026_18": "done"}})["review"] is None


def test_championship_final_round_is_certain(results):
    cal = ingest.load_calendar()
    sim = championship.simulate(results, cal, 2025, 24, {"strength": {"features": []}, "dnf": {"features": []}},
                                n_sims=100)
    assert sim["remaining"] == 0
    assert sim["drivers"].iloc[0]["p_title"] == 1.0


def test_lap_categories():
    rows = []
    for d, base in [("a", 90.0), ("b", 90.5)]:
        for lap in range(1, 21):
            rows.append({"driver_id": d, "lap": lap, "time_s": base + (30 if lap == 10 else 0) + (5 if lap == 1 else 0),
                         "position": 1 if d == "a" else 2, "pit_this_lap": lap == 10 and d == "a",
                         "stint": 1 + (lap > 10 and d == "a"), "tyre_age": lap})
    c = categorise(pd.DataFrame(rows)).set_index(["driver_id", "lap"])["category"]
    assert c["a", 1] == "lap1" and c["a", 10] == "in" and c["a", 11] == "out"


def test_update_completes_partial_round(tmp_path, monkeypatch):
    full = ingest.load_results(seasons=[2025])
    partial = full[~((full["round"] == 24) & (full["position"] > 10))]
    monkeypatch.setattr(ingest, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(ingest, "CALENDAR_CSV", tmp_path / "cal.csv")
    monkeypatch.setattr(ingest, "calendar_rows", lambda s, t: [])
    partial.to_csv(tmp_path / "2025.csv", index=False)
    monkeypatch.setattr(ingest, "season_results", lambda s, t=None: full)
    ingest.update(seasons=[2025], current=2025)
    out = pd.read_csv(tmp_path / "2025.csv")
    assert len(out) == len(full)
    pd.testing.assert_frame_equal(out[out["round"] < 24].reset_index(drop=True),
                                  partial[partial["round"] < 24].reset_index(drop=True), check_dtype=False)
