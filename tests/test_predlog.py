from datetime import datetime, timezone

import pandas as pd
import pytest

from f1 import predlog


@pytest.fixture
def tmp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(predlog, "PRED_LOG", tmp_path / "log.jsonl")
    monkeypatch.setattr(predlog, "SCORES_LOG", tmp_path / "scores.jsonl")
    return tmp_path


def _pred():
    return pd.DataFrame({"driver_id": ["a", "b"], "code": ["A", "B"], "constructor_id": ["x", "y"],
                         "start_pos": [1, 2], "p_win": [0.7, 0.3], "p_top3": [1, 1], "p_top10": [1, 1],
                         "exp_pos": [1.3, 1.7], "pred_rank": [1, 2]})


KW = dict(season=2030, rnd=1, race_name="Test GP", race_utc=pd.Timestamp("2030-03-01T12:00:00Z"), fmt="conventional",
          role="champion", model_version="v1", git_commit="abc", data_hash="d", feature_hash="f")


def test_append_only_and_idempotent(tmp_log):
    now = datetime(2030, 3, 1, 10, tzinfo=timezone.utc)
    assert predlog.log_forecast(_pred(), now=now, **KW) is not None
    before = (tmp_log / "log.jsonl").read_text()
    assert predlog.log_forecast(_pred(), now=now, **KW) is None
    assert (tmp_log / "log.jsonl").read_text() == before
    predlog.log_forecast(_pred(), now=now, **{**KW, "model_version": "v2"})
    after = (tmp_log / "log.jsonl").read_text()
    assert after.startswith(before) and len(predlog.read_log()) == 2


def test_late_forecast_refused(tmp_log):
    late = datetime(2030, 3, 1, 12, 5, tzinfo=timezone.utc)
    with pytest.raises(predlog.LateForecast):
        predlog.log_forecast(_pred(), now=late, **KW)
    e = predlog.log_forecast(_pred(), now=late, allow_late=True, **KW)
    assert e["late"] is True
