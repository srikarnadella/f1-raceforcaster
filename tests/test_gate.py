import numpy as np
import pandas as pd

from eval import gate


def _res(n, logloss, pb, tb, sp, ece=0.02, seed=0):
    rng = np.random.default_rng(seed)
    pr = pd.DataFrame({"season": 2025, "round": range(n),
                       "winner_logloss": logloss + rng.normal(0, 0.05, n), "podium_brier": pb + rng.normal(0, 0.002, n),
                       "top10_brier": tb + rng.normal(0, 0.002, n), "spearman": sp + rng.normal(0, 0.01, n)})
    return {"per_race": pr, "summary": pr.drop(columns=["season", "round"]).mean().to_dict(),
            "calibration": {"ece_win": ece, "ece_top3": ece, "ece_top10": ece}}


def test_identical_candidate_not_promoted():
    a = _res(100, 1.4, 0.07, 0.16, 0.65)
    d = gate.decide(a, a, run_leakage=False)
    assert not d["promote"] and d["composite"] == 0


def test_clearly_better_candidate_promoted():
    champ = _res(100, 1.4, 0.07, 0.16, 0.65, seed=1)
    cand = _res(100, 1.3, 0.068, 0.155, 0.67, seed=1)
    d = gate.decide(cand, champ, run_leakage=False)
    assert d["promote"], d


def test_regression_on_one_target_blocks():
    champ = _res(100, 1.4, 0.07, 0.16, 0.65, seed=1)
    cand = _res(100, 1.2, 0.075, 0.16, 0.65, seed=1)  # podium Brier 7% worse
    d = gate.decide(cand, champ, run_leakage=False)
    assert not d["checks"]["no_target_regression"] and not d["promote"]


def test_order_metrics_do_not_decide_promotion():
    # owner's objective (2026-10-25): winner and podium decide; top-10 / order are reported only
    champ = _res(100, 1.4, 0.07, 0.16, 0.65, seed=1)
    cand = _res(100, 1.2, 0.07, 0.18, 0.60, seed=1)   # better winner, worse top-10 and order
    d = gate.decide(cand, champ, run_leakage=False)
    assert d["checks"]["no_target_regression"] and d["checks"]["composite"]


def test_calibration_degradation_blocks():
    champ = _res(100, 1.4, 0.07, 0.16, 0.65, seed=1)
    cand = _res(100, 1.3, 0.068, 0.155, 0.67, ece=0.05, seed=1)
    assert not gate.decide(cand, champ, run_leakage=False)["checks"]["calibration"]


def test_generated_candidates_change_one_thing():
    from f1.improve import deep_merge, generate_candidates, pick_candidates
    from f1.models import DEFAULT_SPEC
    cands = generate_candidates(DEFAULT_SPEC)
    assert len(cands) > 15 and len({c["slug"] for c in cands}) == len(cands)
    for c in cands:
        assert deep_merge(DEFAULT_SPEC, c["patch"]) != DEFAULT_SPEC
    log = [{"slug": cands[0]["slug"], "attempt_key": f"{cands[0]['slug']}|v1|{273 // 6}"}]
    picked = {c["slug"] for c in pick_candidates("v1", DEFAULT_SPEC, 273, log) if c.get("auto")}
    assert cands[0]["slug"] not in picked                     # just tried: skipped
    picked_later = {c["slug"] for c in pick_candidates("v1", DEFAULT_SPEC, 273 + 6, log) if c.get("auto")}
    assert cands[0]["slug"] in picked_later                   # re-eligible after 6 more races


def test_candidates_identical_to_the_champion_are_skipped(tmp_path, monkeypatch):
    from f1 import improve
    from f1.models import DEFAULT_SPEC
    champ = improve.deep_merge(DEFAULT_SPEC, {"strength": {"adaptive_win_temperature": 24}})
    q = tmp_path / "queue.json"
    q.write_text(__import__("json").dumps([
        {"slug": "already-in-champion", "patch": {"strength": {"adaptive_win_temperature": 24}}},
        {"slug": "real-change", "patch": {"strength": {"l2": 3.0}}}]))
    monkeypatch.setattr(improve, "EXP_QUEUE", q)
    slugs = [c["slug"] for c in improve.pick_candidates("v3", champ, 270, [])]
    assert "already-in-champion" not in slugs and slugs[0] == "real-change"


def _scores(rows):
    return [{"season": 2026, "round": r, "model_version": v, "late": False, "role": "x",
             "metrics": {"winner_logloss": ll, "podium_brier": 0.06}} for r, v, ll in rows]


def test_no_rollback_before_three_live_races(monkeypatch):
    from f1 import improve, predlog, registry
    reg = {"champion": "v3", "shadow": "v1", "shadow_since": [2026, 19], "history": [], "versions": {}}
    bad = [(19, "v3", 2.0), (19, "v1", 0.5), (20, "v3", 2.0), (20, "v1", 0.5)]   # much worse, but only 2 races
    monkeypatch.setattr(predlog, "read_scores", lambda: _scores(bad))
    monkeypatch.setattr(registry, "save", lambda reg: None)
    assert improve.check_rollback(reg) is None and reg["champion"] == "v3"
    bad += [(21, "v3", 2.0), (21, "v1", 0.5)]                                     # third race: roll back
    monkeypatch.setattr(predlog, "read_scores", lambda: _scores(bad))
    monkeypatch.setattr(improve, "changelog", lambda lines: None)
    monkeypatch.setattr(improve, "append_jsonl", lambda *a: None)
    assert improve.check_rollback(reg) == "v3" and reg["champion"] == "v1"


def test_shadow_keeps_a_live_record(monkeypatch):
    from f1 import predlog, registry
    monkeypatch.setattr(predlog, "read_scores", lambda: _scores([(18, "v1", 0.7)]))
    reg = {"champion": "v1", "shadow": None, "history": []}
    registry.promote(reg, "v2", (2026, 19), "a")
    assert reg["shadow"] == "v1"            # v1 has a live race
    registry.promote(reg, "v3", (2026, 19), "b")
    assert reg["champion"] == "v3" and reg["shadow"] == "v1"   # v2 was never live: keep v1
