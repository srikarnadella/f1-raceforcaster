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
