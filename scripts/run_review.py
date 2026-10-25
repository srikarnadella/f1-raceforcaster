"""Trigger B: post-race review. Idempotent; exits 0 when results are not ready.

Usage: python scripts/run_review.py 2026_18 [--no-improve]
"""
import json
import sys

import _path  # noqa: F401

from f1 import championship, improve, ingest, registry, review, state
from f1.config import CHAMP_DIR

MIN_RESULT_ROWS = 18


def main(key: str, run_improve: bool = True):
    season, rnd = map(int, key.split("_"))
    if key in state.load().get("review", {}):
        print(f"{key}: already reviewed")
        return
    ingest.update(seasons=[season], current=season)
    results = ingest.load_results()
    race = results[(results.season == season) & (results["round"] == rnd)]
    if len(race) < MIN_RESULT_ROWS:
        print(f"{key}: results not available yet ({len(race)} rows); will retry next tick")
        return
    scores = review.score_logged(results, season, rnd)
    print(f"{key}: scored {len(scores)} logged forecast(s)")
    review.pace_report(results, season, rnd)

    # championship: append this round's title probabilities (keyed by round, idempotent)
    reg = registry.ensure_initialized()
    sim = championship.simulate(results, ingest.load_calendar(), season, rnd, registry.champion_spec(reg))
    CHAMP_DIR.mkdir(parents=True, exist_ok=True)
    path = CHAMP_DIR / f"{season}.json"
    hist = json.loads(path.read_text()) if path.exists() else {}
    hist[str(rnd)] = {**championship.to_json(sim), "model_version": reg["champion"]}
    path.write_text(json.dumps(hist, indent=1, default=float))

    rb = improve.check_rollback(reg)
    if rb:
        print(f"rolled back {rb}")
    if run_improve:
        cal = ingest.load_calendar()
        nxt = cal[(cal.season == season) & (cal["round"] > rnd)]
        nxt_key = (season, int(nxt["round"].iloc[0])) if len(nxt) else (season + 1, 1)
        print(json.dumps(improve.run_cycle(nxt_key), default=float))
    # refresh the walk-forward backtest of the (possibly new) champion on the grown data set
    import backtest
    from f1.config import REPORTS
    reg = registry.load()
    spec = registry.champion_spec(reg)
    backtest.write(spec, REPORTS / "backtest.json", version=reg["champion"])
    backtest.write(spec, REPORTS / f"backtest_{season}.json", test_from=season, version=reg["champion"])
    state.mark("review", season, rnd, "done")


if __name__ == "__main__":
    main(sys.argv[1], "--no-improve" not in sys.argv)
