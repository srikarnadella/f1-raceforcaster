"""Race simulator lab. Calibrate the simulator's parameters with Optuna (TPE,
seeded) on the calibration seasons only, then score it walk-forward on the
seasons after them with the frozen harness, against the champion, a
champion+simulator blend and the grid baseline. Every trial is an MLflow run.

Writes reports/race_sim_lab.json. Usage: python scripts/race_sim_lab.py
"""
import argparse
import json
import warnings
from dataclasses import asdict, replace


from eval.harness import walk_forward
from eval.metrics import summarize
import optuna

from f1 import ingest, registry, tracking
from f1.analytics.racesim import SimParams
from f1.analytics.simmodel import Blend, RaceSimModel, SimInputs
from f1.config import REPORTS
from f1.features import build_features
from f1.models import BASELINES, RaceModel

COLS = ["n_races", "winner_logloss", "winner_brier", "top1_hit", "podium_brier", "top10_brier", "spearman", "mae_pos"]


def score(per_race):
    s = summarize(per_race)
    return {k: float(s[k]) for k in COLS if k in s}


def objective(s):
    return s["winner_logloss"] - s["spearman"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="2023-2024", help="calibration seasons, e.g. 2023-2024")
    ap.add_argument("--trials", type=int, default=40)
    a = ap.parse_args()
    c_from, c_to = map(int, a.calib.split("-"))
    test_from = c_to + 1
    warnings.simplefilter("ignore")
    reg = registry.ensure_initialized()
    spec = registry.champion_spec(reg)
    res = ingest.load_results()
    feats = build_features(res, **spec.get("form", {}))
    inp = SimInputs(results=res)
    print(f"calibrating on {c_from}-{c_to} with Optuna ({a.trials} trials) ...", flush=True)
    trials = []

    def objective_fn(trial):
        prm = SimParams(w_quali=trial.suggest_float("w_quali", 0.3, 1.0),
                        pace_unc=trial.suggest_float("pace_unc", 0.15, 1.2, log=True),
                        c1=trial.suggest_float("c1", 0.5, 4.0),
                        c2=trial.suggest_float("c2", 0.5, 5.0),
                        start_noise=trial.suggest_float("start_noise", 0.3, 1.2),
                        lap_noise=trial.suggest_float("lap_noise", 0.2, 0.6), n_sims=400)
        pr, _ = walk_forward(lambda: RaceSimModel(inp, prm), feats, test_from=c_from, test_to=c_to)
        s = score(pr)
        trials.append({**trial.params, **s})
        with tracking.run("race-simulator-calibration", f"{a.calib}-trial-{trial.number}", params=trial.params,
                          tags={"calibration_seasons": a.calib}) as log:
            log(s)
        return objective(s)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective_fn, n_trials=a.trials)
    best = (SimParams(**study.best_params, n_sims=400), None)
    print("best trial", study.best_trial.number, study.best_params, round(study.best_value, 4), flush=True)
    prm = replace(best[0], n_sims=1000)
    # blend weight, also chosen on 2023
    blend_trials = {}
    for w in (0.25, 0.5, 0.75):
        pr, _ = walk_forward(lambda w=w: Blend(RaceSimModel(inp, prm), RaceModel(spec), w), feats,
                             test_from=c_from, test_to=c_to)
        blend_trials[w] = score(pr)
    w_best = min(blend_trials, key=lambda w: objective(blend_trials[w]))
    print("best", asdict(prm), "blend w", w_best, flush=True)
    # out-of-sample: the seasons after calibration
    models = {
        "grid baseline": lambda: BASELINES["grid"](),
        "champion (Plackett-Luce)": lambda: RaceModel(spec),
        "race simulator": lambda: RaceSimModel(inp, prm),
        f"blend ({w_best:.0%} simulator)": lambda: Blend(RaceSimModel(inp, prm), RaceModel(spec), w_best),
    }
    test, per = {}, {}
    for name, f in models.items():
        pr, preds = walk_forward(f, feats, test_from=test_from)
        test[name] = {f"{test_from}-2026": score(pr), "2026": score(pr[pr["season"] == 2026])}
        per[name] = pr
        print(name, test[name][f"{test_from}-2026"], flush=True)
    out = {"note": f"Simulator parameters and blend weight chosen on {c_from}-{c_to} races only; scores are "
                   f"walk-forward on {test_from} onward (frozen harness), each race predicted from earlier races. "
                   "Lower log-loss / Brier is better; higher Spearman / top-1 hit is better.",
           "calibration_seasons": [c_from, c_to],
           "params": asdict(prm), "blend_weight": w_best, "calibration_trials": trials, "optimizer": "optuna TPE (seed 0)",
           "blend_trials": {str(k): v for k, v in blend_trials.items()}, "test": test}
    path = REPORTS / "race_sim_lab.json"
    allruns = json.loads(path.read_text()) if path.exists() else {}
    if "runs" not in allruns:
        allruns = {"runs": {"2023-2023": allruns}} if allruns else {"runs": {}}
    allruns["runs"][a.calib] = out
    path.write_text(json.dumps(allruns, indent=1, default=float))


if __name__ == "__main__":
    main()
